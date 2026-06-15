package fusion

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/bpross/fuse-sidecar/internal/config"
	"github.com/bpross/fuse-sidecar/internal/providers"
)

// Sink is what the pipeline emits into. The server adapts this to SSE.
//
// The pipeline calls these methods from a single goroutine; implementations
// do not need to be concurrent-safe across calls but must be safe relative
// to whatever heartbeat or background work the implementation itself owns.
type Sink interface {
	Progress(text string) error
	Content(text string) error
	ToolCallDelta(d providers.ToolCallDelta) error
	Done(finishReason string) error
}

// Decision is the outcome of one pipeline run; the server uses it to
// produce a snapshot and metrics.
type Decision struct {
	Kind           DecisionKind
	FallbackReason string
	Panel          []PanelResult
	JudgeLatency   time.Duration
	JudgeAnalysis  map[string]any
	TotalLatency   time.Duration
}

// DecisionKind classifies how a request was served.
type DecisionKind string

const (
	DecisionPassthrough DecisionKind = "passthrough"
	DecisionFusion      DecisionKind = "fusion"
	DecisionFallback    DecisionKind = "fallback"
)

// PanelResult is a snapshot-friendly summary of one panel call.
type PanelResult struct {
	Provider  string
	Model     string
	LatencyMs int64
	OK        bool
	Error     string
}

// Pipeline owns the fusion decision logic. It is provider- and transport-
// agnostic — server.go wires it to SSE; tests wire it to a capture sink.
type Pipeline struct {
	Registry        *providers.Registry
	Logger          *slog.Logger
	EmitProgress    bool // emit Sink.Progress between phases
}

// Run executes the pipeline against a single client request.
//
// model is the resolved config.Model entry for the request. base is the
// translated provider request without the per-endpoint model field set
// (Run fills it in per call).
func (p *Pipeline) Run(
	ctx context.Context,
	model config.Model,
	base providers.CompletionRequest,
	sink Sink,
) (Decision, error) {
	start := time.Now()
	dec := Decision{Kind: DecisionPassthrough}

	primaryEP := model.Primary
	primary, ok := p.Registry.Get(primaryEP.Provider)
	if !ok {
		return dec, fmt.Errorf("primary provider %q not registered", primaryEP.Provider)
	}

	// ---- 1. Speculative buffered call to primary ----
	if p.EmitProgress {
		_ = sink.Progress("fuse: evaluating turn")
	}
	specReq := withEndpoint(base, primaryEP)
	specResp, err := primary.Complete(ctx, specReq)
	if err != nil {
		dec.TotalLatency = time.Since(start)
		dec.Kind = DecisionFallback
		dec.FallbackReason = "speculative_failed"
		return dec, fmt.Errorf("speculative: %w", err)
	}

	// ---- 2. Detect ----
	if !IsFinalization(specResp) {
		// Tool-call turn: emit the buffered primary response as deltas.
		if err := emitBuffered(sink, specResp); err != nil {
			dec.TotalLatency = time.Since(start)
			return dec, err
		}
		dec.Kind = DecisionPassthrough
		dec.TotalLatency = time.Since(start)
		return dec, nil
	}

	// ---- 3. Panel fan-out ----
	if p.EmitProgress {
		_ = sink.Progress(fmt.Sprintf("fuse: querying panel (%d models)", len(model.Panel)))
	}
	panelTimeout := time.Duration(model.PanelTimeoutMs) * time.Millisecond
	rawResults := runPanel(ctx, p.Registry, model.Panel, base, panelTimeout)
	dec.Panel = panelResultsForSnapshot(rawResults)
	panel := summarizePanel(rawResults)

	if len(panel.Successes) < model.PanelMinSuccess {
		// Not enough panel responses — fall back to buffered speculative.
		if p.EmitProgress {
			_ = sink.Progress("fuse: insufficient panel, using speculative answer")
		}
		if err := emitBuffered(sink, specResp); err != nil {
			dec.TotalLatency = time.Since(start)
			return dec, err
		}
		dec.Kind = DecisionFallback
		dec.FallbackReason = "panel_insufficient"
		dec.TotalLatency = time.Since(start)
		return dec, nil
	}

	// ---- 4. Judge ----
	if p.EmitProgress {
		_ = sink.Progress(fmt.Sprintf("fuse: judging (%d responses)", len(panel.Successes)))
	}
	judgeStart := time.Now()
	analysis, err := runJudge(ctx, p.Registry, model.Judge, base.Messages, panel)
	dec.JudgeLatency = time.Since(judgeStart)
	if err != nil {
		if p.EmitProgress {
			_ = sink.Progress("fuse: judge failed, using speculative answer")
		}
		if emitErr := emitBuffered(sink, specResp); emitErr != nil {
			dec.TotalLatency = time.Since(start)
			return dec, emitErr
		}
		dec.Kind = DecisionFallback
		dec.FallbackReason = "judge_failed"
		dec.TotalLatency = time.Since(start)
		p.Logger.Warn("judge failed, fell back", "error", err)
		return dec, nil
	}
	dec.JudgeAnalysis = analysis.asMap()

	// ---- 5. Final streaming primary call with handoff ----
	if p.EmitProgress {
		_ = sink.Progress("fuse: writing final answer")
	}
	finalReq := withEndpoint(base, primaryEP)
	finalReq.Messages = buildHandoffMessages(base.Messages, analysis)
	// Final call writes the answer — no tools needed, the model is just
	// producing text. Stripping them keeps the request shorter and
	// discourages the model from initiating another tool round.
	finalReq.Tools = nil
	finalReq.ToolChoice = nil

	finalSink := &finalStreamSink{out: sink}
	if err := primary.Stream(ctx, finalReq, finalSink); err != nil {
		// We can't replay the buffered speculative cleanly if the final
		// primary partially streamed text first. If the sink hasn't seen
		// any content yet, fall back; otherwise propagate the error.
		if !finalSink.sawContent {
			if p.EmitProgress {
				_ = sink.Progress("fuse: final call failed, using speculative answer")
			}
			if emitErr := emitBuffered(sink, specResp); emitErr != nil {
				dec.TotalLatency = time.Since(start)
				return dec, emitErr
			}
			dec.Kind = DecisionFallback
			dec.FallbackReason = "primary_final_failed"
			dec.TotalLatency = time.Since(start)
			p.Logger.Warn("final primary failed before content, fell back", "error", err)
			return dec, nil
		}
		dec.Kind = DecisionFallback
		dec.FallbackReason = "primary_final_failed_midstream"
		dec.TotalLatency = time.Since(start)
		return dec, fmt.Errorf("final primary mid-stream: %w", err)
	}

	dec.Kind = DecisionFusion
	dec.TotalLatency = time.Since(start)
	return dec, nil
}

// withEndpoint copies base and applies the endpoint's model and optional
// temperature override. base is not mutated.
func withEndpoint(base providers.CompletionRequest, ep config.Endpoint) providers.CompletionRequest {
	out := base
	out.Model = ep.Model
	if ep.Temperature != nil {
		t := *ep.Temperature
		out.Temperature = &t
	}
	return out
}

// emitBuffered re-emits a non-streaming CompletionResponse as a sequence of
// Sink deltas, ending in Done. Used for passthrough and fallback paths.
func emitBuffered(sink Sink, resp *providers.CompletionResponse) error {
	if resp == nil {
		return errors.New("nil response")
	}
	if resp.Content != "" {
		if err := sink.Content(resp.Content); err != nil {
			return err
		}
	}
	// Tool calls in the buffered response need to be emitted as deltas in
	// OpenAI shape so the client can build them up.
	for i, tc := range resp.ToolCalls {
		idDelta := providers.ToolCallDelta{
			Index:        i,
			ID:           tc.ID,
			FunctionName: tc.Function.Name,
		}
		if err := sink.ToolCallDelta(idDelta); err != nil {
			return err
		}
		if tc.Function.Arguments != "" {
			argDelta := providers.ToolCallDelta{
				Index:            i,
				ArgumentsPartial: tc.Function.Arguments,
			}
			if err := sink.ToolCallDelta(argDelta); err != nil {
				return err
			}
		}
	}
	reason := resp.FinishReason
	if reason == "" {
		reason = "stop"
	}
	return sink.Done(reason)
}

func panelResultsForSnapshot(results []PanelMemberResult) []PanelResult {
	out := make([]PanelResult, 0, len(results))
	for _, r := range results {
		pr := PanelResult{
			Provider:  r.Endpoint.Provider,
			Model:     r.Endpoint.Model,
			LatencyMs: r.Latency.Milliseconds(),
			OK:        r.OK(),
		}
		if r.Err != nil {
			pr.Error = r.Err.Error()
		}
		out = append(out, pr)
	}
	return out
}

// finalStreamSink adapts a Pipeline.Sink to a providers.StreamSink for the
// final primary call, tracking whether any content has been emitted so the
// pipeline can decide between fallback and propagate on mid-stream errors.
type finalStreamSink struct {
	out        Sink
	sawContent bool
}

func (s *finalStreamSink) Delta(d providers.Delta) error {
	if d.Content != "" {
		s.sawContent = true
		if err := s.out.Content(d.Content); err != nil {
			return err
		}
	}
	if d.ToolCallDelta != nil {
		// Shouldn't happen on the final call (we stripped tools), but if it
		// does, pass it through so we don't lose information.
		if err := s.out.ToolCallDelta(*d.ToolCallDelta); err != nil {
			return err
		}
	}
	return nil
}

func (s *finalStreamSink) Done(reason string) error {
	return s.out.Done(reason)
}
