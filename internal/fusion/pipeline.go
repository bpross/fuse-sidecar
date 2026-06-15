// Package fusion implements the speculative → detect → panel → judge →
// final-primary pipeline.
//
// The entry point is Pipeline.Run, which takes the OpenAI-shaped request,
// resolves providers, and either streams a tool-call passthrough response or
// runs the full fusion handoff. The pipeline emits into a providers.Sink
// throughout — both model content and fuse-level progress narration travel
// as Delta values on the same sink. The server adapts that sink to SSE.
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

// Decision is the outcome of one pipeline run; the server uses it to
// produce a snapshot and metrics.
type Decision struct {
	Kind             DecisionKind
	FallbackReason   string
	Panel            []PanelResult
	JudgeLatency     time.Duration
	JudgeAnalysis    map[string]any
	TotalLatency     time.Duration
	FinalAnswerBytes int
	FinalAnswerHead  string
	// Usage is the token usage rolled up across every upstream call made
	// during this turn: speculative, every panel member, judge, and the
	// final-primary stream. CacheReadTokens vs CacheCreationTokens shows
	// how much of the input was served from a provider's prefix cache.
	Usage providers.Usage
}

// DecisionKind classifies how a request was served. The four kinds form a
// 2x2: was this turn fused (or was fusion N/A or skipped)? Did the client
// receive a usable answer (or only an error)?
type DecisionKind string

const (
	// DecisionToolCall: the speculative response contained tool_calls, so
	// fusion was not applicable. The buffered speculative response was
	// re-emitted to the client. This is the dominant case during an agent
	// investigation loop and is not a failure of any kind.
	DecisionToolCall DecisionKind = "tool_call"

	// DecisionFusion: the full speculative → panel → judge → final-primary
	// pipeline ran and the client received the fused final answer.
	DecisionFusion DecisionKind = "fusion"

	// DecisionDegraded: fusion was applicable but the panel/judge layer
	// could not run; the client received the buffered speculative answer
	// as a usable degradation. The Decision.FallbackReason names which
	// layer fell back.
	DecisionDegraded DecisionKind = "degraded"

	// DecisionFailed: the pipeline could not deliver any useful answer.
	// The client saw an error event on the stream or an HTTP error.
	DecisionFailed DecisionKind = "failed"
)

// PanelResult is a snapshot-friendly summary of one panel call.
type PanelResult struct {
	Provider            string
	Model               string
	LatencyMs           int64
	OK                  bool
	Error               string
	Attempts            int
	InputTokens         int
	OutputTokens        int
	CacheReadTokens     int
	CacheCreationTokens int
}

// Pipeline owns the fusion decision logic. It is provider- and transport-
// agnostic — server.go wires it to SSE; tests wire it to a capture sink.
type Pipeline struct {
	Registry     *providers.Registry
	Logger       *slog.Logger
	EmitProgress bool
}

// Run executes the pipeline against a single client request.
//
// model is the resolved config.Model entry for the request. base is the
// translated provider request without the per-endpoint model field set
// (Run fills it in per phase).
func (p *Pipeline) Run(
	ctx context.Context,
	model config.Model,
	base providers.CompletionRequest,
	sink providers.Sink,
) (dec Decision, err error) {
	start := time.Now()

	counted := &countingSink{inner: sink}
	sink = counted
	// usage accumulates across every upstream call; the defer copies it into
	// the returned Decision regardless of which branch returns. Because dec
	// is a named return, branches that build a fresh Decision literal still
	// get the accumulated usage stamped on by this defer.
	var usage providers.Usage
	defer func() {
		dec.TotalLatency = time.Since(start)
		dec.FinalAnswerBytes = counted.bytes
		dec.FinalAnswerHead = counted.head()
		dec.Usage = usage
	}()

	primary, ok := p.Registry.Get(model.Primary.Provider)
	if !ok {
		return Decision{Kind: DecisionFailed, FallbackReason: "primary_provider_missing"},
			fmt.Errorf("primary provider %q not registered", model.Primary.Provider)
	}

	// Phase 1: speculative buffered call to primary.
	p.progress(sink, "fuse: evaluating turn")
	specResp, err := primary.Complete(ctx, withEndpoint(base, model.Primary))
	if err != nil {
		return Decision{Kind: DecisionFailed, FallbackReason: "speculative_failed"},
			fmt.Errorf("speculative: %w", err)
	}
	usage.Add(specResp.Usage)

	// Phase 2: detect. Tool-call turns short-circuit to passthrough.
	if !IsFinalization(specResp) {
		if err := emitBuffered(sink, specResp); err != nil {
			return Decision{Kind: DecisionFailed, FallbackReason: "tool_call_emit_failed"}, err
		}
		return Decision{Kind: DecisionToolCall}, nil
	}

	// Phase 3: panel fan-out.
	p.progress(sink, fmt.Sprintf("fuse: querying panel (%d models)", len(model.Panel)))
	rawResults := runPanel(ctx, p.Registry, model.Panel, base,
		time.Duration(model.PanelTimeoutMs)*time.Millisecond)
	dec.Panel = panelResultsForSnapshot(rawResults)
	for _, r := range rawResults {
		if r.Response != nil {
			usage.Add(r.Response.Usage)
		}
	}
	panel := summarizePanel(rawResults)

	// If the real panel is too sparse, promote the speculative response to a
	// virtual panel member so fusion still runs. If even the speculative is
	// empty there is nothing to fuse — degrade.
	if len(panel.Successes) < model.PanelMinSuccess {
		if specResp == nil || specResp.Content == "" {
			p.progress(sink, "fuse: ✗ NOT FUSED — no panel responses, returning speculative answer as fallback")
			return p.degradeWithSpeculative(sink, specResp, "panel_insufficient")
		}
		panel.Successes = append(panel.Successes, speculativePanelist(model.Primary, specResp))
		p.progress(sink, "fuse: panel sparse, using speculative as fallback panelist")
	}

	// Phase 4: judge.
	p.progress(sink, fmt.Sprintf("fuse: judging (%d responses)", len(panel.Successes)))
	judgeStart := time.Now()
	analysis, judgeUsage, err := runJudge(ctx, p.Registry, model.Judge, base.Messages, panel)
	dec.JudgeLatency = time.Since(judgeStart)
	usage.Add(judgeUsage)
	if err != nil {
		p.Logger.Warn("judge failed, degrading", "error", err)
		p.progress(sink, "fuse: ✗ NOT FUSED — judge failed, returning speculative answer as fallback")
		return p.degradeWithSpeculative(sink, specResp, "judge_failed")
	}
	dec.JudgeAnalysis = analysis.asMap()

	// Phase 5: final streaming primary call with handoff.
	p.progress(sink, fmt.Sprintf(
		"fuse: ✦ FUSED ANSWER from %d/%d panel models + judge — writing now",
		len(panel.Successes), len(model.Panel),
	))
	finalDec, finalUsage, err := p.finalStream(ctx, model.Primary, base, analysis, specResp, sink, dec)
	usage.Add(finalUsage)
	return finalDec, err
}

// degradeWithSpeculative emits the buffered speculative response as the
// final answer and returns a DecisionDegraded decision. Used by every
// "panel/judge layer couldn't run, but the speculative is usable" branch.
func (p *Pipeline) degradeWithSpeculative(
	sink providers.Sink,
	specResp *providers.CompletionResponse,
	reason string,
) (Decision, error) {
	if err := emitBuffered(sink, specResp); err != nil {
		return Decision{Kind: DecisionFailed, FallbackReason: reason + "_emit_failed"}, err
	}
	return Decision{Kind: DecisionDegraded, FallbackReason: reason}, nil
}

// finalStream runs the streaming primary call with the judge analysis
// folded into the conversation, then settles the decision based on whether
// any content reached the client. It returns the stream's token usage so
// the caller can roll it into the per-turn total.
func (p *Pipeline) finalStream(
	ctx context.Context,
	primaryEP config.Endpoint,
	base providers.CompletionRequest,
	analysis *JudgeAnalysis,
	specResp *providers.CompletionResponse,
	sink providers.Sink,
	dec Decision,
) (Decision, providers.Usage, error) {
	primary, _ := p.Registry.Get(primaryEP.Provider) // checked in Run
	finalReq := withEndpoint(base, primaryEP)
	finalReq.Messages = buildHandoffMessages(base.Messages, analysis)
	// The final call writes text; no tools needed. Stripping keeps the
	// request shorter and prevents another tool round.
	finalReq.Tools = nil
	finalReq.ToolChoice = nil

	gate := &contentGate{inner: sink}
	streamUsage, err := primary.Stream(ctx, finalReq, gate)
	if err != nil {
		// If we never streamed any content, we can still salvage by
		// emitting the buffered speculative as the answer.
		if !gate.sawContent {
			p.Logger.Warn("final primary failed before content, degrading", "error", err)
			p.progress(sink, "fuse: ✗ NOT FUSED — final call failed, returning speculative answer as fallback")
			d, derr := p.degradeWithSpeculative(sink, specResp, "primary_final_failed")
			return d, streamUsage, derr
		}
		// Mid-stream failure: we've written partial content already; the
		// best we can do is surface the error to the caller.
		dec.Kind = DecisionFailed
		dec.FallbackReason = "primary_final_failed_midstream"
		return dec, streamUsage, fmt.Errorf("final primary mid-stream: %w", err)
	}

	dec.Kind = DecisionFusion
	return dec, streamUsage, nil
}

// progress emits a reasoning_content delta describing fusion progress.
// Errors are intentionally ignored: a Progress write may fail because the
// client disconnected, but the pipeline still needs to finish the panel
// and judge work so the snapshot records what happened. The next Content
// or Done write will surface the disconnect to the caller.
func (p *Pipeline) progress(sink providers.Sink, text string) {
	if !p.EmitProgress {
		return
	}
	_ = sink.Delta(providers.Delta{ReasoningContent: text})
}

// withEndpoint copies base and applies the endpoint's model and optional
// sampling overrides. base is not mutated.
//
// When the endpoint opts into caching, CachePrefix is set so the provider
// marks the conversation prefix cacheable. For the primary endpoint this is
// the key to sharing the prefix cache between the speculative call and the
// final-primary call, which use the same model and the same (large) prefix.
func withEndpoint(base providers.CompletionRequest, ep config.Endpoint) providers.CompletionRequest {
	out := base
	out.Model = ep.Model
	if ep.Temperature != nil {
		t := *ep.Temperature
		out.Temperature = &t
	}
	if ep.MaxTokens != nil {
		n := *ep.MaxTokens
		out.MaxTokens = &n
	}
	out.CachePrefix = ep.CacheEnabled()
	return out
}

// speculativePanelist wraps the speculative response as a virtual panel
// member, labeled so the judge can see it's a self-reference.
func speculativePanelist(ep config.Endpoint, resp *providers.CompletionResponse) PanelMemberResult {
	return PanelMemberResult{
		Endpoint: config.Endpoint{
			Provider: ep.Provider,
			Model:    ep.Model + " (speculative)",
		},
		Response: resp,
		Attempts: 1,
	}
}

// emitBuffered re-emits a non-streaming CompletionResponse as a sequence
// of Sink Deltas, ending in Done. Used for tool-call passthrough and
// every degradation path.
func emitBuffered(sink providers.Sink, resp *providers.CompletionResponse) error {
	if resp == nil {
		return errors.New("nil response")
	}
	if resp.Content != "" {
		if err := sink.Delta(providers.Delta{Content: resp.Content}); err != nil {
			return err
		}
	}
	for i, tc := range resp.ToolCalls {
		if err := sink.Delta(providers.Delta{ToolCallDelta: &providers.ToolCallDelta{
			Index:        i,
			ID:           tc.ID,
			FunctionName: tc.Function.Name,
		}}); err != nil {
			return err
		}
		if tc.Function.Arguments != "" {
			if err := sink.Delta(providers.Delta{ToolCallDelta: &providers.ToolCallDelta{
				Index:            i,
				ArgumentsPartial: tc.Function.Arguments,
			}}); err != nil {
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
			Attempts:  r.Attempts,
		}
		if r.Response != nil {
			pr.InputTokens = r.Response.Usage.PromptTokens
			pr.OutputTokens = r.Response.Usage.CompletionTokens
			pr.CacheReadTokens = r.Response.Usage.CacheReadTokens
			pr.CacheCreationTokens = r.Response.Usage.CacheCreationTokens
		}
		switch {
		case r.Err != nil:
			pr.Error = r.Err.Error()
		case r.Response == nil:
			pr.Error = "no response returned"
		case r.Response.Content == "":
			pr.Error = fmt.Sprintf("response had empty content (finish=%q, prompt_tokens=%d, completion_tokens=%d)",
				r.Response.FinishReason, r.Response.Usage.PromptTokens, r.Response.Usage.CompletionTokens)
		}
		out = append(out, pr)
	}
	return out
}

// countingSink wraps a providers.Sink and tallies bytes of Content emitted,
// retaining a UTF-8-safe prefix for postmortem inspection. ReasoningContent
// and ToolCallDelta deltas are forwarded but not counted.
type countingSink struct {
	inner providers.Sink
	bytes int
	buf   []byte
}

const countingSinkHeadCap = 512

func (c *countingSink) Delta(d providers.Delta) error {
	if d.Content != "" {
		c.bytes += len(d.Content)
		if len(c.buf) < countingSinkHeadCap {
			need := countingSinkHeadCap - len(c.buf)
			if need > len(d.Content) {
				need = len(d.Content)
			}
			c.buf = append(c.buf, d.Content[:need]...)
		}
	}
	return c.inner.Delta(d)
}

func (c *countingSink) Done(reason string) error {
	return c.inner.Done(reason)
}

// head returns up to 200 valid UTF-8 bytes from the buffer.
func (c *countingSink) head() string {
	if len(c.buf) == 0 {
		return ""
	}
	const maxHead = 200
	end := len(c.buf)
	if end > maxHead {
		end = maxHead
	}
	// Walk back to a UTF-8 boundary if we sliced mid-rune.
	for end > 0 && (c.buf[end-1]&0xC0) == 0x80 {
		end--
	}
	return string(c.buf[:end])
}

// contentGate observes whether any Content delta has reached the inner
// sink. The final-stream branch uses this to decide between degrading to
// the speculative answer (no content yet) and propagating a mid-stream
// error (some content already written).
type contentGate struct {
	inner      providers.Sink
	sawContent bool
}

func (g *contentGate) Delta(d providers.Delta) error {
	if d.Content != "" {
		g.sawContent = true
	}
	return g.inner.Delta(d)
}

func (g *contentGate) Done(reason string) error {
	return g.inner.Done(reason)
}
