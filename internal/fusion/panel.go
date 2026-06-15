package fusion

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/bpross/fuse-sidecar/internal/config"
	"github.com/bpross/fuse-sidecar/internal/providers"
)

// PanelMemberResult is the outcome of a single panel call.
//
// Attempts is the number of upstream calls made (1 = first try, 2 = one retry).
// Useful when a panel member returned empty content the first time and was
// retried with a larger token budget.
type PanelMemberResult struct {
	Endpoint config.Endpoint
	Response *providers.CompletionResponse
	Latency  time.Duration
	Err      error
	Attempts int
}

// OK reports whether this member returned a usable response.
func (r PanelMemberResult) OK() bool {
	return r.Err == nil && r.Response != nil && r.Response.Content != ""
}

// panelRetryFloor is the smallest max_tokens we'll force on a retry. Reasoning
// models that returned empty on the first attempt are often starved for output
// budget; bumping the floor recovers most of these.
const panelRetryFloor = 8192

// runPanel fans out the original request to every panel endpoint in
// parallel and gathers results up to the wall-clock cap.
//
// Each panel member gets one retry if it returned empty content (zero-length
// completion) on the first attempt. Empty completions are common for
// reasoning models that exhaust their token budget on hidden reasoning before
// emitting any visible output; the retry forces a generous max_tokens floor.
//
// The returned slice is in the same order as the input endpoints. Members
// that did not complete before the cap have Err == context.DeadlineExceeded.
func runPanel(
	ctx context.Context,
	reg *providers.Registry,
	endpoints []config.Endpoint,
	base providers.CompletionRequest,
	timeout time.Duration,
) []PanelMemberResult {
	results := make([]PanelMemberResult, len(endpoints))
	var wg sync.WaitGroup
	panelCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	for i, ep := range endpoints {
		i, ep := i, ep
		results[i].Endpoint = ep
		prov, ok := reg.Get(ep.Provider)
		if !ok {
			results[i].Err = fmt.Errorf("provider %q not registered", ep.Provider)
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			started := time.Now()
			results[i] = callPanelMember(panelCtx, prov, ep, base, results[i])
			results[i].Latency = time.Since(started)
		}()
	}
	wg.Wait()
	return results
}

// callPanelMember invokes one panel endpoint with up to one empty-content retry.
// The carry parameter holds the endpoint that was set on the result before the
// goroutine started; we preserve it and only fill in response/error/attempts.
//
// Tools and tool_choice are stripped from the panel call. Panel members
// cannot execute tools — the agent loop on the client side owns tool
// execution and only sees one model at a time. If we left tools attached,
// panel members would routinely emit tool_call responses we can't satisfy,
// burning tokens and producing zero usable text. The conversation history
// already contains the tool_call/tool_result pairs from the primary's
// investigation, so panel members see what was found and can reason from it.
func callPanelMember(
	ctx context.Context,
	prov providers.Provider,
	ep config.Endpoint,
	base providers.CompletionRequest,
	carry PanelMemberResult,
) PanelMemberResult {
	req := withEndpoint(base, ep)
	req.Tools = nil
	req.ToolChoice = nil
	out := carry
	out.Attempts = 1
	resp, err := prov.Complete(ctx, req)
	out.Response = resp
	out.Err = err

	if err == nil && resp != nil && resp.Content == "" {
		// Empty content — likely token budget starved on a reasoning model.
		// Retry once with a forced floor; if the endpoint already set a
		// larger max_tokens, keep it.
		retry := req
		floor := panelRetryFloor
		if ep.MaxTokens != nil && *ep.MaxTokens > floor {
			floor = *ep.MaxTokens
		}
		retry.MaxTokens = &floor
		out.Attempts = 2
		resp2, err2 := prov.Complete(ctx, retry)
		out.Response = resp2
		out.Err = err2
	}
	return out
}

// PanelSummary collapses raw results into the bits the judge needs.
type PanelSummary struct {
	Successes []PanelMemberResult
	Failures  []PanelMemberResult
}

func summarizePanel(results []PanelMemberResult) PanelSummary {
	var s PanelSummary
	for _, r := range results {
		if r.OK() {
			s.Successes = append(s.Successes, r)
		} else {
			s.Failures = append(s.Failures, r)
		}
	}
	return s
}
