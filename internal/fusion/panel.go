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
type PanelMemberResult struct {
	Endpoint config.Endpoint
	Response *providers.CompletionResponse
	Latency  time.Duration
	Err      error
}

// OK reports whether this member returned a usable response.
func (r PanelMemberResult) OK() bool {
	return r.Err == nil && r.Response != nil && r.Response.Content != ""
}

// runPanel fans out the original request to every panel endpoint in
// parallel and gathers results up to the wall-clock cap.
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
			req := base
			req.Model = ep.Model
			if ep.Temperature != nil {
				t := *ep.Temperature
				req.Temperature = &t
			}
			started := time.Now()
			resp, err := prov.Complete(panelCtx, req)
			results[i].Latency = time.Since(started)
			results[i].Response = resp
			results[i].Err = err
		}()
	}
	wg.Wait()
	return results
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
