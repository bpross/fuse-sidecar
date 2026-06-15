// Package fusion implements the speculative → detect → panel → judge →
// final-primary pipeline.
//
// The entry point is Pipeline.Run, which takes the OpenAI-shaped request,
// resolves providers, and either streams a passthrough response or runs the
// full fusion handoff. All decisions are recorded in a Decision that the
// server uses to emit snapshots and metrics.
package fusion

import "github.com/bpross/fuse-sidecar/internal/providers"

// IsFinalization reports whether a speculative response is a text-only
// finalization turn (true) or a tool-call turn that should be re-emitted as
// passthrough (false).
//
// The predicate is: response has zero tool_calls AND finish_reason is not
// "tool_calls". Anything else — stop, length, content_filter, or empty —
// is treated as finalization.
func IsFinalization(resp *providers.CompletionResponse) bool {
	if resp == nil {
		return false
	}
	if len(resp.ToolCalls) > 0 {
		return false
	}
	if resp.FinishReason == "tool_calls" {
		return false
	}
	return true
}
