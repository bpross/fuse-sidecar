package fusion

import (
	"encoding/json"

	"github.com/bpross/fuse-sidecar/internal/providers"
)

// buildHandoffMessages augments the original conversation with a synthetic
// assistant turn carrying the judge analysis and a synthetic user turn
// instructing the primary to write the final answer.
//
// The original messages slice is not mutated; a new slice is returned. The
// system prompt prefix is preserved byte-for-byte to keep prompt caching
// effective on providers that cache (Anthropic explicit, OpenAI implicit).
func buildHandoffMessages(original []providers.Message, analysis *JudgeAnalysis) []providers.Message {
	out := make([]providers.Message, 0, len(original)+2)
	out = append(out, original...)

	// Mark the last original message as the cache breakpoint so a
	// cache-aware provider caches the conversation prefix shared with the
	// speculative call. The two synthetic turns appended below sit after
	// the breakpoint and stay out of the cached span.
	if len(out) > 0 {
		out[len(out)-1].CacheBreakpoint = true
	}

	compact, err := json.Marshal(analysis)
	if err != nil {
		// JudgeAnalysis is plain types; marshaling cannot realistically
		// fail. Fall back to an empty object so the handoff still works.
		compact = []byte("{}")
	}

	out = append(out, providers.Message{
		Role: "assistant",
		Content: "Before writing my final answer I consulted a panel of models. " +
			"Here is the structured judge analysis of their responses:\n\n" +
			string(compact),
	})
	out = append(out, providers.Message{
		Role: "user",
		Content: "Using the panel analysis above as additional context, now write your complete final answer to my original request. " +
			"Write the full answer in your own voice — do not just summarize or reference the analysis, and do not mention the panel. " +
			"If my original request was for a plan, document, or detailed explanation, produce that in full now.",
	})
	return out
}
