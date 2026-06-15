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

	compact, err := json.Marshal(analysis)
	if err != nil {
		// JudgeAnalysis is plain types; marshaling cannot realistically
		// fail. Fall back to an empty object so the handoff still works.
		compact = []byte("{}")
	}

	out = append(out, providers.Message{
		Role: "assistant",
		Content: "I consulted a panel of models and received this structured analysis:\n\n" +
			string(compact),
	})
	out = append(out, providers.Message{
		Role:    "user",
		Content: "Using the panel analysis above as additional context, now write the final answer to my original request. Do not mention the panel or the analysis in your answer.",
	})
	return out
}
