package fusion

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/bpross/fuse-sidecar/internal/config"
	"github.com/bpross/fuse-sidecar/internal/providers"
)

// JudgeAnalysis is the structured output the judge produces. Mirrors the
// fields described in OpenRouter's fusion docs: it is comparison, not
// merging. The primary writes the final answer using this as context.
type JudgeAnalysis struct {
	Consensus      []string            `json:"consensus"`
	Contradictions []string            `json:"contradictions"`
	Partial        []string            `json:"partial"`
	Unique         map[string][]string `json:"unique,omitempty"`
	BlindSpots     []string            `json:"blind_spots,omitempty"`
}

const judgeSystemPrompt = `You are a judge comparing multiple model responses to the same prompt.
Your job is comparison, NOT merging. Return ONLY a JSON object with these keys:
  consensus:      points all or most responses agree on (highest confidence)
  contradictions: points where responses directly disagree
  partial:        points covered by some responses but not others
  unique:         map of response label -> list of points only that response made
  blind_spots:    points none of the responses addressed but should have

Be concise. Each entry is a short sentence. Output strictly valid JSON, no prose.`

// runJudge calls the judge provider with the original user prompt and the
// panel responses, then parses the structured analysis.
func runJudge(
	ctx context.Context,
	reg *providers.Registry,
	ep config.Endpoint,
	originalMessages []providers.Message,
	panel PanelSummary,
) (*JudgeAnalysis, error) {
	prov, ok := reg.Get(ep.Provider)
	if !ok {
		return nil, fmt.Errorf("judge provider %q not registered", ep.Provider)
	}

	userPrompt := buildJudgeUserPrompt(originalMessages, panel)

	req := providers.CompletionRequest{
		Model: ep.Model,
		Messages: []providers.Message{
			{Role: "system", Content: judgeSystemPrompt},
			{Role: "user", Content: userPrompt},
		},
		Temperature: ep.Temperature,
		ResponseFormat: &providers.ResponseFormat{
			Type: "json_object",
		},
	}
	// Note: we used to default temperature to 0 when unset, but some newer
	// models (e.g. Anthropic claude-opus-4-8) reject the field entirely.
	// Leave it nil if the config didn't set it.

	resp, err := prov.Complete(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("judge call: %w", err)
	}

	analysis, err := parseJudgeOutput(resp.Content)
	if err == nil {
		return analysis, nil
	}

	// Single retry with a stricter system message appended.
	req.Messages[0].Content += "\n\nIMPORTANT: Your previous response was not valid JSON. Output ONLY a JSON object, nothing else."
	resp, err2 := prov.Complete(ctx, req)
	if err2 != nil {
		return nil, fmt.Errorf("judge retry: %w", err2)
	}
	analysis, err = parseJudgeOutput(resp.Content)
	if err != nil {
		return nil, fmt.Errorf("judge output parse: %w", err)
	}
	return analysis, nil
}

func buildJudgeUserPrompt(originalMessages []providers.Message, panel PanelSummary) string {
	var b strings.Builder
	b.WriteString("Original task (final user message):\n")
	b.WriteString(lastUserContent(originalMessages))
	b.WriteString("\n\n")
	b.WriteString("Panel responses:\n\n")
	for i, m := range panel.Successes {
		label := fmt.Sprintf("Model %c (%s/%s)", 'A'+rune(i), m.Endpoint.Provider, m.Endpoint.Model)
		fmt.Fprintf(&b, "=== %s ===\n%s\n\n", label, m.Response.Content)
	}
	return b.String()
}

func lastUserContent(messages []providers.Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" && messages[i].Content != "" {
			return messages[i].Content
		}
	}
	return ""
}

// parseJudgeOutput extracts a JudgeAnalysis from the judge's response text.
// Some models wrap JSON in ```json fences; we tolerate that.
func parseJudgeOutput(s string) (*JudgeAnalysis, error) {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "```json")
	s = strings.TrimPrefix(s, "```")
	s = strings.TrimSuffix(s, "```")
	s = strings.TrimSpace(s)

	// Locate the outermost JSON object if there's trailing prose.
	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start < 0 || end <= start {
		return nil, fmt.Errorf("no JSON object found in judge output")
	}
	body := s[start : end+1]

	var a JudgeAnalysis
	if err := json.Unmarshal([]byte(body), &a); err != nil {
		return nil, fmt.Errorf("unmarshal judge output: %w", err)
	}
	return &a, nil
}

// asMap renders the analysis as a generic map (used for snapshots).
func (a *JudgeAnalysis) asMap() map[string]any {
	out := map[string]any{
		"consensus":      a.Consensus,
		"contradictions": a.Contradictions,
		"partial":        a.Partial,
	}
	if len(a.Unique) > 0 {
		out["unique"] = a.Unique
	}
	if len(a.BlindSpots) > 0 {
		out["blind_spots"] = a.BlindSpots
	}
	return out
}
