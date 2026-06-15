package providers

import (
	"fmt"
	"os"

	"github.com/bpross/fuse-sidecar/internal/config"
)

// BuildRegistry constructs concrete providers from a config.
//
// Built-in provider IDs:
//   - "anthropic" → Anthropic Messages API
//   - "openai"    → OpenAI Chat Completions API
//
// Any other ID is treated as a generic OpenAI-compatible provider, which
// requires the config to provide a base_url. This lets users plug in
// OpenRouter, LM Studio, llama.cpp, Together, Groq, or anything else that
// speaks /v1/chat/completions without code changes.
func BuildRegistry(cfg *config.Config) (*Registry, error) {
	reg := NewRegistry()
	for id, p := range cfg.Providers {
		key := os.Getenv(p.APIKeyEnv)
		if key == "" {
			return nil, fmt.Errorf("provider %q: %s is empty", id, p.APIKeyEnv)
		}
		switch id {
		case "anthropic":
			reg.Register(NewAnthropic(AnthropicConfig{
				APIKey:  key,
				BaseURL: p.BaseURL,
				Headers: p.Headers,
			}))
		case "openai":
			reg.Register(NewOpenAI(OpenAIConfig{
				APIKey:  key,
				BaseURL: p.BaseURL,
				Headers: p.Headers,
			}))
		default:
			if p.BaseURL == "" {
				return nil, fmt.Errorf("provider %q is not a built-in; set base_url to use it as an OpenAI-compatible endpoint", id)
			}
			reg.Register(NewOpenAICompatible(id, OpenAIConfig{
				APIKey:  key,
				BaseURL: p.BaseURL,
				Headers: p.Headers,
			}))
		}
	}
	return reg, nil
}
