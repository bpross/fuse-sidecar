package providers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// OpenAI implements Provider over the ChatCompletions API.
//
// The provider name is "openai". Use OpenAICompatible (separate constructor)
// for OpenRouter, LM Studio, llama.cpp, or any other OpenAI-compatible
// endpoint that differs only in base URL and headers.
type OpenAI struct {
	t transport
}

// OpenAIConfig contains construction parameters.
type OpenAIConfig struct {
	Name    string // defaults to "openai"
	APIKey  string
	BaseURL string // defaults to https://api.openai.com
	Headers map[string]string
	Client  *http.Client
}

// NewOpenAI constructs an OpenAI provider with default base URL.
func NewOpenAI(cfg OpenAIConfig) *OpenAI {
	name := cfg.Name
	if name == "" {
		name = "openai"
	}
	base := cfg.BaseURL
	if base == "" {
		base = "https://api.openai.com"
	}
	c := cfg.Client
	if c == nil {
		c = &http.Client{}
	}
	return &OpenAI{
		t: transport{
			client:     c,
			baseURL:    strings.TrimRight(base, "/"),
			headers:    cfg.Headers,
			authHeader: "Authorization",
			authValue:  "Bearer " + cfg.APIKey,
			name:       name,
		},
	}
}

// NewOpenAICompatible constructs a provider that speaks OpenAI's protocol
// but lives at a different URL. Used for OpenRouter, LM Studio, etc.
//
// The provider's Name() returns the given name so it can be registered
// under any ID (e.g. "openrouter") even though the protocol is OpenAI.
func NewOpenAICompatible(name string, cfg OpenAIConfig) *OpenAI {
	cfg.Name = name
	return NewOpenAI(cfg)
}

// Name returns the provider identifier (e.g. "openai" or "openrouter").
func (o *OpenAI) Name() string { return o.t.name }

// Complete runs a non-streaming chat completion.
func (o *OpenAI) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	body, err := buildOpenAIRequest(req, false)
	if err != nil {
		return nil, err
	}
	resp, err := o.t.post(ctx, "/v1/chat/completions", body, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var raw openAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("%s decode: %w", o.t.name, err)
	}
	return raw.toCompletionResponse(), nil
}

// Stream runs a streaming chat completion, emitting deltas to sink and
// returning the usage from the final include_usage chunk.
func (o *OpenAI) Stream(ctx context.Context, req CompletionRequest, sink Sink) (Usage, error) {
	body, err := buildOpenAIRequest(req, true)
	if err != nil {
		return Usage{}, err
	}
	resp, err := o.t.post(ctx, "/v1/chat/completions", body, http.Header{"Accept": []string{"text/event-stream"}})
	if err != nil {
		return Usage{}, err
	}
	defer resp.Body.Close()
	return parseOpenAIStream(resp.Body, sink)
}

// --- Request shapes ---

type openAIRequest struct {
	Model     string          `json:"model"`
	Messages  []openAIMessage `json:"messages"`
	Tools     []Tool          `json:"tools,omitempty"`
	ToolChoice json.RawMessage `json:"tool_choice,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
	// MaxTokens is the legacy field used by gpt-4 and earlier.
	MaxTokens *int `json:"max_tokens,omitempty"`
	// MaxCompletionTokens is required by gpt-5 and o-series reasoning models.
	// Exactly one of MaxTokens / MaxCompletionTokens should be set per request.
	MaxCompletionTokens *int                `json:"max_completion_tokens,omitempty"`
	TopP                *float64            `json:"top_p,omitempty"`
	Stop                []string            `json:"stop,omitempty"`
	Stream              bool                `json:"stream,omitempty"`
	StreamOptions       *openAIStreamOption `json:"stream_options,omitempty"`
	ResponseFormat      *ResponseFormat     `json:"response_format,omitempty"`
}

// openAIStreamOption requests a final usage chunk on streaming responses.
type openAIStreamOption struct {
	IncludeUsage bool `json:"include_usage"`
}

// modelUsesMaxCompletionTokens reports whether the given OpenAI model ID
// requires max_completion_tokens instead of the legacy max_tokens. Heuristic
// based on observed API behavior:
//   - gpt-5*  (gpt-5, gpt-5-mini, gpt-5-codex, etc.)
//   - o1*, o3*, o4* (reasoning models)
// Anything else uses max_tokens. False positives here cost nothing because
// OpenAI accepts max_completion_tokens on newer models without complaint;
// false negatives produce the 400 we just hit.
func modelUsesMaxCompletionTokens(model string) bool {
	switch {
	case strings.HasPrefix(model, "gpt-5"):
		return true
	case strings.HasPrefix(model, "o1"), strings.HasPrefix(model, "o3"), strings.HasPrefix(model, "o4"):
		return true
	default:
		return false
	}
}

type openAIMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

func buildOpenAIRequest(req CompletionRequest, stream bool) ([]byte, error) {
	out := openAIRequest{
		Model:          req.Model,
		Messages:       make([]openAIMessage, 0, len(req.Messages)),
		Tools:          req.Tools,
		ToolChoice:     req.ToolChoice,
		Temperature:    req.Temperature,
		TopP:           req.TopP,
		Stop:           req.Stop,
		Stream:         stream,
		ResponseFormat: req.ResponseFormat,
	}
	if req.MaxTokens != nil {
		if modelUsesMaxCompletionTokens(req.Model) {
			out.MaxCompletionTokens = req.MaxTokens
		} else {
			out.MaxTokens = req.MaxTokens
		}
	}
	if stream {
		// Ask for a trailing usage chunk so streaming calls can report
		// token + cache counts the same way non-streaming calls do.
		out.StreamOptions = &openAIStreamOption{IncludeUsage: true}
	}
	for _, m := range req.Messages {
		out.Messages = append(out.Messages, openAIMessage{
			Role:       m.Role,
			Content:    m.Content,
			ToolCalls:  m.ToolCalls,
			ToolCallID: m.ToolCallID,
			Name:       m.Name,
		})
	}
	return json.Marshal(out)
}

// --- Response shapes ---

type openAIResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Choices []openAIChoice `json:"choices"`
	Usage   openAIUsage    `json:"usage"`
}

type openAIChoice struct {
	Index        int           `json:"index"`
	Message      openAIMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type openAIUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

// toUsage converts OpenAI's usage block. OpenAI does automatic prefix
// caching with no opt-in; cached_tokens reports how many prompt tokens
// were served from cache. prompt_tokens already includes the cached
// tokens, so we subtract to keep PromptTokens meaning "uncached input"
// consistent with the Anthropic mapping. OpenAI does not bill a cache
// write premium, so CacheCreationTokens stays zero.
func (u openAIUsage) toUsage() Usage {
	cached := u.PromptTokensDetails.CachedTokens
	uncached := u.PromptTokens - cached
	if uncached < 0 {
		uncached = 0
	}
	return Usage{
		PromptTokens:        uncached,
		CompletionTokens:    u.CompletionTokens,
		TotalTokens:         u.TotalTokens,
		CacheReadTokens:     cached,
		CacheCreationTokens: 0,
	}
}

func (r *openAIResponse) toCompletionResponse() *CompletionResponse {
	out := &CompletionResponse{
		Usage: r.Usage.toUsage(),
	}
	if len(r.Choices) > 0 {
		c := r.Choices[0]
		out.Content = c.Message.Content
		out.ToolCalls = c.Message.ToolCalls
		out.FinishReason = c.FinishReason
	}
	return out
}

// --- Streaming ---

type openAIStreamChunk struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Choices []openAIStreamChoice `json:"choices"`
	// Usage is present only on the final chunk when include_usage is set.
	Usage *openAIUsage `json:"usage"`
}

type openAIStreamChoice struct {
	Index        int                 `json:"index"`
	Delta        openAIStreamDelta   `json:"delta"`
	FinishReason *string             `json:"finish_reason"`
}

type openAIStreamDelta struct {
	Role             string                 `json:"role,omitempty"`
	Content          string                 `json:"content,omitempty"`
	ReasoningContent string                 `json:"reasoning_content,omitempty"`
	ToolCalls        []openAIStreamToolCall `json:"tool_calls,omitempty"`
}

type openAIStreamToolCall struct {
	Index    int                      `json:"index"`
	ID       string                   `json:"id,omitempty"`
	Type     string                   `json:"type,omitempty"`
	Function openAIStreamFunctionCall `json:"function,omitempty"`
}

type openAIStreamFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

func parseOpenAIStream(body io.Reader, sink Sink) (Usage, error) {
	var (
		finish string
		usage  Usage
	)
	err := scanSSE(body, func(ev sseEvent) error {
		if ev.Data == "[DONE]" {
			return errStreamDone
		}
		var chunk openAIStreamChunk
		if err := json.Unmarshal([]byte(ev.Data), &chunk); err != nil {
			return fmt.Errorf("openai stream chunk: %w", err)
		}
		if chunk.Usage != nil {
			usage = chunk.Usage.toUsage()
		}
		for _, c := range chunk.Choices {
			if c.FinishReason != nil && *c.FinishReason != "" {
				finish = *c.FinishReason
			}
			if c.Delta.Content != "" {
				if err := sink.Delta(Delta{Content: c.Delta.Content}); err != nil {
					return err
				}
			}
			if c.Delta.ReasoningContent != "" {
				if err := sink.Delta(Delta{ReasoningContent: c.Delta.ReasoningContent}); err != nil {
					return err
				}
			}
			for _, tc := range c.Delta.ToolCalls {
				if err := sink.Delta(Delta{ToolCallDelta: &ToolCallDelta{
					Index:            tc.Index,
					ID:               tc.ID,
					FunctionName:     tc.Function.Name,
					ArgumentsPartial: tc.Function.Arguments,
				}}); err != nil {
					return err
				}
			}
		}
		return nil
	})
	if err != nil && !errors.Is(err, errStreamDone) {
		return Usage{}, err
	}
	return usage, sink.Done(finish)
}

var _ Provider = (*OpenAI)(nil)
