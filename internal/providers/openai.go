package providers

import (
	"bufio"
	"bytes"
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
	name    string
	apiKey  string
	baseURL string
	headers map[string]string
	client  *http.Client
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
		name:    name,
		apiKey:  cfg.APIKey,
		baseURL: strings.TrimRight(base, "/"),
		headers: cfg.Headers,
		client:  c,
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
func (o *OpenAI) Name() string { return o.name }

// Complete runs a non-streaming chat completion.
func (o *OpenAI) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	body, err := buildOpenAIRequest(req, false)
	if err != nil {
		return nil, err
	}
	httpReq, err := o.newRequest(ctx, body)
	if err != nil {
		return nil, err
	}
	resp, err := o.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("%s http: %w", o.name, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, readError(resp, o.name)
	}

	var raw openAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return nil, fmt.Errorf("%s decode: %w", o.name, err)
	}
	return raw.toCompletionResponse(), nil
}

// Stream runs a streaming chat completion, emitting deltas to sink.
func (o *OpenAI) Stream(ctx context.Context, req CompletionRequest, sink StreamSink) error {
	body, err := buildOpenAIRequest(req, true)
	if err != nil {
		return err
	}
	httpReq, err := o.newRequest(ctx, body)
	if err != nil {
		return err
	}
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := o.client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("%s http: %w", o.name, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return readError(resp, o.name)
	}
	return parseOpenAIStream(resp.Body, sink)
}

func (o *OpenAI) newRequest(ctx context.Context, body []byte) (*http.Request, error) {
	url := o.baseURL + "/v1/chat/completions"
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Authorization", "Bearer "+o.apiKey)
	for k, v := range o.headers {
		r.Header.Set(k, v)
	}
	return r, nil
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
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	TopP                *float64        `json:"top_p,omitempty"`
	Stop                []string        `json:"stop,omitempty"`
	Stream              bool            `json:"stream,omitempty"`
	ResponseFormat      *ResponseFormat `json:"response_format,omitempty"`
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
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

func (r *openAIResponse) toCompletionResponse() *CompletionResponse {
	out := &CompletionResponse{
		Usage: Usage{
			PromptTokens:     r.Usage.PromptTokens,
			CompletionTokens: r.Usage.CompletionTokens,
			TotalTokens:      r.Usage.TotalTokens,
		},
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

func parseOpenAIStream(body io.Reader, sink StreamSink) error {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	var finish string
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(line[len("data:"):])
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			return sink.Done(finish)
		}
		var chunk openAIStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return fmt.Errorf("openai stream chunk: %w", err)
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
	}
	if err := scanner.Err(); err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return err
		}
		return fmt.Errorf("openai stream scan: %w", err)
	}
	return sink.Done(finish)
}

var _ Provider = (*OpenAI)(nil)
