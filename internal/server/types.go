// Package server implements the OpenAI-compatible HTTP surface.
//
// The wire types here mirror OpenAI's ChatCompletions API. They are the
// sidecar's external contract — any client that targets these types can
// drive the sidecar regardless of which upstream provider answers.
package server

import (
	"encoding/json"

	"github.com/bpross/fuse-sidecar/internal/providers"
)

// ChatRequest is the body of POST /v1/chat/completions.
type ChatRequest struct {
	Model          string                    `json:"model"`
	Messages       []providers.Message       `json:"messages"`
	Tools          []providers.Tool          `json:"tools,omitempty"`
	ToolChoice     json.RawMessage           `json:"tool_choice,omitempty"`
	Stream         bool                      `json:"stream,omitempty"`
	Temperature    *float64                  `json:"temperature,omitempty"`
	MaxTokens      *int                      `json:"max_tokens,omitempty"`
	TopP           *float64                  `json:"top_p,omitempty"`
	Stop           []string                  `json:"stop,omitempty"`
	ResponseFormat *providers.ResponseFormat `json:"response_format,omitempty"`
	// Unknown fields are accepted; we don't DisallowUnknownFields here because
	// clients (e.g. opencode via the AI SDK) attach provider-specific extras
	// that we want to forward or ignore gracefully.
}

// ChatChoice is one choice in a non-streaming response.
type ChatChoice struct {
	Index        int             `json:"index"`
	Message      AssistantPayload `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

// AssistantPayload is the assistant message in a response.
type AssistantPayload struct {
	Role      string               `json:"role"`
	Content   string               `json:"content"`
	ToolCalls []providers.ToolCall `json:"tool_calls,omitempty"`
}

// ChatResponse is a non-streaming response. We don't currently emit these;
// the type is here for completeness and future use.
type ChatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
	Usage   UsageOut     `json:"usage"`
}

// UsageOut is the OpenAI usage block.
type UsageOut struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// StreamChunk is one SSE event payload (the "data:" value).
type StreamChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"` // always "chat.completion.chunk"
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []StreamChoice `json:"choices"`
}

// StreamChoice is one choice's delta.
type StreamChoice struct {
	Index        int          `json:"index"`
	Delta        StreamDelta  `json:"delta"`
	FinishReason *string      `json:"finish_reason"`
}

// StreamDelta is the incremental change for one choice. OpenAI shape: each
// chunk has one of role/content/tool_calls set (or empty). Reasoning is a
// best-effort extension (some providers/SDKs render it; others ignore).
type StreamDelta struct {
	Role             string            `json:"role,omitempty"`
	Content          string            `json:"content,omitempty"`
	ReasoningContent string            `json:"reasoning_content,omitempty"`
	ToolCalls        []StreamToolCall  `json:"tool_calls,omitempty"`
}

// StreamToolCall is one tool-call delta. Index is required so the client
// can match deltas to the right slot as they accumulate.
type StreamToolCall struct {
	Index    int                  `json:"index"`
	ID       string               `json:"id,omitempty"`
	Type     string               `json:"type,omitempty"`
	Function StreamFunctionCall   `json:"function,omitempty"`
}

// StreamFunctionCall is the function-call delta payload.
type StreamFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// ModelListResponse is the body of GET /v1/models.
type ModelListResponse struct {
	Object string         `json:"object"` // "list"
	Data   []ModelInfo    `json:"data"`
}

// ModelInfo is one entry in /v1/models.
type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // "model"
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ErrorResponse is the OpenAI-shaped error envelope.
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail is the OpenAI error body.
type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}
