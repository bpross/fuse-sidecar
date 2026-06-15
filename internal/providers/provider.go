// Package providers abstracts over upstream LLM APIs.
//
// The shared types here are deliberately shaped like OpenAI ChatCompletions
// because that is the sidecar's wire surface — every Provider implementation
// translates to and from its native API. Translation that is provider-
// specific lives in that provider's file.
package providers

import (
	"context"
	"encoding/json"
)

// Provider is one upstream LLM API.
//
// Complete runs a single non-streaming completion and returns the full
// response. It is used for the speculative primary call, the panel calls,
// and the judge call.
//
// Stream runs a single streaming completion, emitting deltas to sink as
// they arrive. It is used for the final primary call.
type Provider interface {
	Name() string
	Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error)
	Stream(ctx context.Context, req CompletionRequest, sink StreamSink) error
}

// StreamSink receives streaming deltas. Implementations must be safe to
// call from the goroutine that owns Provider.Stream (we do not call sink
// from multiple goroutines per request).
type StreamSink interface {
	// Delta is called for each incremental chunk. Either Content or
	// ToolCallDelta or ReasoningContent is non-empty; never more than one.
	Delta(d Delta) error
	// Done is called exactly once at end of stream with the finish reason.
	Done(finishReason string) error
}

// CompletionRequest is the provider-agnostic request shape.
type CompletionRequest struct {
	Model       string
	Messages    []Message
	Tools       []Tool
	ToolChoice  json.RawMessage
	Temperature *float64
	MaxTokens   *int
	TopP        *float64
	Stop        []string
	// ResponseFormat is used by the judge call to coerce structured output.
	ResponseFormat *ResponseFormat
}

// CompletionResponse is the provider-agnostic response shape, modeled on
// OpenAI's choice[0] for simplicity. Provider implementations are
// responsible for collapsing multi-choice or multi-block responses into
// this shape.
type CompletionResponse struct {
	Content      string
	ToolCalls    []ToolCall
	FinishReason string // "stop" | "length" | "tool_calls" | "content_filter" | provider-specific
	Usage        Usage
}

// Usage is token counts when available; zero values are fine.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// Delta is one streaming chunk.
type Delta struct {
	Content          string
	ReasoningContent string // optional; opencode may or may not render
	ToolCallDelta    *ToolCallDelta
}

// Message is a single conversation message in OpenAI shape.
type Message struct {
	Role       string     `json:"role"` // "system" | "user" | "assistant" | "tool"
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

// Tool is one function-tool definition.
type Tool struct {
	Type     string         `json:"type"` // always "function" for now
	Function ToolDefinition `json:"function"`
}

// ToolDefinition is the function-tool body.
type ToolDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// ToolCall is a fully-formed tool invocation in an assistant message.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"` // always "function" for now
	Function FunctionCall `json:"function"`
}

// FunctionCall is the function-call body.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string; preserve verbatim
}

// ToolCallDelta is one incremental piece of a tool call during streaming.
// Providers buffer-and-accumulate these so the consumer sees them in OpenAI
// streaming shape.
type ToolCallDelta struct {
	Index            int
	ID               string
	FunctionName     string
	ArgumentsPartial string
}

// ResponseFormat carries OpenAI's response_format directive.
type ResponseFormat struct {
	Type       string          `json:"type"` // "text" | "json_object" | "json_schema"
	JSONSchema json.RawMessage `json:"json_schema,omitempty"`
}

// Registry resolves a provider name to its implementation.
type Registry struct {
	byName map[string]Provider
}

// NewRegistry constructs an empty registry.
func NewRegistry() *Registry {
	return &Registry{byName: make(map[string]Provider)}
}

// Register adds a provider. Last registration wins for a given name.
func (r *Registry) Register(p Provider) {
	r.byName[p.Name()] = p
}

// Get returns the provider with the given name and whether it was found.
func (r *Registry) Get(name string) (Provider, bool) {
	p, ok := r.byName[name]
	return p, ok
}
