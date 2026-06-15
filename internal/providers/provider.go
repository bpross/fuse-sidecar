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
// they arrive and returning the token usage reported at end of stream.
// It is used for the final primary call. Usage is zero-valued if the
// provider did not report it.
type Provider interface {
	Name() string
	Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error)
	Stream(ctx context.Context, req CompletionRequest, sink Sink) (Usage, error)
}

// Sink receives streaming events from providers and from the fusion
// pipeline. It is the single emission interface for the whole system:
// providers stream deltas into it; the pipeline writes progress events
// into it as Delta{ReasoningContent: ...}; the server adapts it to SSE.
//
// Implementations must be safe to call from the goroutine that owns the
// caller (we do not call Sink methods from multiple goroutines per
// request).
type Sink interface {
	// Delta is called for each incremental chunk. Exactly one of Content,
	// ReasoningContent, or ToolCallDelta is non-empty per Delta.
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
	// CachePrefix requests that the provider mark the conversation prefix
	// (system + tools + messages, up to and including the last message) as
	// cacheable. Providers that support prefix caching (Anthropic) emit a
	// cache_control breakpoint; providers that cache automatically (OpenAI)
	// or not at all ignore the flag. The synthetic handoff turns appended
	// after fusion are intentionally added *after* this prefix so they do
	// not invalidate the cache shared with the speculative call.
	CachePrefix bool
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
//
// CacheReadTokens and CacheCreationTokens break out the prompt token spend
// across prompt caching: tokens served from a provider's prefix cache vs
// tokens written into it. Both providers surface these (Anthropic via
// cache_read_input_tokens / cache_creation_input_tokens, OpenAI via
// prompt_tokens_details.cached_tokens) and they are zero when caching did
// not apply.
type Usage struct {
	PromptTokens        int
	CompletionTokens    int
	TotalTokens         int
	CacheReadTokens     int
	CacheCreationTokens int
}

// Add accumulates another Usage into this one. Used to roll per-call usage
// up into a per-turn total.
func (u *Usage) Add(other Usage) {
	u.PromptTokens += other.PromptTokens
	u.CompletionTokens += other.CompletionTokens
	u.TotalTokens += other.TotalTokens
	u.CacheReadTokens += other.CacheReadTokens
	u.CacheCreationTokens += other.CacheCreationTokens
}

// Delta is one streaming chunk.
type Delta struct {
	Content          string
	ReasoningContent string // optional; opencode may or may not render
	ToolCallDelta    *ToolCallDelta
}

// Message is a single conversation message in OpenAI shape.
//
// CacheBreakpoint marks this message as the end of the cacheable prefix.
// It is an internal control flag, not part of the wire format (the json
// tag is "-"). When set, a cache-aware provider places its cache_control
// marker on this message instead of the last message. This lets the
// fusion pipeline cache the original conversation prefix while leaving the
// appended handoff turns uncached.
type Message struct {
	Role            string     `json:"role"` // "system" | "user" | "assistant" | "tool"
	Content         string     `json:"content,omitempty"`
	ToolCalls       []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID      string     `json:"tool_call_id,omitempty"`
	Name            string     `json:"name,omitempty"`
	CacheBreakpoint bool       `json:"-"`
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
