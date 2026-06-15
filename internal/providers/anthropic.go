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

// Anthropic implements Provider over the Messages API.
//
// Translation notes:
//   - OpenAI "system" messages are concatenated into Anthropic's top-level
//     "system" field; all other messages become Anthropic messages.
//   - OpenAI tool_calls in an assistant message become tool_use blocks.
//   - OpenAI tool messages become user messages containing a tool_result
//     block (this is how Anthropic models the round trip).
//   - Tools translate function-name + JSON-schema parameters into
//     Anthropic's input_schema.
type Anthropic struct {
	apiKey  string
	baseURL string
	headers map[string]string
	client  *http.Client
}

// AnthropicConfig contains construction parameters.
type AnthropicConfig struct {
	APIKey  string
	BaseURL string // defaults to https://api.anthropic.com
	Headers map[string]string
	Client  *http.Client
}

// NewAnthropic constructs an Anthropic provider.
func NewAnthropic(cfg AnthropicConfig) *Anthropic {
	base := cfg.BaseURL
	if base == "" {
		base = "https://api.anthropic.com"
	}
	c := cfg.Client
	if c == nil {
		c = &http.Client{Timeout: 0} // streaming needs no overall timeout; context handles it
	}
	return &Anthropic{
		apiKey:  cfg.APIKey,
		baseURL: strings.TrimRight(base, "/"),
		headers: cfg.Headers,
		client:  c,
	}
}

// Name returns the provider identifier.
func (a *Anthropic) Name() string { return "anthropic" }

// Complete runs a non-streaming Messages API call.
func (a *Anthropic) Complete(ctx context.Context, req CompletionRequest) (*CompletionResponse, error) {
	body, err := buildAnthropicRequest(req, false)
	if err != nil {
		return nil, err
	}
	httpReq, err := a.newRequest(ctx, body)
	if err != nil {
		return nil, err
	}
	resp, err := a.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode/100 != 2 {
		return nil, readError(resp, "anthropic")
	}

	var msg anthropicMessage
	if err := json.NewDecoder(resp.Body).Decode(&msg); err != nil {
		return nil, fmt.Errorf("anthropic decode: %w", err)
	}
	return msg.toCompletionResponse(), nil
}

// Stream runs a streaming Messages API call, emitting deltas to sink.
func (a *Anthropic) Stream(ctx context.Context, req CompletionRequest, sink StreamSink) error {
	body, err := buildAnthropicRequest(req, true)
	if err != nil {
		return err
	}
	httpReq, err := a.newRequest(ctx, body)
	if err != nil {
		return err
	}
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := a.client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("anthropic http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode/100 != 2 {
		return readError(resp, "anthropic")
	}
	return parseAnthropicStream(resp.Body, sink)
}

func (a *Anthropic) newRequest(ctx context.Context, body []byte) (*http.Request, error) {
	url := a.baseURL + "/v1/messages"
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("x-api-key", a.apiKey)
	r.Header.Set("anthropic-version", "2023-06-01")
	for k, v := range a.headers {
		r.Header.Set(k, v)
	}
	return r, nil
}

// readError reads up to 4KiB of an error body and wraps it.
func readError(resp *http.Response, provider string) error {
	const maxErrBody = 4096
	limited := io.LimitReader(resp.Body, maxErrBody)
	b, _ := io.ReadAll(limited)
	return fmt.Errorf("%s http %d: %s", provider, resp.StatusCode, strings.TrimSpace(string(b)))
}

// --- Request translation ---

type anthropicRequest struct {
	Model       string             `json:"model"`
	System      string             `json:"system,omitempty"`
	Messages    []anthropicMessage `json:"messages"`
	MaxTokens   int                `json:"max_tokens"`
	Temperature *float64           `json:"temperature,omitempty"`
	TopP        *float64           `json:"top_p,omitempty"`
	StopSeq     []string           `json:"stop_sequences,omitempty"`
	Stream      bool               `json:"stream,omitempty"`
	Tools       []anthropicTool    `json:"tools,omitempty"`
	ToolChoice  json.RawMessage    `json:"tool_choice,omitempty"`
}

type anthropicMessage struct {
	Role    string           `json:"role"`
	Content []anthropicBlock `json:"content"`
	// Fields populated on the response path:
	ID         string         `json:"id,omitempty"`
	Type       string         `json:"type,omitempty"`
	Model      string         `json:"model,omitempty"`
	StopReason string         `json:"stop_reason,omitempty"`
	Usage      *anthropicUsage `json:"usage,omitempty"`
}

type anthropicBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"` // tool_result content
	// streaming-only:
	PartialJSON string `json:"partial_json,omitempty"`
}

type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

func buildAnthropicRequest(req CompletionRequest, stream bool) ([]byte, error) {
	systemParts, msgs := splitSystem(req.Messages)

	out := anthropicRequest{
		Model:       req.Model,
		System:      strings.Join(systemParts, "\n\n"),
		Messages:    make([]anthropicMessage, 0, len(msgs)),
		Temperature: req.Temperature,
		TopP:        req.TopP,
		StopSeq:     req.Stop,
		Stream:      stream,
	}

	// Anthropic requires max_tokens. Pick a reasonable cap if the caller
	// didn't specify; favor a generous value because plans can be long.
	if req.MaxTokens != nil {
		out.MaxTokens = *req.MaxTokens
	} else {
		out.MaxTokens = 8192
	}

	// Translate messages into Anthropic blocks. We pair adjacent assistant
	// tool_calls + the matching tool responses into Anthropic's
	// tool_use / tool_result block sequence.
	for _, m := range msgs {
		switch m.Role {
		case "user":
			out.Messages = append(out.Messages, anthropicMessage{
				Role:    "user",
				Content: []anthropicBlock{{Type: "text", Text: m.Content}},
			})
		case "assistant":
			blocks := make([]anthropicBlock, 0, 1+len(m.ToolCalls))
			if m.Content != "" {
				blocks = append(blocks, anthropicBlock{Type: "text", Text: m.Content})
			}
			for _, tc := range m.ToolCalls {
				blocks = append(blocks, anthropicBlock{
					Type:  "tool_use",
					ID:    tc.ID,
					Name:  tc.Function.Name,
					Input: json.RawMessage(orEmptyObject(tc.Function.Arguments)),
				})
			}
			out.Messages = append(out.Messages, anthropicMessage{
				Role:    "assistant",
				Content: blocks,
			})
		case "tool":
			// Tool responses become user messages with a tool_result block.
			// Anthropic expects "content" to be either a string or an array
			// of blocks; we send the OpenAI string content as a raw JSON string.
			raw, err := json.Marshal(m.Content)
			if err != nil {
				return nil, err
			}
			out.Messages = append(out.Messages, anthropicMessage{
				Role: "user",
				Content: []anthropicBlock{{
					Type:      "tool_result",
					ToolUseID: m.ToolCallID,
					Content:   raw,
				}},
			})
		case "system":
			// Already accumulated above; ignore here.
		default:
			return nil, fmt.Errorf("anthropic: unsupported role %q", m.Role)
		}
	}

	// Translate OpenAI tool_choice into Anthropic's shape. If the caller
	// sent "none" we drop tools entirely; for anything else we set
	// tool_choice and pass tools through.
	toolChoice, dropTools, err := translateToolChoice(req.ToolChoice)
	if err != nil {
		return nil, err
	}

	if !dropTools && len(req.Tools) > 0 {
		out.Tools = make([]anthropicTool, 0, len(req.Tools))
		for _, t := range req.Tools {
			if t.Type != "function" {
				return nil, fmt.Errorf("anthropic: unsupported tool type %q", t.Type)
			}
			out.Tools = append(out.Tools, anthropicTool{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				InputSchema: orEmptySchema(t.Function.Parameters),
			})
		}
		if len(toolChoice) > 0 {
			out.ToolChoice = toolChoice
		}
	}

	return json.Marshal(out)
}

// translateToolChoice converts an OpenAI-shaped tool_choice into Anthropic's
// form. OpenAI accepts:
//   - "none"     → no tools should be called (we drop the tools array)
//   - "auto"     → model decides (Anthropic default; we can omit or send {"type":"auto"})
//   - "required" → must call some tool (Anthropic: {"type":"any"})
//   - {"type":"function","function":{"name":"X"}} → force a specific tool
//
// Returns the Anthropic-shaped tool_choice (or nil to omit), a flag indicating
// whether to drop the tools array entirely, and any translation error.
func translateToolChoice(raw json.RawMessage) (json.RawMessage, bool, error) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" {
		return nil, false, nil
	}

	// String form: "auto" | "none" | "required"
	var s string
	if err := json.Unmarshal([]byte(trimmed), &s); err == nil {
		switch s {
		case "none":
			return nil, true, nil
		case "auto":
			return json.RawMessage(`{"type":"auto"}`), false, nil
		case "required":
			return json.RawMessage(`{"type":"any"}`), false, nil
		default:
			return nil, false, fmt.Errorf("anthropic: unsupported tool_choice string %q", s)
		}
	}

	// Object form: {"type":"function","function":{"name":"X"}}
	var obj struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
		Name string `json:"name"` // already-Anthropic-shaped passthrough
	}
	if err := json.Unmarshal([]byte(trimmed), &obj); err != nil {
		return nil, false, fmt.Errorf("anthropic: tool_choice not a string or object: %w", err)
	}
	switch obj.Type {
	case "function":
		if obj.Function.Name == "" {
			return nil, false, fmt.Errorf("anthropic: tool_choice function.name is required")
		}
		out, _ := json.Marshal(map[string]string{"type": "tool", "name": obj.Function.Name})
		return out, false, nil
	case "auto", "any":
		// Already in Anthropic shape; pass through.
		return json.RawMessage(trimmed), false, nil
	case "tool":
		// Already Anthropic-shaped; pass through.
		if obj.Name == "" {
			return nil, false, fmt.Errorf("anthropic: tool_choice type=tool requires name")
		}
		return json.RawMessage(trimmed), false, nil
	default:
		return nil, false, fmt.Errorf("anthropic: unsupported tool_choice type %q", obj.Type)
	}
}

func splitSystem(msgs []Message) (system []string, rest []Message) {
	rest = make([]Message, 0, len(msgs))
	for _, m := range msgs {
		if m.Role == "system" {
			if m.Content != "" {
				system = append(system, m.Content)
			}
			continue
		}
		rest = append(rest, m)
	}
	return system, rest
}

func orEmptyObject(s string) string {
	if strings.TrimSpace(s) == "" {
		return "{}"
	}
	return s
}

func orEmptySchema(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return json.RawMessage(`{"type":"object","properties":{}}`)
	}
	return raw
}

// --- Response translation ---

func (m *anthropicMessage) toCompletionResponse() *CompletionResponse {
	out := &CompletionResponse{}
	for _, b := range m.Content {
		switch b.Type {
		case "text":
			out.Content += b.Text
		case "tool_use":
			out.ToolCalls = append(out.ToolCalls, ToolCall{
				ID:   b.ID,
				Type: "function",
				Function: FunctionCall{
					Name:      b.Name,
					Arguments: string(b.Input),
				},
			})
		}
	}
	out.FinishReason = mapStopReason(m.StopReason)
	if m.Usage != nil {
		out.Usage = Usage{
			PromptTokens:     m.Usage.InputTokens,
			CompletionTokens: m.Usage.OutputTokens,
			TotalTokens:      m.Usage.InputTokens + m.Usage.OutputTokens,
		}
	}
	return out
}

func mapStopReason(s string) string {
	switch s {
	case "end_turn":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	case "stop_sequence":
		return "stop"
	case "":
		return ""
	default:
		return s
	}
}

// --- Streaming ---

// parseAnthropicStream consumes the SSE stream from Messages API and emits
// OpenAI-shaped deltas. Anthropic events we care about:
//
//   message_start              -> usage prompt tokens
//   content_block_start        -> begin text or tool_use block
//   content_block_delta        -> text delta or input_json_delta
//   content_block_stop         -> end of a block (no emit needed)
//   message_delta              -> stop_reason + usage output tokens
//   message_stop               -> end of stream
//   ping                       -> ignore
func parseAnthropicStream(body io.Reader, sink StreamSink) error {
	scanner := bufio.NewScanner(body)
	// SSE lines can be long; raise the buffer.
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	var (
		stopReason string
		// Track active block: index, type, and (for tool_use) the tool call delta state.
		blockTypes = map[int]string{}
		toolCalls  = map[int]*ToolCallDelta{}
	)

	var eventName string
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			eventName = ""
			continue
		}
		if strings.HasPrefix(line, ":") {
			continue // SSE comment / keepalive
		}
		if strings.HasPrefix(line, "event:") {
			eventName = strings.TrimSpace(line[len("event:"):])
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(line[len("data:"):])
		if data == "" {
			continue
		}

		switch eventName {
		case "message_start":
			// ignore for now; could surface usage
		case "content_block_start":
			var ev struct {
				Index        int            `json:"index"`
				ContentBlock anthropicBlock `json:"content_block"`
			}
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				return fmt.Errorf("anthropic stream content_block_start: %w", err)
			}
			blockTypes[ev.Index] = ev.ContentBlock.Type
			if ev.ContentBlock.Type == "tool_use" {
				tcd := &ToolCallDelta{
					Index:        ev.Index,
					ID:           ev.ContentBlock.ID,
					FunctionName: ev.ContentBlock.Name,
				}
				toolCalls[ev.Index] = tcd
				if err := sink.Delta(Delta{ToolCallDelta: tcd}); err != nil {
					return err
				}
			}
		case "content_block_delta":
			var ev struct {
				Index int `json:"index"`
				Delta struct {
					Type        string `json:"type"`
					Text        string `json:"text,omitempty"`
					PartialJSON string `json:"partial_json,omitempty"`
				} `json:"delta"`
			}
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				return fmt.Errorf("anthropic stream content_block_delta: %w", err)
			}
			switch ev.Delta.Type {
			case "text_delta":
				if ev.Delta.Text != "" {
					if err := sink.Delta(Delta{Content: ev.Delta.Text}); err != nil {
						return err
					}
				}
			case "input_json_delta":
				if ev.Delta.PartialJSON != "" {
					if err := sink.Delta(Delta{ToolCallDelta: &ToolCallDelta{
						Index:            ev.Index,
						ArgumentsPartial: ev.Delta.PartialJSON,
					}}); err != nil {
						return err
					}
				}
			}
		case "content_block_stop":
			// nothing to emit
		case "message_delta":
			var ev struct {
				Delta struct {
					StopReason string `json:"stop_reason"`
				} `json:"delta"`
			}
			if err := json.Unmarshal([]byte(data), &ev); err == nil {
				if ev.Delta.StopReason != "" {
					stopReason = ev.Delta.StopReason
				}
			}
		case "message_stop":
			return sink.Done(mapStopReason(stopReason))
		case "ping", "":
			// ignore
		default:
			// Unknown event; ignore for forward compatibility.
		}
	}
	if err := scanner.Err(); err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return err
		}
		return fmt.Errorf("anthropic stream scan: %w", err)
	}
	// Stream ended without message_stop; treat as a normal finish.
	return sink.Done(mapStopReason(stopReason))
}

// Compile-time check.
var _ Provider = (*Anthropic)(nil)
