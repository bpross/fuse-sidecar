package providers

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestBuildAnthropicRequestBasic(t *testing.T) {
	maxTok := 1000
	temp := 0.2
	body, err := buildAnthropicRequest(CompletionRequest{
		Model: "claude-opus-4-7",
		Messages: []Message{
			{Role: "system", Content: "you are a helper"},
			{Role: "user", Content: "hi"},
		},
		MaxTokens:   &maxTok,
		Temperature: &temp,
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatal(err)
	}
	if got["system"] != "you are a helper" {
		t.Errorf("system = %v", got["system"])
	}
	if got["max_tokens"].(float64) != 1000 {
		t.Errorf("max_tokens = %v", got["max_tokens"])
	}
	if got["temperature"].(float64) != 0.2 {
		t.Errorf("temperature = %v", got["temperature"])
	}
	msgs := got["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	first := msgs[0].(map[string]any)
	if first["role"] != "user" {
		t.Errorf("role = %v", first["role"])
	}
}

func TestBuildAnthropicRequestToolCallTurn(t *testing.T) {
	body, err := buildAnthropicRequest(CompletionRequest{
		Model: "claude-opus-4-7",
		Messages: []Message{
			{Role: "user", Content: "read foo.txt"},
			{
				Role: "assistant",
				ToolCalls: []ToolCall{{
					ID:   "call_1",
					Type: "function",
					Function: FunctionCall{
						Name:      "read",
						Arguments: `{"path":"foo.txt"}`,
					},
				}},
			},
			{Role: "tool", ToolCallID: "call_1", Content: "hello"},
			{Role: "user", Content: "what did it say?"},
		},
		Tools: []Tool{{
			Type: "function",
			Function: ToolDefinition{
				Name:        "read",
				Description: "read a file",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`),
			},
		}},
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatal(err)
	}
	msgs := got["messages"].([]any)
	if len(msgs) != 4 {
		t.Fatalf("messages len = %d, want 4 (user, assistant-with-tool_use, user-with-tool_result, user)", len(msgs))
	}
	// assistant message should contain a tool_use block
	assistant := msgs[1].(map[string]any)
	content := assistant["content"].([]any)
	tu := content[0].(map[string]any)
	if tu["type"] != "tool_use" {
		t.Errorf("expected tool_use, got %v", tu["type"])
	}
	if tu["id"] != "call_1" {
		t.Errorf("tool_use id = %v", tu["id"])
	}
	// tool message should be user/tool_result
	toolMsg := msgs[2].(map[string]any)
	if toolMsg["role"] != "user" {
		t.Errorf("tool message role = %v, want user", toolMsg["role"])
	}
	tr := toolMsg["content"].([]any)[0].(map[string]any)
	if tr["type"] != "tool_result" {
		t.Errorf("expected tool_result, got %v", tr["type"])
	}
	if tr["tool_use_id"] != "call_1" {
		t.Errorf("tool_use_id = %v", tr["tool_use_id"])
	}
	tools := got["tools"].([]any)
	if len(tools) != 1 {
		t.Errorf("tools len = %d", len(tools))
	}
}

func TestMapStopReason(t *testing.T) {
	cases := map[string]string{
		"end_turn":      "stop",
		"max_tokens":    "length",
		"tool_use":      "tool_calls",
		"stop_sequence": "stop",
		"":              "",
		"weird":         "weird",
	}
	for in, want := range cases {
		if got := mapStopReason(in); got != want {
			t.Errorf("mapStopReason(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestAnthropicCompleteHappyPath(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("missing api key header")
		}
		if r.Header.Get("anthropic-version") == "" {
			t.Errorf("missing anthropic-version header")
		}
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"model":"claude-opus-4-7"`) {
			t.Errorf("body missing model: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
		  "id": "msg_1",
		  "type": "message",
		  "role": "assistant",
		  "model": "claude-opus-4-7",
		  "content": [{"type":"text","text":"hello world"}],
		  "stop_reason": "end_turn",
		  "usage": {"input_tokens": 10, "output_tokens": 5}
		}`))
	}))
	defer srv.Close()

	a := NewAnthropic(AnthropicConfig{
		APIKey:  "test-key",
		BaseURL: srv.URL,
	})
	maxTok := 100
	resp, err := a.Complete(context.Background(), CompletionRequest{
		Model:     "claude-opus-4-7",
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: &maxTok,
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "hello world" {
		t.Errorf("content = %q", resp.Content)
	}
	if resp.FinishReason != "stop" {
		t.Errorf("finish_reason = %q", resp.FinishReason)
	}
	if resp.Usage.PromptTokens != 10 || resp.Usage.CompletionTokens != 5 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestAnthropicCompleteToolUse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
		  "content": [
		    {"type":"text","text":"thinking..."},
		    {"type":"tool_use","id":"toolu_x","name":"read","input":{"path":"a"}}
		  ],
		  "stop_reason": "tool_use"
		}`))
	}))
	defer srv.Close()

	a := NewAnthropic(AnthropicConfig{APIKey: "k", BaseURL: srv.URL})
	resp, err := a.Complete(context.Background(), CompletionRequest{
		Model:    "claude-opus-4-7",
		Messages: []Message{{Role: "user", Content: "go"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.FinishReason != "tool_calls" {
		t.Errorf("finish_reason = %q", resp.FinishReason)
	}
	if resp.Content != "thinking..." {
		t.Errorf("content = %q", resp.Content)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("tool_calls len = %d", len(resp.ToolCalls))
	}
	tc := resp.ToolCalls[0]
	if tc.ID != "toolu_x" || tc.Function.Name != "read" || tc.Function.Arguments != `{"path":"a"}` {
		t.Errorf("tool call = %+v", tc)
	}
}

func TestAnthropicStreamHappyPath(t *testing.T) {
	sseBody := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_1"}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":", world"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, sseBody)
	}))
	defer srv.Close()

	a := NewAnthropic(AnthropicConfig{APIKey: "k", BaseURL: srv.URL})
	sink := &captureSink{}
	if err := a.Stream(context.Background(), CompletionRequest{
		Model:    "claude-opus-4-7",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}, sink); err != nil {
		t.Fatal(err)
	}

	if got := strings.Join(sink.contents, ""); got != "Hello, world" {
		t.Errorf("text content = %q", got)
	}
	if sink.finish != "stop" {
		t.Errorf("finish = %q", sink.finish)
	}
}

func TestAnthropicStreamToolUse(t *testing.T) {
	sseBody := strings.Join([]string{
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_a","name":"read","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"foo.txt\"}"}}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, sseBody)
	}))
	defer srv.Close()

	a := NewAnthropic(AnthropicConfig{APIKey: "k", BaseURL: srv.URL})
	sink := &captureSink{}
	if err := a.Stream(context.Background(), CompletionRequest{
		Model:    "claude-opus-4-7",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}, sink); err != nil {
		t.Fatal(err)
	}
	if sink.finish != "tool_calls" {
		t.Errorf("finish = %q", sink.finish)
	}
	if len(sink.toolDeltas) < 3 {
		t.Fatalf("expected at least 3 tool deltas (start + 2 partials), got %d", len(sink.toolDeltas))
	}
	if sink.toolDeltas[0].ID != "toolu_a" || sink.toolDeltas[0].FunctionName != "read" {
		t.Errorf("first tool delta should announce id/name, got %+v", sink.toolDeltas[0])
	}
	args := ""
	for _, d := range sink.toolDeltas[1:] {
		args += d.ArgumentsPartial
	}
	if args != `{"path":"foo.txt"}` {
		t.Errorf("accumulated args = %q", args)
	}
}

type captureSink struct {
	contents   []string
	toolDeltas []ToolCallDelta
	reasoning  []string
	finish     string
}

func (c *captureSink) Delta(d Delta) error {
	if d.Content != "" {
		c.contents = append(c.contents, d.Content)
	}
	if d.ReasoningContent != "" {
		c.reasoning = append(c.reasoning, d.ReasoningContent)
	}
	if d.ToolCallDelta != nil {
		c.toolDeltas = append(c.toolDeltas, *d.ToolCallDelta)
	}
	return nil
}

func (c *captureSink) Done(reason string) error {
	c.finish = reason
	return nil
}
