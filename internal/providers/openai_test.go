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

func TestOpenAICompleteHappyPath(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("auth = %q", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
		  "id":"chatcmpl-1",
		  "choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
		  "usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}
		}`)
	}))
	defer srv.Close()

	o := NewOpenAI(OpenAIConfig{APIKey: "test-key", BaseURL: srv.URL})
	resp, err := o.Complete(context.Background(), CompletionRequest{
		Model:    "gpt-5",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "hello" {
		t.Errorf("content = %q", resp.Content)
	}
	if resp.FinishReason != "stop" {
		t.Errorf("finish = %q", resp.FinishReason)
	}
	if resp.Usage.TotalTokens != 6 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestOpenAICompleteToolCall(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
		  "choices":[{
		    "index":0,
		    "message":{
		      "role":"assistant",
		      "content":null,
		      "tool_calls":[{
		        "id":"call_1",
		        "type":"function",
		        "function":{"name":"read","arguments":"{\"path\":\"x\"}"}
		      }]
		    },
		    "finish_reason":"tool_calls"
		  }]
		}`)
	}))
	defer srv.Close()

	o := NewOpenAI(OpenAIConfig{APIKey: "k", BaseURL: srv.URL})
	resp, err := o.Complete(context.Background(), CompletionRequest{
		Model:    "gpt-5",
		Messages: []Message{{Role: "user", Content: "read x"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.FinishReason != "tool_calls" {
		t.Errorf("finish = %q", resp.FinishReason)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("tool_calls = %d", len(resp.ToolCalls))
	}
	tc := resp.ToolCalls[0]
	if tc.Function.Name != "read" || tc.Function.Arguments != `{"path":"x"}` {
		t.Errorf("tool call = %+v", tc)
	}
}

func TestOpenAIStreamHappyPath(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant"}}]}`,
		``,
		`data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hello"}}]}`,
		``,
		`data: {"id":"x","choices":[{"index":0,"delta":{"content":", world"}}]}`,
		``,
		`data: {"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, sse)
	}))
	defer srv.Close()

	o := NewOpenAI(OpenAIConfig{APIKey: "k", BaseURL: srv.URL})
	sink := &captureSink{}
	if _, err := o.Stream(context.Background(), CompletionRequest{
		Model:    "gpt-5",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}, sink); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(sink.contents, ""); got != "Hello, world" {
		t.Errorf("content = %q", got)
	}
	if sink.finish != "stop" {
		t.Errorf("finish = %q", sink.finish)
	}
}

func TestOpenAIStreamToolCallDeltas(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read"}}]}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"pa"}}]}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"x\"}"}}]}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, sse)
	}))
	defer srv.Close()

	o := NewOpenAI(OpenAIConfig{APIKey: "k", BaseURL: srv.URL})
	sink := &captureSink{}
	if _, err := o.Stream(context.Background(), CompletionRequest{
		Model:    "gpt-5",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}, sink); err != nil {
		t.Fatal(err)
	}
	if sink.finish != "tool_calls" {
		t.Errorf("finish = %q", sink.finish)
	}
	if len(sink.toolDeltas) < 3 {
		t.Fatalf("got %d tool deltas", len(sink.toolDeltas))
	}
	if sink.toolDeltas[0].ID != "call_1" {
		t.Errorf("first delta id = %q", sink.toolDeltas[0].ID)
	}
	args := ""
	for _, d := range sink.toolDeltas {
		args += d.ArgumentsPartial
	}
	if args != `{"path":"x"}` {
		t.Errorf("accumulated args = %q", args)
	}
}

func TestOpenAIMaxCompletionTokensRouting(t *testing.T) {
	cases := []struct {
		model    string
		wantNew  bool
	}{
		{"gpt-4", false},
		{"gpt-4o", false},
		{"gpt-4o-mini", false},
		{"gpt-4.1", false},
		{"gpt-5", true},
		{"gpt-5-mini", true},
		{"gpt-5-codex", true},
		{"gpt-5-pro", true},
		{"o1", true},
		{"o1-mini", true},
		{"o3", true},
		{"o3-mini", true},
		{"o4-mini", true},
	}
	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			max := 1234
			body, err := buildOpenAIRequest(CompletionRequest{
				Model:     tc.model,
				Messages:  []Message{{Role: "user", Content: "hi"}},
				MaxTokens: &max,
			}, false)
			if err != nil {
				t.Fatal(err)
			}
			var got map[string]any
			if err := json.Unmarshal(body, &got); err != nil {
				t.Fatal(err)
			}
			if tc.wantNew {
				if _, ok := got["max_completion_tokens"]; !ok {
					t.Errorf("expected max_completion_tokens for %s, body=%v", tc.model, got)
				}
				if _, ok := got["max_tokens"]; ok {
					t.Errorf("expected NO max_tokens for %s, body=%v", tc.model, got)
				}
			} else {
				if _, ok := got["max_tokens"]; !ok {
					t.Errorf("expected max_tokens for %s, body=%v", tc.model, got)
				}
				if _, ok := got["max_completion_tokens"]; ok {
					t.Errorf("expected NO max_completion_tokens for %s, body=%v", tc.model, got)
				}
			}
		})
	}
}

func TestOpenAICompleteCacheUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
		  "choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
		  "usage":{"prompt_tokens":5000,"completion_tokens":20,"total_tokens":5020,"prompt_tokens_details":{"cached_tokens":4500}}
		}`)
	}))
	defer srv.Close()

	o := NewOpenAI(OpenAIConfig{APIKey: "k", BaseURL: srv.URL})
	resp, err := o.Complete(context.Background(), CompletionRequest{
		Model:    "gpt-5",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	// prompt_tokens includes cached; we subtract so PromptTokens is uncached.
	if resp.Usage.PromptTokens != 500 {
		t.Errorf("uncached prompt = %d, want 500", resp.Usage.PromptTokens)
	}
	if resp.Usage.CacheReadTokens != 4500 {
		t.Errorf("cache_read = %d", resp.Usage.CacheReadTokens)
	}
}

func TestOpenAIStreamUsage(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		``,
		`data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		``,
		`data: {"choices":[],"usage":{"prompt_tokens":3000,"completion_tokens":15,"total_tokens":3015,"prompt_tokens_details":{"cached_tokens":2000}}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Confirm we asked for usage on the stream.
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"include_usage":true`) {
			t.Errorf("stream request missing include_usage: %s", body)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, sse)
	}))
	defer srv.Close()

	o := NewOpenAI(OpenAIConfig{APIKey: "k", BaseURL: srv.URL})
	sink := &captureSink{}
	usage, err := o.Stream(context.Background(), CompletionRequest{
		Model:    "gpt-5",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}, sink)
	if err != nil {
		t.Fatal(err)
	}
	if usage.PromptTokens != 1000 {
		t.Errorf("uncached prompt = %d, want 1000", usage.PromptTokens)
	}
	if usage.CacheReadTokens != 2000 {
		t.Errorf("cache_read = %d", usage.CacheReadTokens)
	}
	if usage.CompletionTokens != 15 {
		t.Errorf("output = %d", usage.CompletionTokens)
	}
}

func TestOpenAICompatibleNameOverride(t *testing.T) {
	p := NewOpenAICompatible("openrouter", OpenAIConfig{APIKey: "k", BaseURL: "https://example.com"})
	if p.Name() != "openrouter" {
		t.Errorf("name = %q, want openrouter", p.Name())
	}
}
