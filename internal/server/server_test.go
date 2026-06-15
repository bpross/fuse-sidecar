package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/bpross/fuse-sidecar/internal/config"
	"github.com/bpross/fuse-sidecar/internal/obs"
	"github.com/bpross/fuse-sidecar/internal/providers"
)

func TestHealthz(t *testing.T) {
	s := newTestServer(t, &mockProvider{})
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rr := httptest.NewRecorder()
	s.Handler().ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("status = %d", rr.Code)
	}
	if !strings.Contains(rr.Body.String(), `"ok":true`) {
		t.Errorf("body = %q", rr.Body.String())
	}
}

func TestModelsList(t *testing.T) {
	s := newTestServer(t, &mockProvider{})
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rr := httptest.NewRecorder()
	s.Handler().ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Errorf("status = %d", rr.Code)
	}
	if !strings.Contains(rr.Body.String(), `"fusion-plan"`) {
		t.Errorf("body = %q", rr.Body.String())
	}
}

func TestUnknownModel(t *testing.T) {
	s := newTestServer(t, &mockProvider{})
	body, _ := json.Marshal(ChatRequest{Model: "nope", Stream: true, Messages: []providers.Message{{Role: "user", Content: "hi"}}})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	s.Handler().ServeHTTP(rr, req)
	if rr.Code != http.StatusNotFound {
		t.Errorf("status = %d, body=%s", rr.Code, rr.Body.String())
	}
}

func TestNonStreamingRejected(t *testing.T) {
	s := newTestServer(t, &mockProvider{})
	body, _ := json.Marshal(ChatRequest{Model: "fusion-plan", Stream: false, Messages: []providers.Message{{Role: "user", Content: "hi"}}})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	s.Handler().ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("status = %d", rr.Code)
	}
}

// TestPassthroughOnToolCalls verifies that a primary returning tool_calls
// triggers the passthrough branch and emits the tool deltas as SSE.
func TestPassthroughOnToolCalls(t *testing.T) {
	mp := &mockProvider{
		completeByModel: map[string]*providers.CompletionResponse{
			"mock-primary": {
				Content:      "let me check",
				FinishReason: "tool_calls",
				ToolCalls: []providers.ToolCall{{
					ID:   "call_1",
					Type: "function",
					Function: providers.FunctionCall{
						Name:      "read",
						Arguments: `{"path":"foo"}`,
					},
				}},
			},
		},
	}

	s := newTestServer(t, mp)
	body, _ := json.Marshal(ChatRequest{
		Model:    "fusion-plan",
		Stream:   true,
		Messages: []providers.Message{{Role: "user", Content: "read foo"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	s.Handler().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, body=%s", rr.Code, rr.Body.String())
	}
	out := rr.Body.String()
	if !strings.Contains(out, `"content":"let me check"`) {
		t.Errorf("missing content: %s", out)
	}
	if !strings.Contains(out, `"name":"read"`) {
		t.Errorf("missing tool name: %s", out)
	}
	if !strings.Contains(out, `"arguments":"{\"path\":\"foo\"}"`) {
		t.Errorf("missing tool args: %s", out)
	}
	if !strings.Contains(out, `"finish_reason":"tool_calls"`) {
		t.Errorf("missing finish: %s", out)
	}
	if !strings.Contains(out, "data: [DONE]") {
		t.Errorf("missing DONE: %s", out)
	}

	// Status endpoint should record passthrough.
	statusReq := httptest.NewRequest(http.MethodGet, "/admin/status", nil)
	statusRR := httptest.NewRecorder()
	s.Handler().ServeHTTP(statusRR, statusReq)
	if !strings.Contains(statusRR.Body.String(), `"decision":"tool_call"`) {
		t.Errorf("status missing passthrough: %s", statusRR.Body.String())
	}
}

// TestFusionEndToEnd verifies the full fusion path through SSE.
func TestFusionEndToEnd(t *testing.T) {
	mp := &mockProvider{
		completeByModel: map[string]*providers.CompletionResponse{
			"mock-primary":  {Content: "speculative draft", FinishReason: "stop"},
			"mock-panel-1":  {Content: "Panel A says approach X.", FinishReason: "stop"},
			"mock-panel-2":  {Content: "Panel B says approach Y.", FinishReason: "stop"},
			"mock-judge":    {Content: `{"consensus":["both agree on the goal"],"contradictions":["X vs Y"],"partial":[],"unique":{},"blind_spots":[]}`, FinishReason: "stop"},
		},
		streamChunks: []providers.Delta{
			{Content: "The final answer combining "},
			{Content: "the panel's analysis."},
		},
		streamFinish: "stop",
	}

	s := newTestServerMulti(t, mp)
	body, _ := json.Marshal(ChatRequest{
		Model:    "fusion-plan",
		Stream:   true,
		Messages: []providers.Message{{Role: "user", Content: "what should I do?"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	s.Handler().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, body=%s", rr.Code, rr.Body.String())
	}
	out := rr.Body.String()
	if !strings.Contains(out, "The final answer combining") {
		t.Errorf("missing streamed final content: %s", out)
	}
	if !strings.Contains(out, `"finish_reason":"stop"`) {
		t.Errorf("missing finish: %s", out)
	}

	// Reasoning blocks present (config has reasoning_blocks_enabled=true).
	if !strings.Contains(out, `"reasoning_content"`) {
		t.Errorf("missing reasoning blocks: %s", out)
	}

	// Status records fusion.
	statusReq := httptest.NewRequest(http.MethodGet, "/admin/status", nil)
	statusRR := httptest.NewRecorder()
	s.Handler().ServeHTTP(statusRR, statusReq)
	if !strings.Contains(statusRR.Body.String(), `"decision":"fusion"`) {
		t.Errorf("status missing fusion: %s", statusRR.Body.String())
	}
}

func TestFusionUsageRollupAndMetrics(t *testing.T) {
	mp := &mockProvider{
		completeByModel: map[string]*providers.CompletionResponse{
			"mock-primary": {Content: "spec", FinishReason: "stop",
				Usage: providers.Usage{PromptTokens: 100, CompletionTokens: 10, CacheReadTokens: 4000}},
			"mock-panel-1": {Content: "A", FinishReason: "stop",
				Usage: providers.Usage{PromptTokens: 50, CompletionTokens: 5, CacheReadTokens: 0, CacheCreationTokens: 1000}},
			"mock-panel-2": {Content: "B", FinishReason: "stop",
				Usage: providers.Usage{PromptTokens: 50, CompletionTokens: 5, CacheReadTokens: 1000}},
			"mock-judge": {Content: `{"consensus":["x"],"contradictions":[],"partial":[]}`, FinishReason: "stop",
				Usage: providers.Usage{PromptTokens: 200, CompletionTokens: 30}},
		},
		streamChunks: []providers.Delta{{Content: "final"}},
		streamFinish: "stop",
		streamUsage:  providers.Usage{PromptTokens: 100, CompletionTokens: 500, CacheReadTokens: 4000},
	}

	s := newTestServerMulti(t, mp)
	body, _ := json.Marshal(ChatRequest{
		Model:    "fusion-plan",
		Stream:   true,
		Messages: []providers.Message{{Role: "user", Content: "go"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	s.Handler().ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d", rr.Code)
	}

	// Snapshot rollup: input = 100+50+50+200+100 = 500; output = 10+5+5+30+500 = 550;
	// cache_read = 4000+0+1000+0+4000 = 9000; cache_creation = 1000.
	statusReq := httptest.NewRequest(http.MethodGet, "/admin/status", nil)
	statusRR := httptest.NewRecorder()
	s.Handler().ServeHTTP(statusRR, statusReq)
	statusBody := statusRR.Body.String()
	for _, want := range []string{
		`"input_tokens":500`,
		`"output_tokens":550`,
		`"cache_read_tokens":9000`,
		`"cache_creation_tokens":1000`,
	} {
		if !strings.Contains(statusBody, want) {
			t.Errorf("status usage missing %q in:\n%s", want, statusBody)
		}
	}

	// Metrics counters reflect the same rollup.
	metricsReq := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	metricsRR := httptest.NewRecorder()
	s.Handler().ServeHTTP(metricsRR, metricsReq)
	metricsBody := metricsRR.Body.String()
	for _, want := range []string{
		`fuse_input_tokens_total{decision="fusion",model="fusion-plan"} 500`,
		`fuse_output_tokens_total{decision="fusion",model="fusion-plan"} 550`,
		`fuse_cache_read_tokens_total{decision="fusion",model="fusion-plan"} 9000`,
		`fuse_cache_creation_tokens_total{decision="fusion",model="fusion-plan"} 1000`,
	} {
		if !strings.Contains(metricsBody, want) {
			t.Errorf("metrics missing %q in:\n%s", want, metricsBody)
		}
	}
}

// TestFusionPanelFailureStillFuses verifies that when every real panel
// member fails, fusion still runs by promoting the speculative response
// to a virtual panel member. The judge runs and a fused answer is
// returned through the streaming primary call.
func TestFusionPanelFailureStillFuses(t *testing.T) {
	mp := &mockProvider{
		completeByModel: map[string]*providers.CompletionResponse{
			"mock-primary": {Content: "speculative answer", FinishReason: "stop"},
			"mock-judge":   {Content: `{"consensus":["only one perspective"],"contradictions":[],"partial":[]}`, FinishReason: "stop"},
		},
		errByModel: map[string]error{
			"mock-panel-1": errors.New("panel boom"),
			"mock-panel-2": errors.New("panel boom"),
		},
		streamChunks: []providers.Delta{{Content: "Final streamed answer."}},
		streamFinish: "stop",
	}

	s := newTestServerMulti(t, mp)
	body, _ := json.Marshal(ChatRequest{
		Model:    "fusion-plan",
		Stream:   true,
		Messages: []providers.Message{{Role: "user", Content: "go"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	s.Handler().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d", rr.Code)
	}
	out := rr.Body.String()
	if !strings.Contains(out, "Final streamed answer.") {
		t.Errorf("expected fused final content: %s", out)
	}

	statusReq := httptest.NewRequest(http.MethodGet, "/admin/status", nil)
	statusRR := httptest.NewRecorder()
	s.Handler().ServeHTTP(statusRR, statusReq)
	if !strings.Contains(statusRR.Body.String(), `"decision":"fusion"`) {
		t.Errorf("status should record fusion (speculative promoted), got: %s", statusRR.Body.String())
	}
}

// --- helpers ---

func newTestServer(t *testing.T, prov providers.Provider) *Server {
	t.Helper()
	cfg := &config.Config{
		Listen:                 "127.0.0.1:0",
		LogDir:                 "",
		LogLevel:               "info",
		ReasoningBlocksEnabled: true,
		SnapshotRetention:      10,
		Providers: map[string]config.Provider{
			"mock": {APIKeyEnv: "MOCK_KEY"},
		},
		Models: map[string]config.Model{
			"fusion-plan": {
				Primary:         config.Endpoint{Provider: "mock", Model: "mock-primary"},
				Panel:           []config.Endpoint{{Provider: "mock", Model: "mock-panel-1"}},
				Judge:           config.Endpoint{Provider: "mock", Model: "mock-judge"},
				PanelTimeoutMs:  25000,
				PanelMinSuccess: 1,
			},
		},
	}
	reg := providers.NewRegistry()
	reg.Register(prov)

	logger := slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelWarn}))
	metrics := obs.NewMetrics()
	statusBuf := obs.NewStatusRing(50)
	snaps, err := obs.NewSnapshotWriter("", 0)
	if err != nil {
		t.Fatal(err)
	}
	return New(cfg, reg, logger, metrics, statusBuf, snaps)
}

func newTestServerMulti(t *testing.T, prov providers.Provider) *Server {
	t.Helper()
	cfg := &config.Config{
		Listen:                 "127.0.0.1:0",
		LogDir:                 "",
		LogLevel:               "info",
		ReasoningBlocksEnabled: true,
		SnapshotRetention:      10,
		Providers: map[string]config.Provider{
			"mock": {APIKeyEnv: "MOCK_KEY"},
		},
		Models: map[string]config.Model{
			"fusion-plan": {
				Primary: config.Endpoint{Provider: "mock", Model: "mock-primary"},
				Panel: []config.Endpoint{
					{Provider: "mock", Model: "mock-panel-1"},
					{Provider: "mock", Model: "mock-panel-2"},
				},
				Judge:           config.Endpoint{Provider: "mock", Model: "mock-judge"},
				PanelTimeoutMs:  25000,
				PanelMinSuccess: 2,
			},
		},
	}
	reg := providers.NewRegistry()
	reg.Register(prov)

	logger := slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelWarn}))
	metrics := obs.NewMetrics()
	statusBuf := obs.NewStatusRing(50)
	snaps, err := obs.NewSnapshotWriter("", 0)
	if err != nil {
		t.Fatal(err)
	}
	return New(cfg, reg, logger, metrics, statusBuf, snaps)
}

type mockProvider struct {
	mu              sync.Mutex
	completeByModel map[string]*providers.CompletionResponse
	errByModel      map[string]error

	streamChunks []providers.Delta
	streamFinish string
	streamUsage  providers.Usage
}

func (m *mockProvider) Name() string { return "mock" }

func (m *mockProvider) Complete(_ context.Context, req providers.CompletionRequest) (*providers.CompletionResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err, ok := m.errByModel[req.Model]; ok && err != nil {
		return nil, err
	}
	if r, ok := m.completeByModel[req.Model]; ok {
		return r, nil
	}
	return &providers.CompletionResponse{Content: "default", FinishReason: "stop"}, nil
}

func (m *mockProvider) Stream(_ context.Context, req providers.CompletionRequest, sink providers.Sink) (providers.Usage, error) {
	m.mu.Lock()
	chunks := m.streamChunks
	finish := m.streamFinish
	usage := m.streamUsage
	m.mu.Unlock()
	for _, c := range chunks {
		if err := sink.Delta(c); err != nil {
			return usage, err
		}
	}
	if finish == "" {
		finish = "stop"
	}
	return usage, sink.Done(finish)
}
