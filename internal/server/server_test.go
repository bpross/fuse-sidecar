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
	if !strings.Contains(statusRR.Body.String(), `"decision":"passthrough"`) {
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

// TestFusionFallbackOnPanelFailure verifies that when panel fails the
// buffered speculative response is emitted as the answer.
func TestFusionFallbackOnPanelFailure(t *testing.T) {
	mp := &mockProvider{
		completeByModel: map[string]*providers.CompletionResponse{
			"mock-primary": {Content: "speculative answer", FinishReason: "stop"},
		},
		errByModel: map[string]error{
			"mock-panel-1": errors.New("panel boom"),
			"mock-panel-2": errors.New("panel boom"),
		},
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
	if !strings.Contains(out, "speculative answer") {
		t.Errorf("expected speculative content in fallback: %s", out)
	}

	statusReq := httptest.NewRequest(http.MethodGet, "/admin/status", nil)
	statusRR := httptest.NewRecorder()
	s.Handler().ServeHTTP(statusRR, statusReq)
	if !strings.Contains(statusRR.Body.String(), `"fallback_reason":"panel_insufficient"`) {
		t.Errorf("status missing fallback reason: %s", statusRR.Body.String())
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

func (m *mockProvider) Stream(_ context.Context, req providers.CompletionRequest, sink providers.StreamSink) error {
	m.mu.Lock()
	chunks := m.streamChunks
	finish := m.streamFinish
	m.mu.Unlock()
	for _, c := range chunks {
		if err := sink.Delta(c); err != nil {
			return err
		}
	}
	if finish == "" {
		finish = "stop"
	}
	return sink.Done(finish)
}
