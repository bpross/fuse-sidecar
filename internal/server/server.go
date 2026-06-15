package server

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/bpross/fuse-sidecar/internal/config"
	"github.com/bpross/fuse-sidecar/internal/fusion"
	"github.com/bpross/fuse-sidecar/internal/obs"
	"github.com/bpross/fuse-sidecar/internal/providers"
)

// Version is overridden at build time via -ldflags.
var Version = "dev"

// liveState holds the swappable bits of the server: config + registry +
// pipeline travel together so that hot reload is atomic from a handler's
// view. The logger, metrics, status buffer, and snapshot writer remain
// stable across reloads.
type liveState struct {
	cfg      *config.Config
	registry *providers.Registry
	pipeline *fusion.Pipeline
}

// Server holds dependencies shared across HTTP handlers.
type Server struct {
	state     atomic.Pointer[liveState]
	logger    *slog.Logger
	metrics   *obs.Metrics
	statusBuf *obs.StatusRing
	snapshots *obs.SnapshotWriter
}

// New constructs a Server.
func New(
	cfg *config.Config,
	registry *providers.Registry,
	logger *slog.Logger,
	metrics *obs.Metrics,
	statusBuf *obs.StatusRing,
	snapshots *obs.SnapshotWriter,
) *Server {
	s := &Server{
		logger:    logger,
		metrics:   metrics,
		statusBuf: statusBuf,
		snapshots: snapshots,
	}
	s.state.Store(&liveState{
		cfg:      cfg,
		registry: registry,
		pipeline: &fusion.Pipeline{
			Registry:     registry,
			Logger:       logger,
			EmitProgress: cfg.ReasoningBlocksEnabled,
		},
	})
	return s
}

// Reload atomically swaps config and registry. In-flight requests continue
// against the old state until they complete; new requests see the new state.
func (s *Server) Reload(cfg *config.Config, registry *providers.Registry) {
	s.state.Store(&liveState{
		cfg:      cfg,
		registry: registry,
		pipeline: &fusion.Pipeline{
			Registry:     registry,
			Logger:       s.logger,
			EmitProgress: cfg.ReasoningBlocksEnabled,
		},
	})
	s.logger.Info("config reloaded", "models", modelIDs(cfg))
}

func modelIDs(c *config.Config) []string {
	ids := make([]string, 0, len(c.Models))
	for id := range c.Models {
		ids = append(ids, id)
	}
	return ids
}

func (s *Server) currentState() *liveState {
	return s.state.Load()
}

// Handler returns the configured http.Handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("GET /admin/status", s.handleStatus)
	mux.HandleFunc("GET /metrics", s.handleMetrics)
	return mux
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"ok":      true,
		"version": Version,
	})
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	st := s.currentState()
	now := time.Now().Unix()
	resp := ModelListResponse{Object: "list"}
	for id := range st.cfg.Models {
		resp.Data = append(resp.Data, ModelInfo{
			ID:      id,
			Object:  "model",
			Created: now,
			OwnedBy: "fuse-sidecar",
		})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleStatus(w http.ResponseWriter, _ *http.Request) {
	recent := s.statusBuf.Recent(50)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"version": Version,
		"recent":  recent,
	})
}

func (s *Server) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	_, _ = fmt.Fprint(w, s.metrics.Render())
}

// handleChatCompletions runs the fusion pipeline for one streaming request.
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	st := s.currentState()
	requestID := newRequestID()
	logger := s.logger.With("request_id", requestID)

	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		logger.Warn("decode request", "error", err)
		writeError(w, http.StatusBadRequest, "invalid_request_error", "could not decode request: "+err.Error())
		return
	}

	model, ok := st.cfg.Models[req.Model]
	if !ok {
		logger.Warn("unknown model", "model", req.Model)
		writeError(w, http.StatusNotFound, "model_not_found", fmt.Sprintf("model %q is not configured", req.Model))
		return
	}

	if !req.Stream {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "stream=true is required")
		return
	}

	s.metrics.Inc("fuse_requests_total", "model", req.Model)
	start := time.Now()

	sse, err := NewSSEWriter(w, requestID, req.Model)
	if err != nil {
		logger.Error("init sse", "error", err)
		writeError(w, http.StatusInternalServerError, "internal_error", "could not init stream")
		return
	}
	defer sse.Close()

	if st.cfg.ReasoningBlocksEnabled {
		_ = sse.SendRoleStart()
	}

	// Heartbeat protects the connection against idle-timeout middleboxes
	// during the panel + judge gap.
	sse.StartHeartbeat(5 * time.Second)

	sink := &fusionSink{sse: sse, emitReasoning: st.cfg.ReasoningBlocksEnabled}
	base := translateRequest(req)

	dec, err := st.pipeline.Run(r.Context(), model, base, sink)
	latency := time.Since(start)

	if err != nil {
		logger.Error("pipeline error", "error", err)
		_ = sse.SendErrorEvent(fmt.Sprintf("fusion error: %v", err))
		_ = sse.sendRaw("data: [DONE]\n\n")
		s.metrics.Inc("fuse_fallback_total", "reason", "pipeline_error")
		s.recordSnapshot(obs.Snapshot{
			RequestID:      requestID,
			Timestamp:      start,
			ModelID:        req.Model,
			Decision:       "fallback",
			FallbackReason: "pipeline_error",
			TotalLatencyMs: latency.Milliseconds(),
		})
		return
	}

	snap := obs.Snapshot{
		RequestID:        requestID,
		Timestamp:        start,
		ModelID:          req.Model,
		Decision:         string(dec.Kind),
		FallbackReason:   dec.FallbackReason,
		TotalLatencyMs:   latency.Milliseconds(),
		JudgeLatencyMs:   dec.JudgeLatency.Milliseconds(),
		JudgeAnalysis:    dec.JudgeAnalysis,
		Panel:            panelResultsToObs(dec.Panel),
		FinalAnswerBytes: dec.FinalAnswerBytes,
		FinalAnswerHead:  dec.FinalAnswerHead,
	}
	s.recordSnapshot(snap)

	s.metrics.Observe("fuse_total_latency", latency, "model", req.Model, "decision", string(dec.Kind))
	switch dec.Kind {
	case fusion.DecisionPassthrough:
		s.metrics.Inc("fuse_passthrough_total", "model", req.Model)
	case fusion.DecisionFusion:
		s.metrics.Inc("fuse_fusion_total", "model", req.Model)
	case fusion.DecisionFallback:
		s.metrics.Inc("fuse_fallback_total", "reason", dec.FallbackReason)
	}

	logger.Info("request done",
		"model", req.Model,
		"decision", string(dec.Kind),
		"fallback_reason", dec.FallbackReason,
		"latency_ms", latency.Milliseconds(),
	)
}

func (s *Server) recordSnapshot(snap obs.Snapshot) {
	s.statusBuf.Push(snap)
	if err := s.snapshots.Write(snap); err != nil {
		s.logger.Warn("snapshot write", "error", err, "request_id", snap.RequestID)
	}
}

func panelResultsToObs(in []fusion.PanelResult) []obs.PanelResult {
	out := make([]obs.PanelResult, 0, len(in))
	for _, r := range in {
		out = append(out, obs.PanelResult{
			Provider:  r.Provider,
			Model:     r.Model,
			LatencyMs: r.LatencyMs,
			OK:        r.OK,
			Error:     r.Error,
			Attempts:  r.Attempts,
		})
	}
	return out
}

// translateRequest maps an incoming ChatRequest into a provider request.
// The Model field is left empty; the pipeline fills it in per call.
func translateRequest(req ChatRequest) providers.CompletionRequest {
	return providers.CompletionRequest{
		Messages:       req.Messages,
		Tools:          req.Tools,
		ToolChoice:     req.ToolChoice,
		Temperature:    req.Temperature,
		MaxTokens:      req.MaxTokens,
		TopP:           req.TopP,
		Stop:           req.Stop,
		ResponseFormat: req.ResponseFormat,
	}
}

func newRequestID() string {
	var b [8]byte
	_, _ = rand.Read(b[:])
	return "req_" + hex.EncodeToString(b[:])
}
