package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/bpross/fuse-sidecar/internal/providers"
)

// SSEWriter encodes OpenAI-compatible streaming chunks onto an HTTP response.
//
// Concurrency: the heartbeat goroutine and the main request goroutine both
// write to the underlying ResponseWriter. We serialize all writes with a
// mutex. Callers should treat SSEWriter as a single-owner abstraction per
// request — its methods are safe to call from one goroutine plus the
// internal heartbeat goroutine, not multiple arbitrary writers.
type SSEWriter struct {
	mu      sync.Mutex
	w       http.ResponseWriter
	flusher http.Flusher
	id      string
	model   string
	closed  bool

	hbStop chan struct{}
	hbDone chan struct{}

	// toolCallIndex tracks the next index to assign for tool-call deltas
	// emitted via Delta. Anthropic gives us an Index already; we pass it
	// through. Providers that don't will need to assign one before calling.
}

// NewSSEWriter initializes SSE headers and writes them. It does not start
// the heartbeat goroutine — call StartHeartbeat for that.
func NewSSEWriter(w http.ResponseWriter, id, model string) (*SSEWriter, error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, fmt.Errorf("response writer does not support flushing")
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache, no-store, no-transform")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no") // disable proxy buffering if any
	w.WriteHeader(http.StatusOK)
	flusher.Flush()
	return &SSEWriter{w: w, flusher: flusher, id: id, model: model}, nil
}

// StartHeartbeat begins emitting SSE comments every interval. It stops on
// Close. Useful to keep idle connections alive during the panel/judge gap.
func (s *SSEWriter) StartHeartbeat(interval time.Duration) {
	s.hbStop = make(chan struct{})
	s.hbDone = make(chan struct{})
	go func() {
		defer close(s.hbDone)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-s.hbStop:
				return
			case <-ticker.C:
				s.mu.Lock()
				if s.closed {
					s.mu.Unlock()
					return
				}
				_, _ = fmt.Fprintf(s.w, ": ping %d\n\n", time.Now().Unix())
				s.flusher.Flush()
				s.mu.Unlock()
			}
		}
	}()
}

// SendRoleStart emits the initial assistant-role chunk that OpenAI streams
// begin with. Some clients expect this; harmless if not.
func (s *SSEWriter) SendRoleStart() error {
	chunk := s.makeChunk(StreamDelta{Role: "assistant"}, nil)
	return s.writeChunk(chunk)
}

// SendReasoning emits a reasoning_content delta. Returns nil error if the
// writer is closed (caller can keep trying without panic).
func (s *SSEWriter) SendReasoning(text string) error {
	chunk := s.makeChunk(StreamDelta{ReasoningContent: text}, nil)
	return s.writeChunk(chunk)
}

// SendContent emits a content delta.
func (s *SSEWriter) SendContent(text string) error {
	chunk := s.makeChunk(StreamDelta{Content: text}, nil)
	return s.writeChunk(chunk)
}

// SendToolCallDelta emits one tool-call delta. The provider has already
// assigned an Index.
func (s *SSEWriter) SendToolCallDelta(d providers.ToolCallDelta) error {
	stc := StreamToolCall{
		Index: d.Index,
	}
	if d.ID != "" {
		stc.ID = d.ID
		stc.Type = "function"
	}
	if d.FunctionName != "" {
		stc.Function.Name = d.FunctionName
	}
	if d.ArgumentsPartial != "" {
		stc.Function.Arguments = d.ArgumentsPartial
	}
	chunk := s.makeChunk(StreamDelta{ToolCalls: []StreamToolCall{stc}}, nil)
	return s.writeChunk(chunk)
}

// SendFinish emits the final chunk with finish_reason set, then the SSE
// terminator [DONE], then stops the heartbeat.
func (s *SSEWriter) SendFinish(reason string) error {
	chunk := s.makeChunk(StreamDelta{}, &reason)
	if err := s.writeChunk(chunk); err != nil {
		return err
	}
	return s.sendRaw("data: [DONE]\n\n")
}

// SendErrorEvent emits a non-standard "error" event on the stream. Useful
// when something fails after we've already opened the stream and can no
// longer send an HTTP error code. Clients may or may not surface it.
func (s *SSEWriter) SendErrorEvent(msg string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	payload, _ := json.Marshal(map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    "fuse_error",
		},
	})
	if _, err := fmt.Fprintf(s.w, "event: error\ndata: %s\n\n", payload); err != nil {
		return err
	}
	s.flusher.Flush()
	return nil
}

// Close stops the heartbeat and marks the writer closed. It does not close
// the underlying response writer (the HTTP framework owns that).
func (s *SSEWriter) Close() {
	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()
	if s.hbStop != nil {
		close(s.hbStop)
		<-s.hbDone
		s.hbStop = nil
	}
}

func (s *SSEWriter) makeChunk(delta StreamDelta, finish *string) StreamChunk {
	return StreamChunk{
		ID:      s.id,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   s.model,
		Choices: []StreamChoice{{
			Index:        0,
			Delta:        delta,
			FinishReason: finish,
		}},
	}
}

func (s *SSEWriter) writeChunk(c StreamChunk) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	payload, err := json.Marshal(c)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(s.w, "data: %s\n\n", payload); err != nil {
		return err
	}
	s.flusher.Flush()
	return nil
}

func (s *SSEWriter) sendRaw(raw string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	if _, err := fmt.Fprint(s.w, raw); err != nil {
		return err
	}
	s.flusher.Flush()
	return nil
}

// StreamSink adapts SSEWriter to providers.StreamSink.
type StreamSink struct {
	SSE              *SSEWriter
	EmitReasoning    bool
}

// Delta implements providers.StreamSink.
func (s *StreamSink) Delta(d providers.Delta) error {
	if d.Content != "" {
		if err := s.SSE.SendContent(d.Content); err != nil {
			return err
		}
	}
	if d.ReasoningContent != "" && s.EmitReasoning {
		if err := s.SSE.SendReasoning(d.ReasoningContent); err != nil {
			return err
		}
	}
	if d.ToolCallDelta != nil {
		if err := s.SSE.SendToolCallDelta(*d.ToolCallDelta); err != nil {
			return err
		}
	}
	return nil
}

// Done implements providers.StreamSink.
func (s *StreamSink) Done(reason string) error {
	return s.SSE.SendFinish(reason)
}
