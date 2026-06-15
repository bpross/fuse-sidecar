package obs

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// Snapshot is one record of how a single /v1/chat/completions request was
// handled. Written to disk per request when fusion fires (and for some
// fallback paths) so postmortems are possible without re-running.
type Snapshot struct {
	RequestID        string         `json:"request_id"`
	Timestamp        time.Time      `json:"timestamp"`
	ModelID          string         `json:"model_id"`
	Decision         string         `json:"decision"` // "passthrough", "fusion", "fallback"
	FallbackReason   string         `json:"fallback_reason,omitempty"`
	TotalLatencyMs   int64          `json:"total_latency_ms"`
	Panel            []PanelResult  `json:"panel,omitempty"`
	JudgeLatencyMs   int64          `json:"judge_latency_ms,omitempty"`
	JudgeAnalysis    map[string]any `json:"judge_analysis,omitempty"`
	FinalAnswerBytes int            `json:"final_answer_bytes,omitempty"`
	FinalAnswerHead  string         `json:"final_answer_head,omitempty"`
}

// PanelResult is one panel member's response metadata. The response body
// itself is intentionally not snapshotted by default to keep file sizes
// bounded; callers that need the bodies can attach them via Snapshot.Panel
// if the model config opts in (future work).
type PanelResult struct {
	Provider  string `json:"provider"`
	Model     string `json:"model"`
	LatencyMs int64  `json:"latency_ms"`
	OK        bool   `json:"ok"`
	Error     string `json:"error,omitempty"`
	Attempts  int    `json:"attempts,omitempty"`
}

// SnapshotWriter persists snapshots to a directory, pruning to a retention
// count. Concurrent calls to Write are safe; pruning runs after each write
// and uses lexicographic filename order (timestamp-prefixed) as a proxy for
// age, avoiding a stat-everything pass.
type SnapshotWriter struct {
	dir       string
	retention int
}

// NewSnapshotWriter creates the runs directory under logDir.
func NewSnapshotWriter(logDir string, retention int) (*SnapshotWriter, error) {
	if logDir == "" {
		return &SnapshotWriter{retention: retention}, nil
	}
	dir := filepath.Join(logDir, "runs")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create snapshot dir: %w", err)
	}
	return &SnapshotWriter{dir: dir, retention: retention}, nil
}

// Write persists s to disk. A no-op when retention is zero or the writer
// has no directory.
func (w *SnapshotWriter) Write(s Snapshot) error {
	if w == nil || w.dir == "" || w.retention <= 0 {
		return nil
	}
	name := fmt.Sprintf("%s-%s.json",
		s.Timestamp.UTC().Format("20060102T150405.000000000"),
		s.RequestID)
	path := filepath.Join(w.dir, name)

	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return fmt.Errorf("create snapshot file: %w", err)
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(s); err != nil {
		f.Close()
		return fmt.Errorf("encode snapshot: %w", err)
	}
	if err := f.Close(); err != nil {
		return err
	}
	w.prune()
	return nil
}

// prune deletes the oldest files until the count is <= retention.
// Best-effort: errors are swallowed because they shouldn't fail the request.
func (w *SnapshotWriter) prune() {
	entries, err := os.ReadDir(w.dir)
	if err != nil {
		return
	}
	if len(entries) <= w.retention {
		return
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names) // timestamp prefix gives lexicographic == chronological
	excess := len(names) - w.retention
	for i := 0; i < excess; i++ {
		_ = os.Remove(filepath.Join(w.dir, names[i]))
	}
}
