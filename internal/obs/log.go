// Package obs provides logging, snapshotting, and metrics for the sidecar.
//
// Logs are JSON via slog. Snapshots are one JSON file per finalization,
// retained by count. Metrics are in-memory counters/histograms exposed
// via Snapshot() for the /metrics endpoint and a ring buffer for the
// /admin/status endpoint.
package obs

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
)

// NewLogger constructs a JSON slog logger writing to logDir/sidecar.log,
// or to stderr when logDir is empty. The caller is responsible for closing
// the returned io.Closer if non-nil.
//
// Log rotation is intentionally out of scope for v1; we rely on whoever runs
// the binary to manage growth (e.g. via logrotate or external tooling).
func NewLogger(logDir, level string) (*slog.Logger, io.Closer, error) {
	lvl, err := parseLevel(level)
	if err != nil {
		return nil, nil, err
	}

	opts := &slog.HandlerOptions{Level: lvl}

	if logDir == "" {
		return slog.New(slog.NewJSONHandler(os.Stderr, opts)), nil, nil
	}

	if err := os.MkdirAll(logDir, 0o755); err != nil {
		return nil, nil, fmt.Errorf("create log dir: %w", err)
	}

	path := filepath.Join(logDir, "sidecar.log")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, nil, fmt.Errorf("open log file: %w", err)
	}

	// Tee to stderr as well so an operator running in the foreground sees
	// activity without tailing the file.
	w := io.MultiWriter(os.Stderr, f)
	return slog.New(slog.NewJSONHandler(w, opts)), f, nil
}

func parseLevel(s string) (slog.Level, error) {
	switch strings.ToLower(s) {
	case "", "info":
		return slog.LevelInfo, nil
	case "debug":
		return slog.LevelDebug, nil
	case "warn", "warning":
		return slog.LevelWarn, nil
	case "error":
		return slog.LevelError, nil
	default:
		return 0, fmt.Errorf("unknown log level %q", s)
	}
}
