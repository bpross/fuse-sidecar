package obs

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestMetricsCounterAndHistogram(t *testing.T) {
	m := NewMetrics()
	m.Inc("requests_total", "decision", "passthrough")
	m.Inc("requests_total", "decision", "passthrough")
	m.Inc("requests_total", "decision", "fusion")
	m.Observe("panel_latency", 100*time.Millisecond, "provider", "anthropic")
	m.Observe("panel_latency", 200*time.Millisecond, "provider", "anthropic")
	m.Observe("panel_latency", 300*time.Millisecond, "provider", "anthropic")

	out := m.Render()
	if !strings.Contains(out, `requests_total{decision="passthrough"} 2`) {
		t.Errorf("missing passthrough counter:\n%s", out)
	}
	if !strings.Contains(out, `requests_total{decision="fusion"} 1`) {
		t.Errorf("missing fusion counter:\n%s", out)
	}
	if !strings.Contains(out, `panel_latency_count{provider="anthropic"} 3`) {
		t.Errorf("missing histogram count:\n%s", out)
	}
	if !strings.Contains(out, `panel_latency_p50_ms{provider="anthropic"} 200.000`) {
		t.Errorf("p50 wrong:\n%s", out)
	}
}

func TestStatusRing(t *testing.T) {
	r := NewStatusRing(3)
	r.Push(Snapshot{RequestID: "a"})
	r.Push(Snapshot{RequestID: "b"})
	r.Push(Snapshot{RequestID: "c"})
	r.Push(Snapshot{RequestID: "d"}) // evicts "a"

	got := r.Recent(0)
	if len(got) != 3 {
		t.Fatalf("len = %d", len(got))
	}
	wantOrder := []string{"d", "c", "b"}
	for i, w := range wantOrder {
		if got[i].RequestID != w {
			t.Errorf("position %d: got %q want %q", i, got[i].RequestID, w)
		}
	}

	if got := r.Recent(2); len(got) != 2 || got[0].RequestID != "d" {
		t.Errorf("Recent(2) wrong: %+v", got)
	}
}

func TestSnapshotWriterRetention(t *testing.T) {
	dir := t.TempDir()
	w, err := NewSnapshotWriter(dir, 2)
	if err != nil {
		t.Fatal(err)
	}
	base := time.Now()
	for i := 0; i < 5; i++ {
		err := w.Write(Snapshot{
			RequestID: string(rune('a' + i)),
			Timestamp: base.Add(time.Duration(i) * time.Second),
			ModelID:   "fusion-plan",
			Decision:  "fusion",
		})
		if err != nil {
			t.Fatal(err)
		}
	}

	entries, err := os.ReadDir(filepath.Join(dir, "runs"))
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		var names []string
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Errorf("got %d entries, want 2: %v", len(entries), names)
	}
}

func TestSnapshotWriterDisabled(t *testing.T) {
	// retention=0 should be a no-op (no error)
	w, err := NewSnapshotWriter(t.TempDir(), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.Write(Snapshot{RequestID: "x", Timestamp: time.Now()}); err != nil {
		t.Errorf("Write should not error when retention=0: %v", err)
	}
}
