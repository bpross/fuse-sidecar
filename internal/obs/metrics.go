package obs

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// Metrics is a tiny in-memory metrics store. It is not a Prometheus client;
// it exposes counters and latency histograms and renders them in a
// Prometheus-compatible text format on demand.
//
// The goal is observability without a dependency on a metrics library.
// If we ever need richer aggregation, swap to prometheus/client_golang.
type Metrics struct {
	mu         sync.Mutex
	counters   map[string]int64       // key includes labels: "name{k=v,k=v}"
	histograms map[string]*histBuffer // ditto
}

type histBuffer struct {
	values []float64 // milliseconds
	max    int       // cap to keep memory bounded
}

// NewMetrics returns a ready-to-use Metrics with sensible defaults.
func NewMetrics() *Metrics {
	return &Metrics{
		counters:   make(map[string]int64),
		histograms: make(map[string]*histBuffer),
	}
}

// Inc bumps a counter by 1. labels is a flat k,v,k,v sequence.
func (m *Metrics) Inc(name string, labels ...string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.counters[seriesKey(name, labels)]++
}

// Observe records a duration sample in milliseconds.
func (m *Metrics) Observe(name string, d time.Duration, labels ...string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	key := seriesKey(name, labels)
	h, ok := m.histograms[key]
	if !ok {
		h = &histBuffer{max: 1024}
		m.histograms[key] = h
	}
	if len(h.values) >= h.max {
		// drop the oldest sample; this is a ring, not a sliding window.
		copy(h.values, h.values[1:])
		h.values = h.values[:h.max-1]
	}
	h.values = append(h.values, float64(d)/float64(time.Millisecond))
}

// Render returns Prometheus text-format output of every series.
//
// For histograms we emit count, sum, and a small set of percentiles
// (p50, p90, p99). This is not strictly Prometheus histogram format
// (we'd need bucket layout decisions), but it's machine-parseable and
// useful in practice for a single-binary dev tool.
func (m *Metrics) Render() string {
	m.mu.Lock()
	defer m.mu.Unlock()

	var b strings.Builder
	// Counters
	keys := make([]string, 0, len(m.counters))
	for k := range m.counters {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Fprintf(&b, "%s %d\n", k, m.counters[k])
	}
	// Histograms: emit derived metrics.
	hkeys := make([]string, 0, len(m.histograms))
	for k := range m.histograms {
		hkeys = append(hkeys, k)
	}
	sort.Strings(hkeys)
	for _, k := range hkeys {
		h := m.histograms[k]
		count := len(h.values)
		if count == 0 {
			continue
		}
		var sum float64
		sorted := make([]float64, count)
		copy(sorted, h.values)
		sort.Float64s(sorted)
		for _, v := range sorted {
			sum += v
		}
		name, labels := splitSeries(k)
		fmt.Fprintf(&b, "%s_count%s %d\n", name, labels, count)
		fmt.Fprintf(&b, "%s_sum_ms%s %.3f\n", name, labels, sum)
		fmt.Fprintf(&b, "%s_p50_ms%s %.3f\n", name, labels, percentile(sorted, 0.50))
		fmt.Fprintf(&b, "%s_p90_ms%s %.3f\n", name, labels, percentile(sorted, 0.90))
		fmt.Fprintf(&b, "%s_p99_ms%s %.3f\n", name, labels, percentile(sorted, 0.99))
	}
	return b.String()
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 1 {
		return sorted[len(sorted)-1]
	}
	idx := int(float64(len(sorted)-1) * p)
	return sorted[idx]
}

func seriesKey(name string, labels []string) string {
	if len(labels) == 0 {
		return name
	}
	if len(labels)%2 != 0 {
		// Programmer error: drop the trailing odd label rather than panic.
		labels = labels[:len(labels)-1]
	}
	pairs := make([]string, 0, len(labels)/2)
	for i := 0; i < len(labels); i += 2 {
		pairs = append(pairs, fmt.Sprintf("%s=%q", labels[i], labels[i+1]))
	}
	sort.Strings(pairs)
	return name + "{" + strings.Join(pairs, ",") + "}"
}

// splitSeries inverts seriesKey for rendering derived histogram metric names.
func splitSeries(key string) (name, labels string) {
	i := strings.IndexByte(key, '{')
	if i < 0 {
		return key, ""
	}
	return key[:i], key[i:]
}

// StatusRing is a fixed-size ring buffer of recent decisions for the
// /admin/status endpoint. Safe for concurrent use.
type StatusRing struct {
	mu      sync.Mutex
	items   []Snapshot
	cap     int
	nextIdx int
	count   int
}

// NewStatusRing returns a ring buffer with the given capacity.
func NewStatusRing(cap int) *StatusRing {
	if cap < 1 {
		cap = 1
	}
	return &StatusRing{items: make([]Snapshot, cap), cap: cap}
}

// Push adds a snapshot, evicting the oldest if full.
func (r *StatusRing) Push(s Snapshot) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.items[r.nextIdx] = s
	r.nextIdx = (r.nextIdx + 1) % r.cap
	if r.count < r.cap {
		r.count++
	}
}

// Recent returns up to n snapshots in newest-first order. n <= 0 means "all".
func (r *StatusRing) Recent(n int) []Snapshot {
	r.mu.Lock()
	defer r.mu.Unlock()
	if n <= 0 || n > r.count {
		n = r.count
	}
	out := make([]Snapshot, 0, n)
	// walk backwards from nextIdx-1
	for i := 0; i < n; i++ {
		idx := (r.nextIdx - 1 - i + r.cap) % r.cap
		out = append(out, r.items[idx])
	}
	return out
}
