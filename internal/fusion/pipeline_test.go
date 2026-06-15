package fusion

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/bpross/fuse-sidecar/internal/config"
	"github.com/bpross/fuse-sidecar/internal/providers"
)

func TestIsFinalization(t *testing.T) {
	cases := []struct {
		name string
		resp *providers.CompletionResponse
		want bool
	}{
		{"nil", nil, false},
		{"with tool calls", &providers.CompletionResponse{ToolCalls: []providers.ToolCall{{ID: "x"}}}, false},
		{"finish_reason tool_calls", &providers.CompletionResponse{FinishReason: "tool_calls"}, false},
		{"plain text stop", &providers.CompletionResponse{Content: "hi", FinishReason: "stop"}, true},
		{"empty finish_reason", &providers.CompletionResponse{Content: "hi"}, true},
		{"length", &providers.CompletionResponse{Content: "hi", FinishReason: "length"}, true},
	}
	for _, tc := range cases {
		if got := IsFinalization(tc.resp); got != tc.want {
			t.Errorf("%s: got %v want %v", tc.name, got, tc.want)
		}
	}
}

func TestPipelinePassthroughOnToolCalls(t *testing.T) {
	prov := &fakeProvider{
		name: "test",
		completeResp: &providers.CompletionResponse{
			Content: "checking the file",
			ToolCalls: []providers.ToolCall{{
				ID:   "call_1",
				Type: "function",
				Function: providers.FunctionCall{
					Name:      "read",
					Arguments: `{"path":"foo.txt"}`,
				},
			}},
			FinishReason: "tool_calls",
		},
	}
	reg := providers.NewRegistry()
	reg.Register(prov)

	p := &Pipeline{Registry: reg, Logger: discardLogger()}
	sink := &captureSink{}
	dec, err := p.Run(context.Background(), config.Model{
		Primary:         config.Endpoint{Provider: "test", Model: "primary"},
		Panel:           []config.Endpoint{{Provider: "test", Model: "p1"}},
		Judge:           config.Endpoint{Provider: "test", Model: "j"},
		PanelTimeoutMs:  5000,
		PanelMinSuccess: 1,
	}, providers.CompletionRequest{
		Messages: []providers.Message{{Role: "user", Content: "read foo.txt"}},
	}, sink)
	if err != nil {
		t.Fatal(err)
	}
	if dec.Kind != DecisionPassthrough {
		t.Errorf("kind = %s, want passthrough", dec.Kind)
	}
	if !sink.done || sink.finish != "tool_calls" {
		t.Errorf("done = %v, finish = %q", sink.done, sink.finish)
	}
	if len(sink.toolCalls) < 2 {
		t.Errorf("expected >= 2 tool call deltas (id + args), got %d", len(sink.toolCalls))
	}
}

func TestPipelineFusionFullPath(t *testing.T) {
	// Primary: speculative returns finalization text; final streaming
	// returns a final answer.
	prov := &fakeProvider{
		name: "test",
		completeResp: &providers.CompletionResponse{
			Content:      "draft",
			FinishReason: "stop",
		},
		// Panel uses Complete too; the judge gets called via Complete.
		// We discriminate by model name set on each call.
	}
	prov.completeByModel = map[string]*providers.CompletionResponse{
		"primary":      {Content: "draft", FinishReason: "stop"},
		"panel-a":      {Content: "Option A: do X.", FinishReason: "stop"},
		"panel-b":      {Content: "Option B: do Y.", FinishReason: "stop"},
		"judge":        {Content: `{"consensus":["agree on Z"],"contradictions":["X vs Y"],"partial":[],"unique":{"A":["x"],"B":["y"]},"blind_spots":[]}`, FinishReason: "stop"},
	}
	prov.streamChunks = []providers.Delta{{Content: "Final answer body."}}
	prov.streamFinish = "stop"

	reg := providers.NewRegistry()
	reg.Register(prov)

	p := &Pipeline{Registry: reg, Logger: discardLogger(), EmitProgress: true}
	sink := &captureSink{}
	dec, err := p.Run(context.Background(), config.Model{
		Primary: config.Endpoint{Provider: "test", Model: "primary"},
		Panel: []config.Endpoint{
			{Provider: "test", Model: "panel-a"},
			{Provider: "test", Model: "panel-b"},
		},
		Judge:           config.Endpoint{Provider: "test", Model: "judge"},
		PanelTimeoutMs:  5000,
		PanelMinSuccess: 2,
	}, providers.CompletionRequest{
		Messages: []providers.Message{
			{Role: "system", Content: "system rules"},
			{Role: "user", Content: "what should I do?"},
		},
	}, sink)
	if err != nil {
		t.Fatal(err)
	}
	if dec.Kind != DecisionFusion {
		t.Errorf("kind = %s, want fusion", dec.Kind)
	}
	if !strings.Contains(strings.Join(sink.contents, ""), "Final answer body.") {
		t.Errorf("missing final answer: %v", sink.contents)
	}
	if dec.JudgeAnalysis == nil {
		t.Errorf("judge analysis was nil")
	}
	if len(dec.Panel) != 2 {
		t.Errorf("panel results len = %d", len(dec.Panel))
	}
	if !sink.done || sink.finish != "stop" {
		t.Errorf("done = %v, finish = %q", sink.done, sink.finish)
	}
	if len(sink.progress) == 0 {
		t.Errorf("expected progress events")
	}
}

func TestPipelineFallbackOnPanelInsufficient(t *testing.T) {
	prov := &fakeProvider{
		name: "test",
		completeByModel: map[string]*providers.CompletionResponse{
			"primary": {Content: "speculative answer", FinishReason: "stop"},
			"panel-a": nil, // returns error
		},
		errByModel: map[string]error{
			"panel-a": errors.New("upstream 500"),
		},
	}
	reg := providers.NewRegistry()
	reg.Register(prov)

	p := &Pipeline{Registry: reg, Logger: discardLogger()}
	sink := &captureSink{}
	dec, err := p.Run(context.Background(), config.Model{
		Primary:         config.Endpoint{Provider: "test", Model: "primary"},
		Panel:           []config.Endpoint{{Provider: "test", Model: "panel-a"}},
		Judge:           config.Endpoint{Provider: "test", Model: "judge"},
		PanelTimeoutMs:  5000,
		PanelMinSuccess: 1,
	}, providers.CompletionRequest{
		Messages: []providers.Message{{Role: "user", Content: "go"}},
	}, sink)
	if err != nil {
		t.Fatal(err)
	}
	if dec.Kind != DecisionFallback {
		t.Errorf("kind = %s, want fallback", dec.Kind)
	}
	if dec.FallbackReason != "panel_insufficient" {
		t.Errorf("reason = %q", dec.FallbackReason)
	}
	if !strings.Contains(strings.Join(sink.contents, ""), "speculative answer") {
		t.Errorf("expected buffered speculative content, got %v", sink.contents)
	}
}

func TestPipelineFallbackOnJudgeFailure(t *testing.T) {
	prov := &fakeProvider{
		name: "test",
		completeByModel: map[string]*providers.CompletionResponse{
			"primary": {Content: "speculative", FinishReason: "stop"},
			"panel-a": {Content: "A", FinishReason: "stop"},
			"panel-b": {Content: "B", FinishReason: "stop"},
		},
		errByModel: map[string]error{
			"judge": errors.New("judge upstream error"),
		},
	}
	reg := providers.NewRegistry()
	reg.Register(prov)

	p := &Pipeline{Registry: reg, Logger: discardLogger()}
	sink := &captureSink{}
	dec, err := p.Run(context.Background(), config.Model{
		Primary: config.Endpoint{Provider: "test", Model: "primary"},
		Panel: []config.Endpoint{
			{Provider: "test", Model: "panel-a"},
			{Provider: "test", Model: "panel-b"},
		},
		Judge:           config.Endpoint{Provider: "test", Model: "judge"},
		PanelTimeoutMs:  5000,
		PanelMinSuccess: 2,
	}, providers.CompletionRequest{
		Messages: []providers.Message{{Role: "user", Content: "go"}},
	}, sink)
	if err != nil {
		t.Fatal(err)
	}
	if dec.Kind != DecisionFallback || dec.FallbackReason != "judge_failed" {
		t.Errorf("kind=%s reason=%s", dec.Kind, dec.FallbackReason)
	}
	if !strings.Contains(strings.Join(sink.contents, ""), "speculative") {
		t.Errorf("expected speculative content, got %v", sink.contents)
	}
}

func TestParseJudgeOutput(t *testing.T) {
	cases := []struct {
		name string
		in   string
		ok   bool
	}{
		{"clean", `{"consensus":["a"],"contradictions":[],"partial":[]}`, true},
		{"fenced", "```json\n{\"consensus\":[\"a\"],\"contradictions\":[],\"partial\":[]}\n```", true},
		{"trailing prose", "Here's the analysis: {\"consensus\":[\"a\"],\"contradictions\":[],\"partial\":[]} hope that helps", true},
		{"garbage", `not json at all`, false},
		{"empty", ``, false},
	}
	for _, tc := range cases {
		got, err := parseJudgeOutput(tc.in)
		if (err == nil) != tc.ok {
			t.Errorf("%s: err = %v, ok = %v", tc.name, err, tc.ok)
			continue
		}
		if tc.ok && got == nil {
			t.Errorf("%s: nil analysis", tc.name)
		}
	}
}

func TestHandoffPreservesOriginal(t *testing.T) {
	original := []providers.Message{
		{Role: "system", Content: "rules"},
		{Role: "user", Content: "task"},
	}
	out := buildHandoffMessages(original, &JudgeAnalysis{
		Consensus: []string{"a"},
	})
	if len(out) != 4 {
		t.Fatalf("len = %d", len(out))
	}
	// First two messages must be byte-identical to original (for caching).
	for i := 0; i < 2; i++ {
		if out[i].Role != original[i].Role || out[i].Content != original[i].Content {
			t.Errorf("position %d mutated: %+v vs %+v", i, out[i], original[i])
		}
	}
	if out[2].Role != "assistant" {
		t.Errorf("handoff [2] role = %q", out[2].Role)
	}
	if !strings.Contains(out[2].Content, "consensus") {
		t.Errorf("handoff [2] missing analysis json: %q", out[2].Content)
	}
	if out[3].Role != "user" {
		t.Errorf("handoff [3] role = %q", out[3].Role)
	}
}

// --- helpers ---

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

type fakeProvider struct {
	name string
	mu   sync.Mutex

	completeResp     *providers.CompletionResponse
	completeByModel  map[string]*providers.CompletionResponse
	errByModel       map[string]error

	streamChunks []providers.Delta
	streamFinish string
}

func (f *fakeProvider) Name() string { return f.name }

func (f *fakeProvider) Complete(_ context.Context, req providers.CompletionRequest) (*providers.CompletionResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err, ok := f.errByModel[req.Model]; ok && err != nil {
		return nil, err
	}
	if r, ok := f.completeByModel[req.Model]; ok {
		return r, nil
	}
	return f.completeResp, nil
}

func (f *fakeProvider) Stream(_ context.Context, req providers.CompletionRequest, sink providers.StreamSink) error {
	f.mu.Lock()
	chunks := f.streamChunks
	finish := f.streamFinish
	f.mu.Unlock()
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

type captureSink struct {
	contents   []string
	progress   []string
	toolCalls  []providers.ToolCallDelta
	done       bool
	finish     string
}

func (c *captureSink) Progress(text string) error {
	c.progress = append(c.progress, text)
	return nil
}

func (c *captureSink) Content(text string) error {
	c.contents = append(c.contents, text)
	return nil
}

func (c *captureSink) ToolCallDelta(d providers.ToolCallDelta) error {
	c.toolCalls = append(c.toolCalls, d)
	return nil
}

func (c *captureSink) Done(reason string) error {
	c.done = true
	c.finish = reason
	return nil
}

// silence unused
var _ = time.Second
