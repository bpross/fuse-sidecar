package fusion

import (
	"context"
	"encoding/json"
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

// TestPipelinePanelFailureUsesSpeculativeAsPanelist verifies that when all
// real panel members fail, the speculative response is promoted to panel
// member so fusion still runs end to end.
func TestPipelinePanelFailureUsesSpeculativeAsPanelist(t *testing.T) {
	prov := &fakeProvider{
		name: "test",
		completeByModel: map[string]*providers.CompletionResponse{
			"primary": {Content: "speculative answer", FinishReason: "stop"},
			"panel-a": nil, // returns error
			"judge":   {Content: `{"consensus":["one perspective"],"contradictions":[],"partial":[]}`, FinishReason: "stop"},
		},
		errByModel: map[string]error{
			"panel-a": errors.New("upstream 500"),
		},
		streamChunks: []providers.Delta{{Content: "Final fused answer."}},
		streamFinish: "stop",
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
	if dec.Kind != DecisionFusion {
		t.Errorf("kind = %s, want fusion (speculative promoted to panel)", dec.Kind)
	}
	if dec.JudgeAnalysis == nil {
		t.Errorf("expected judge analysis even with empty real panel")
	}
	if !strings.Contains(strings.Join(sink.contents, ""), "Final fused answer.") {
		t.Errorf("expected fused final-stream content, got %v", sink.contents)
	}
}

// TestPipelinePanelStripsTools verifies that tools and tool_choice are
// removed from the request passed to panel members. Panel members can't
// execute tools (the agent loop on the client side owns execution), so
// passing tools through causes them to emit tool_call responses we can't
// satisfy. The conversation already contains the tool_call/tool_result
// pairs from the primary's investigation; that's enough context.
func TestPipelinePanelStripsTools(t *testing.T) {
	prov := &fakeProvider{
		name: "test",
		completeByModel: map[string]*providers.CompletionResponse{
			"primary": {Content: "speculative", FinishReason: "stop"},
			"panel-a": {Content: "panel response", FinishReason: "stop"},
			"judge":   {Content: `{"consensus":["a"],"contradictions":[],"partial":[]}`, FinishReason: "stop"},
		},
		streamChunks: []providers.Delta{{Content: "Final."}},
		streamFinish: "stop",
	}
	reg := providers.NewRegistry()
	reg.Register(prov)
	p := &Pipeline{Registry: reg, Logger: discardLogger()}
	sink := &captureSink{}

	// Build a base request with tools attached, like opencode would send.
	baseReq := providers.CompletionRequest{
		Messages: []providers.Message{{Role: "user", Content: "do the thing"}},
		Tools: []providers.Tool{{
			Type: "function",
			Function: providers.ToolDefinition{
				Name:       "read",
				Parameters: json.RawMessage(`{"type":"object"}`),
			},
		}},
		ToolChoice: json.RawMessage(`"auto"`),
	}

	_, err := p.Run(context.Background(), config.Model{
		Primary:         config.Endpoint{Provider: "test", Model: "primary"},
		Panel:           []config.Endpoint{{Provider: "test", Model: "panel-a"}},
		Judge:           config.Endpoint{Provider: "test", Model: "judge"},
		PanelTimeoutMs:  5000,
		PanelMinSuccess: 1,
	}, baseReq, sink)
	if err != nil {
		t.Fatal(err)
	}

	// Primary speculative call DOES get tools (it's the agent's real turn).
	if got := prov.lastReq["primary"]; len(got.Tools) == 0 {
		t.Errorf("primary call should have tools, got none")
	}

	// Panel call must NOT carry tools or tool_choice.
	panelReq := prov.lastReq["panel-a"]
	if len(panelReq.Tools) != 0 {
		t.Errorf("panel call should have no tools, got %d", len(panelReq.Tools))
	}
	if len(panelReq.ToolChoice) != 0 {
		t.Errorf("panel call should have no tool_choice, got %s", string(panelReq.ToolChoice))
	}

	// Judge call must also not carry tools.
	judgeReq := prov.lastReq["judge"]
	if len(judgeReq.Tools) != 0 {
		t.Errorf("judge call should have no tools, got %d", len(judgeReq.Tools))
	}
}

// TestPipelinePanelEmptyContentRetries verifies that a panel member returning
// empty content on the first attempt is retried with a forced max_tokens floor.
func TestPipelinePanelEmptyContentRetries(t *testing.T) {
	prov := &fakeProvider{
		name: "test",
		completeByModel: map[string]*providers.CompletionResponse{
			"primary": {Content: "speculative", FinishReason: "stop"},
			"judge":   {Content: `{"consensus":["a"],"contradictions":[],"partial":[]}`, FinishReason: "stop"},
		},
		completeByModelSequence: map[string][]*providers.CompletionResponse{
			// panel-a returns empty first, then content on retry.
			"panel-a": {
				{Content: "", FinishReason: "length"},
				{Content: "now I have words", FinishReason: "stop"},
			},
		},
		streamChunks: []providers.Delta{{Content: "Final."}},
		streamFinish: "stop",
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
	if dec.Kind != DecisionFusion {
		t.Errorf("kind = %s, want fusion", dec.Kind)
	}
	if len(dec.Panel) != 1 || dec.Panel[0].Attempts != 2 {
		t.Errorf("expected 1 panel result with attempts=2, got %+v", dec.Panel)
	}
	if !dec.Panel[0].OK {
		t.Errorf("expected panel OK after retry, got %+v", dec.Panel[0])
	}
}

// TestPipelinePanelAllFailAndSpeculativeEmpty verifies the true-fallback
// branch: when both panel and speculative produced nothing usable, the
// pipeline falls back cleanly to whatever speculative response existed.
func TestPipelinePanelAllFailAndSpeculativeEmpty(t *testing.T) {
	prov := &fakeProvider{
		name: "test",
		completeByModel: map[string]*providers.CompletionResponse{
			"primary": {Content: "", FinishReason: "stop"}, // empty
			"panel-a": nil,
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
	if dec.Kind != DecisionFallback || dec.FallbackReason != "panel_insufficient" {
		t.Errorf("kind=%s reason=%s, want fallback/panel_insufficient", dec.Kind, dec.FallbackReason)
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
		{"flex string instead of array", `{"consensus":"only one point","contradictions":"none","partial":[]}`, true},
		{"flex null", `{"consensus":null,"contradictions":[],"partial":[]}`, true},
		{"missing fields", `{"consensus":["a"]}`, true},
		{"unique with string values", `{"consensus":[],"contradictions":[],"partial":[],"unique":{"A":"single point"}}`, true},
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

func TestFlexStringsUnmarshal(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{`["a","b"]`, []string{"a", "b"}},
		{`"only one"`, []string{"only one"}},
		{`""`, nil},
		{`"none"`, nil},
		{`"None"`, nil},
		{`"n/a"`, nil},
		{`null`, nil},
		{`[]`, []string{}},
	}
	for _, tc := range cases {
		var fs flexStrings
		if err := fs.UnmarshalJSON([]byte(tc.in)); err != nil {
			t.Errorf("%s: err = %v", tc.in, err)
			continue
		}
		got := []string(fs)
		if len(got) != len(tc.want) {
			t.Errorf("%s: len mismatch %d vs %d (%v)", tc.in, len(got), len(tc.want), got)
			continue
		}
		for i := range got {
			if got[i] != tc.want[i] {
				t.Errorf("%s: position %d: %q vs %q", tc.in, i, got[i], tc.want[i])
			}
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

	completeResp    *providers.CompletionResponse
	completeByModel map[string]*providers.CompletionResponse
	errByModel      map[string]error

	// completeByModelSequence lets a test return a different response on
	// each call to the same model. Each call advances the per-model
	// cursor; when exhausted, falls back to completeByModel/completeResp.
	completeByModelSequence map[string][]*providers.CompletionResponse
	sequenceCursor          map[string]int

	// callCount tracks total Complete calls per model for retry tests.
	callCount map[string]int

	// lastReq captures the most recent CompletionRequest per model, useful
	// for asserting on what was passed (e.g. tools stripped).
	lastReq map[string]providers.CompletionRequest

	streamChunks []providers.Delta
	streamFinish string
}

func (f *fakeProvider) Name() string { return f.name }

func (f *fakeProvider) Complete(_ context.Context, req providers.CompletionRequest) (*providers.CompletionResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.callCount == nil {
		f.callCount = map[string]int{}
	}
	if f.lastReq == nil {
		f.lastReq = map[string]providers.CompletionRequest{}
	}
	f.callCount[req.Model]++
	f.lastReq[req.Model] = req
	if err, ok := f.errByModel[req.Model]; ok && err != nil {
		return nil, err
	}
	if seq, ok := f.completeByModelSequence[req.Model]; ok && len(seq) > 0 {
		if f.sequenceCursor == nil {
			f.sequenceCursor = map[string]int{}
		}
		idx := f.sequenceCursor[req.Model]
		if idx < len(seq) {
			f.sequenceCursor[req.Model] = idx + 1
			return seq[idx], nil
		}
		// fall through if exhausted
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
