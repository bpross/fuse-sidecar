package server

import (
	"github.com/bpross/fuse-sidecar/internal/providers"
)

// fusionSink adapts an SSEWriter to fusion.Sink. It is a thin shim: Progress
// becomes a reasoning_content delta (when reasoning blocks are enabled),
// Content becomes a content delta, ToolCallDelta passes through unchanged,
// and Done emits the OpenAI finish chunk + [DONE].
type fusionSink struct {
	sse           *SSEWriter
	emitReasoning bool
}

func (f *fusionSink) Progress(text string) error {
	if !f.emitReasoning {
		return nil
	}
	return f.sse.SendReasoning(text)
}

func (f *fusionSink) Content(text string) error {
	return f.sse.SendContent(text)
}

func (f *fusionSink) ToolCallDelta(d providers.ToolCallDelta) error {
	return f.sse.SendToolCallDelta(d)
}

func (f *fusionSink) Done(reason string) error {
	return f.sse.SendFinish(reason)
}
