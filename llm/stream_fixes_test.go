package llm

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/teilomillet/gollm/providers"
	"github.com/teilomillet/gollm/types"
)

// scriptedRichProvider is a stub Provider that implements richStreamParser with a
// caller-supplied function, for exercising providerStream.Next directly.
type scriptedRichProvider struct {
	stubStreamProvider
	fn func(chunk []byte) (types.StreamChunk, error)
}

func (p *scriptedRichProvider) ParseStreamResponseRich(chunk []byte) (types.StreamChunk, error) {
	return p.fn(chunk)
}

func newStreamFromSSE(provider providers.Provider, sse string) *providerStream {
	return newProviderStream(io.NopCloser(strings.NewReader(sse)), provider,
		&StreamConfig{MaxLineSize: DefaultSSEMaxLineSize},
		func(UsageOutcome, string, string, types.TokenUsage) {})
}

// Fix #1: tool_choice map → strategy string.
func TestToolChoiceValue(t *testing.T) {
	if v, ok := toolChoiceValue(map[string]interface{}{"type": "required"}); !ok || v != "required" {
		t.Errorf(`{"type":"required"} → %q,%v; want "required",true`, v, ok)
	}
	if _, ok := toolChoiceValue(nil); ok {
		t.Error("nil map should yield ok=false")
	}
	if _, ok := toolChoiceValue(map[string]interface{}{}); ok {
		t.Error("empty map should yield ok=false")
	}
	if _, ok := toolChoiceValue(map[string]interface{}{"name": "f"}); ok {
		t.Error("map without a string \"type\" should yield ok=false")
	}
}

// Fix #2: Next surfaces a genuine parser error instead of swallowing it.
func TestNextSurfacesRichError(t *testing.T) {
	prov := &scriptedRichProvider{fn: func([]byte) (types.StreamChunk, error) {
		return types.StreamChunk{}, errors.New("upstream API error")
	}}
	s := newStreamFromSSE(prov, "data: x\n\n")
	if _, err := s.Next(context.Background()); err == nil || !strings.Contains(err.Error(), "upstream API error") {
		t.Fatalf("expected surfaced error, got %v", err)
	}
}

// Fix #2: ErrStreamSkip is still treated as "skip and continue".
func TestNextSkipsErrStreamSkip(t *testing.T) {
	prov := &scriptedRichProvider{fn: func([]byte) (types.StreamChunk, error) {
		return types.StreamChunk{}, types.ErrStreamSkip
	}}
	s := newStreamFromSSE(prov, "data: x\n\n")
	if _, err := s.Next(context.Background()); err != io.EOF {
		t.Fatalf("expected EOF after skipping, got %v", err)
	}
}

// Fix #5: usage is accumulated across chunks (Anthropic-style split), so the
// finish token carries prompt + completion, not just the latest chunk's field.
func TestNextAccumulatesUsage(t *testing.T) {
	prov := &scriptedRichProvider{fn: func(chunk []byte) (types.StreamChunk, error) {
		switch strings.TrimSpace(string(chunk)) {
		case "start":
			return types.StreamChunk{Kind: "usage", Usage: &types.TokenUsage{PromptTokens: 25}}, nil
		case "delta":
			return types.StreamChunk{Kind: "finish", FinishReason: "end_turn", Usage: &types.TokenUsage{CompletionTokens: 42}}, nil
		}
		return types.StreamChunk{}, types.ErrStreamSkip
	}}
	s := newStreamFromSSE(prov, "data: start\n\ndata: delta\n\n")

	if _, err := s.Next(context.Background()); err != nil { // usage token
		t.Fatalf("first Next: %v", err)
	}
	fin, err := s.Next(context.Background()) // finish token
	if err != nil {
		t.Fatalf("second Next: %v", err)
	}
	if fin.Type != "finish" || fin.Metadata == nil {
		t.Fatalf("unexpected finish token: %+v", fin)
	}
	if fin.Metadata["prompt_tokens"] != 25 {
		t.Errorf("prompt_tokens = %v; want 25 (accumulated from earlier chunk)", fin.Metadata["prompt_tokens"])
	}
	if fin.Metadata["completion_tokens"] != 42 {
		t.Errorf("completion_tokens = %v; want 42", fin.Metadata["completion_tokens"])
	}
	if fin.Metadata["total_tokens"] != 67 {
		t.Errorf("total_tokens = %v; want 67 (25+42 synthesized)", fin.Metadata["total_tokens"])
	}
}

// Fix #6: extra parallel tool-call fragments in one chunk are emitted as
// subsequent tokens rather than dropped.
func TestNextDrainsParallelToolCalls(t *testing.T) {
	prov := &scriptedRichProvider{fn: func([]byte) (types.StreamChunk, error) {
		return types.StreamChunk{
			Kind:                "tool_call_delta",
			ToolCallDelta:       &types.ToolCallDelta{Index: 0, ID: "a"},
			ExtraToolCallDeltas: []*types.ToolCallDelta{{Index: 1, ID: "b"}},
		}, nil
	}}
	s := newStreamFromSSE(prov, "data: x\n\n")

	t1, err := s.Next(context.Background())
	if err != nil || t1.ToolCallDelta == nil || t1.ToolCallDelta.ID != "a" {
		t.Fatalf("first tool-call token: %+v err=%v", t1, err)
	}
	t2, err := s.Next(context.Background())
	if err != nil || t2.ToolCallDelta == nil || t2.ToolCallDelta.ID != "b" {
		t.Fatalf("drained second tool-call token: %+v err=%v", t2, err)
	}
	if t1.Index == t2.Index {
		t.Errorf("expected distinct stream indices, both = %d", t1.Index)
	}
}
