package providers

import (
	"io"
	"strings"
	"testing"

	"github.com/teilomillet/gollm/types"
)

// richStreamParser mirrors the optional capability llm.providerStream.Next
// type-asserts for. Kept local so the test fails loudly if a provider drops it.
type richStreamParser interface {
	ParseStreamResponseRich(chunk []byte) (types.StreamChunk, error)
}

func richParser(t *testing.T, p interface{}) richStreamParser {
	t.Helper()
	rp, ok := p.(richStreamParser)
	if !ok {
		t.Fatalf("%T does not implement ParseStreamResponseRich", p)
	}
	return rp
}

func TestOpenAIRichStream_TextUsageFinish(t *testing.T) {
	p := richParser(t, NewOpenAIProvider("k", "gpt-4o-mini", nil))

	// text delta
	c, err := p.ParseStreamResponseRich([]byte(`{"choices":[{"delta":{"content":"Hello"}}]}`))
	if err != nil || c.Kind != "text" || c.Text != "Hello" {
		t.Fatalf("text: kind=%q text=%q err=%v", c.Kind, c.Text, err)
	}

	// finish_reason must NOT end the stream (usage chunk follows)
	c, err = p.ParseStreamResponseRich([]byte(`{"choices":[{"delta":{},"finish_reason":"length"}]}`))
	if err != nil {
		t.Fatalf("finish err: %v", err)
	}
	if c.Kind != "finish" || c.FinishReason != "length" {
		t.Fatalf("finish: kind=%q reason=%q", c.Kind, c.FinishReason)
	}

	// trailing usage-only chunk
	c, err = p.ParseStreamResponseRich([]byte(`{"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":7,"total_tokens":19}}`))
	if err != nil {
		t.Fatalf("usage err: %v", err)
	}
	if c.Kind != "usage" || c.Usage == nil || c.Usage.PromptTokens != 12 || c.Usage.CompletionTokens != 7 {
		t.Fatalf("usage chunk wrong: %+v", c.Usage)
	}

	// [DONE]
	if _, err := p.ParseStreamResponseRich([]byte(`[DONE]`)); err != io.EOF {
		t.Fatalf("expected EOF on [DONE], got %v", err)
	}
}

func TestAnthropicRichStream_SplitUsage(t *testing.T) {
	p := richParser(t, NewAnthropicProvider("k", "claude-haiku-4-5", nil))

	// message_start carries input tokens
	c, err := p.ParseStreamResponseRich([]byte(`{"type":"message_start","message":{"usage":{"input_tokens":25,"output_tokens":1}}}`))
	if err != nil || c.Kind != "usage" || c.Usage == nil || c.Usage.PromptTokens != 25 {
		t.Fatalf("message_start: %+v err=%v", c, err)
	}

	// text
	c, err = p.ParseStreamResponseRich([]byte(`{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}`))
	if err != nil || c.Kind != "text" || c.Text != "Hi" {
		t.Fatalf("text delta: %+v err=%v", c, err)
	}

	// message_delta carries stop reason + output tokens
	c, err = p.ParseStreamResponseRich([]byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":42}}`))
	if err != nil || c.Kind != "finish" || c.FinishReason != "end_turn" || c.Usage == nil || c.Usage.CompletionTokens != 42 {
		t.Fatalf("message_delta: %+v err=%v", c, err)
	}

	// message_stop ends the stream
	if _, err := p.ParseStreamResponseRich([]byte(`{"type":"message_stop"}`)); err != io.EOF {
		t.Fatalf("expected EOF on message_stop, got %v", err)
	}
}

func TestResponsesRichStream_TextAndCompleted(t *testing.T) {
	p := richParser(t, NewOpenAIResponsesProvider("k", "gpt-5.4-mini", nil))

	c, err := p.ParseStreamResponseRich([]byte(`{"type":"response.output_text.delta","delta":"Hey"}`))
	if err != nil || c.Kind != "text" || c.Text != "Hey" {
		t.Fatalf("responses text: %+v err=%v", c, err)
	}

	c, err = p.ParseStreamResponseRich([]byte(`{"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":30,"output_tokens":9,"total_tokens":39}}}`))
	if err != nil {
		t.Fatalf("responses completed err: %v", err)
	}
	if c.Kind != "finish" || c.Usage == nil || c.Usage.PromptTokens != 30 || c.Usage.CompletionTokens != 9 || c.FinishReason != "completed" {
		t.Fatalf("responses completed: %+v", c)
	}
}

// completed/incomplete surface as a finish token carrying status + usage;
// incomplete is a truncated-but-valid response, not an error.
func TestResponsesRichStream_TerminalStatuses(t *testing.T) {
	p := richParser(t, NewOpenAIResponsesProvider("k", "gpt-5.4-mini", nil))
	for _, tc := range []struct {
		event, status string
	}{
		{"response.completed", "completed"},
		{"response.incomplete", "incomplete"},
	} {
		raw := []byte(`{"type":"` + tc.event + `","response":{"status":"` + tc.status + `","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}`)
		c, err := p.ParseStreamResponseRich(raw)
		if err != nil {
			t.Fatalf("%s err: %v", tc.event, err)
		}
		if c.Kind != "finish" || c.FinishReason != tc.status {
			t.Errorf("%s: kind=%q reason=%q", tc.event, c.Kind, c.FinishReason)
		}
		if c.Usage == nil || c.Usage.TotalTokens != 8 {
			t.Errorf("%s: usage not carried: %+v", tc.event, c.Usage)
		}
	}
}

// response.failed and top-level error events surface as errors, not a clean
// finish, so callers can't mistake a failed run for success.
func TestResponsesRichStream_FailureSurfacesError(t *testing.T) {
	p := richParser(t, NewOpenAIResponsesProvider("k", "gpt-5.4-mini", nil))

	_, err := p.ParseStreamResponseRich([]byte(`{"type":"response.failed","response":{"status":"failed","error":{"message":"model overloaded"}}}`))
	if err == nil || !strings.Contains(err.Error(), "model overloaded") {
		t.Fatalf("response.failed: expected error with message, got %v", err)
	}

	_, err = p.ParseStreamResponseRich([]byte(`{"type":"error","message":"invalid request"}`))
	if err == nil || !strings.Contains(err.Error(), "invalid request") {
		t.Fatalf("error event: expected surfaced error, got %v", err)
	}
}

// A gateway that co-locates usage on the finish chunk (instead of a separate
// trailing usage chunk) must not have that usage dropped.
func TestOpenAIRichStream_UsageOnFinishChunk(t *testing.T) {
	p := richParser(t, NewOpenAIProvider("k", "gpt-4o-mini", nil))
	c, err := p.ParseStreamResponseRich([]byte(`{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}`))
	if err != nil || c.Kind != "finish" || c.FinishReason != "stop" {
		t.Fatalf("finish: %+v err=%v", c, err)
	}
	if c.Usage == nil || c.Usage.TotalTokens != 9 {
		t.Fatalf("usage on finish chunk dropped: %+v", c.Usage)
	}
}

// GoogleProvider must inherit the OpenAI rich parser via embedding.
func TestGoogleInheritsRichStream(t *testing.T) {
	p := richParser(t, NewGoogleProvider("k", "gemini-2.5-flash", nil))
	c, err := p.ParseStreamResponseRich([]byte(`{"choices":[{"delta":{"content":"x"}}]}`))
	if err != nil || c.Kind != "text" || c.Text != "x" {
		t.Fatalf("google inherited parser: %+v err=%v", c, err)
	}
}

func TestOpenAIRichStream_ToolCallDeltas(t *testing.T) {
	p := richParser(t, NewOpenAIProvider("k", "gpt-4o-mini", nil))

	// Opening fragment carries id + name (args empty).
	c, err := p.ParseStreamResponseRich([]byte(`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"create-image","arguments":""}}]}}]}`))
	if err != nil || c.Kind != "tool_call_delta" || c.ToolCallDelta == nil {
		t.Fatalf("tool open: %+v err=%v", c, err)
	}
	if c.ToolCallDelta.Index != 0 || c.ToolCallDelta.ID != "call_1" || c.ToolCallDelta.Name != "create-image" {
		t.Fatalf("tool open fields: %+v", c.ToolCallDelta)
	}

	// Argument fragment (no id/name).
	c, err = p.ParseStreamResponseRich([]byte(`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"prompt\":"}}]}}]}`))
	if err != nil || c.Kind != "tool_call_delta" || c.ToolCallDelta.ArgsFragment != `{"prompt":` {
		t.Fatalf("tool args fragment: %+v err=%v", c, err)
	}

	// Tool-call finish reason must surface (and not EOF).
	c, err = p.ParseStreamResponseRich([]byte(`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`))
	if err != nil || c.Kind != "finish" || c.FinishReason != "tool_calls" {
		t.Fatalf("tool finish: %+v err=%v", c, err)
	}
}

func TestAnthropicRichStream_ToolUseDeltas(t *testing.T) {
	p := richParser(t, NewAnthropicProvider("k", "claude-haiku-4-5", nil))

	// content_block_start opens a tool_use block with id + name.
	c, err := p.ParseStreamResponseRich([]byte(`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"create-image"}}`))
	if err != nil || c.Kind != "tool_call_delta" || c.ToolCallDelta == nil {
		t.Fatalf("tool_use start: %+v err=%v", c, err)
	}
	if c.ToolCallDelta.Index != 1 || c.ToolCallDelta.ID != "toolu_1" || c.ToolCallDelta.Name != "create-image" {
		t.Fatalf("tool_use start fields: %+v", c.ToolCallDelta)
	}

	// input_json_delta carries partial JSON for that block index.
	c, err = p.ParseStreamResponseRich([]byte(`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"prompt\":\"cat\"}"}}`))
	if err != nil || c.Kind != "tool_call_delta" || c.ToolCallDelta.Index != 1 || c.ToolCallDelta.ArgsFragment != `{"prompt":"cat"}` {
		t.Fatalf("input_json_delta: %+v err=%v", c, err)
	}

	// message_delta stop_reason=tool_use surfaces as finish.
	c, err = p.ParseStreamResponseRich([]byte(`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":12}}`))
	if err != nil || c.Kind != "finish" || c.FinishReason != "tool_use" {
		t.Fatalf("tool_use finish: %+v err=%v", c, err)
	}
}

// Fix #6: multiple tool_calls in a single chunk surface as ToolCallDelta plus
// ExtraToolCallDeltas (none dropped). Same logic is inherited by OpenRouter.
func TestOpenAIRichStream_ParallelToolCallsInOneChunk(t *testing.T) {
	p := richParser(t, NewOpenAIProvider("k", "gpt-4o-mini", nil))
	c, err := p.ParseStreamResponseRich([]byte(`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"f"}},{"index":1,"id":"b","function":{"name":"g"}}]}}]}`))
	if err != nil || c.Kind != "tool_call_delta" || c.ToolCallDelta == nil {
		t.Fatalf("parallel tool calls: %+v err=%v", c, err)
	}
	if c.ToolCallDelta.ID != "a" {
		t.Errorf("primary tool call ID = %q; want a", c.ToolCallDelta.ID)
	}
	if len(c.ExtraToolCallDeltas) != 1 || c.ExtraToolCallDeltas[0].ID != "b" || c.ExtraToolCallDeltas[0].Index != 1 {
		t.Errorf("extra tool calls = %+v; want one with ID b index 1", c.ExtraToolCallDeltas)
	}
}

func TestResponsesRichStream_FunctionCallDeltas(t *testing.T) {
	p := richParser(t, NewOpenAIResponsesProvider("k", "gpt-5.4-mini", nil))

	// output_item.added announces the function call (id + name).
	c, err := p.ParseStreamResponseRich([]byte(`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"create-image"}}`))
	if err != nil || c.Kind != "tool_call_delta" || c.ToolCallDelta == nil {
		t.Fatalf("responses fn open: %+v err=%v", c, err)
	}
	if c.ToolCallDelta.Index != 0 || c.ToolCallDelta.ID != "call_1" || c.ToolCallDelta.Name != "create-image" {
		t.Fatalf("responses fn open fields: %+v", c.ToolCallDelta)
	}

	// arguments stream in via function_call_arguments.delta.
	c, err = p.ParseStreamResponseRich([]byte(`{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"prompt\":\"cat\"}"}`))
	if err != nil || c.Kind != "tool_call_delta" || c.ToolCallDelta.Index != 0 || c.ToolCallDelta.ArgsFragment != `{"prompt":"cat"}` {
		t.Fatalf("responses fn args: %+v err=%v", c, err)
	}
}
