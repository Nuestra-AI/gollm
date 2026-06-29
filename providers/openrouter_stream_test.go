package providers

import (
	"encoding/json"
	"io"
	"testing"

	"github.com/teilomillet/gollm/types"
)

func TestOpenRouterRichStream_TextUsageFinishTools(t *testing.T) {
	p := richParser(t, NewOpenRouterProvider("k", "openai/gpt-4o-mini", nil))

	// text delta
	c, err := p.ParseStreamResponseRich([]byte(`{"choices":[{"delta":{"content":"Hello"}}]}`))
	if err != nil || c.Kind != "text" || c.Text != "Hello" {
		t.Fatalf("text: %+v err=%v", c, err)
	}

	// tool-call open (id+name) then args fragment
	c, err = p.ParseStreamResponseRich([]byte(`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"create-image","arguments":""}}]}}]}`))
	if err != nil || c.Kind != "tool_call_delta" || c.ToolCallDelta == nil || c.ToolCallDelta.ID != "call_1" || c.ToolCallDelta.Name != "create-image" {
		t.Fatalf("tool open: %+v err=%v", c, err)
	}
	c, err = p.ParseStreamResponseRich([]byte(`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"prompt\":"}}]}}]}`))
	if err != nil || c.Kind != "tool_call_delta" || c.ToolCallDelta.ArgsFragment != `{"prompt":` {
		t.Fatalf("tool args: %+v err=%v", c, err)
	}

	// finish_reason must NOT end the stream
	c, err = p.ParseStreamResponseRich([]byte(`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`))
	if err != nil || c.Kind != "finish" || c.FinishReason != "tool_calls" {
		t.Fatalf("finish: %+v err=%v", c, err)
	}

	// trailing usage-only chunk
	c, err = p.ParseStreamResponseRich([]byte(`{"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":4,"total_tokens":15}}`))
	if err != nil || c.Kind != "usage" || c.Usage == nil || c.Usage.TotalTokens != 15 {
		t.Fatalf("usage: %+v err=%v", c, err)
	}

	// [DONE] (with the SSE decoder's trailing newline) ends the stream
	if _, err := p.ParseStreamResponseRich([]byte("[DONE]\n")); err != io.EOF {
		t.Fatalf("expected EOF on [DONE], got %v", err)
	}

	// streaming error surfaces
	if _, err := p.ParseStreamResponseRich([]byte(`{"error":{"message":"rate limited"}}`)); err == nil {
		t.Fatal("expected error chunk to surface")
	}
}

// TestPrepareStreamRequestWithMessages_OpenRouter verifies OpenRouter preserves
// multi-turn structure when streaming, prepends the system prompt as a leading
// system message, enables streaming, and requests usage accounting.
func TestPrepareStreamRequestWithMessages_OpenRouter(t *testing.T) {
	p, ok := NewOpenRouterProvider("test-key", "openai/gpt-4o-mini", nil).(streamWithMessages)
	if !ok {
		t.Fatal("OpenRouterProvider does not implement PrepareStreamRequestWithMessages")
	}

	msgs, opts := structuredFixture()
	body, err := p.PrepareStreamRequestWithMessages(msgs, opts)
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}

	var req struct {
		Stream bool `json:"stream"`
		Usage  *struct {
			Include bool `json:"include"`
		} `json:"usage"`
		SystemPrompt *string `json:"system_prompt"` // must not leak as a raw field
		Messages     []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if !req.Stream {
		t.Error("stream not enabled")
	}
	if req.Usage == nil || !req.Usage.Include {
		t.Errorf("expected usage.include=true, got %+v", req.Usage)
	}
	if req.SystemPrompt != nil {
		t.Errorf("system_prompt leaked as a raw field: %s", body)
	}
	// leading system message + 3 turns = 4 messages; flattening would yield 1.
	if len(req.Messages) != 4 {
		t.Fatalf("expected 4 messages, got %d: %s", len(req.Messages), body)
	}
	if req.Messages[0].Role != "system" {
		t.Errorf("expected leading system message, got role %q", req.Messages[0].Role)
	}
	if req.Messages[2].Role != "assistant" {
		t.Errorf("expected turn 2 to be assistant, got %q", req.Messages[2].Role)
	}
}

// Fix #7: a provider-level stream_usage default must not leak as a raw field into
// the OpenRouter request body (it would otherwise re-merge from p.options).
func TestOpenRouterStreamUsageNoLeak(t *testing.T) {
	p := NewOpenRouterProvider("k", "openai/gpt-4o", nil)
	p.SetOption("stream_usage", false) // provider-level default

	sp, ok := p.(streamWithMessages)
	if !ok {
		t.Fatal("OpenRouterProvider does not implement PrepareStreamRequestWithMessages")
	}
	msgs, opts := structuredFixture()
	body, err := sp.PrepareStreamRequestWithMessages(msgs, opts)
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, leaked := raw["stream_usage"]; leaked {
		t.Errorf("stream_usage control key leaked into body: %s", body)
	}
}

// A provider-level system_prompt default (set via SetOption, not passed per call)
// must be injected as a leading system message and not leak as a raw field.
func TestOpenRouterSystemPromptProviderDefault(t *testing.T) {
	p := NewOpenRouterProvider("k", "openai/gpt-4o", nil)
	p.SetOption("system_prompt", "You are a helpful assistant.")

	sp := p.(streamWithMessages)
	msgs := []types.MemoryMessage{{Role: "user", Content: "hi"}}
	body, err := sp.PrepareStreamRequestWithMessages(msgs, map[string]interface{}{})
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	var req struct {
		SystemPrompt *string `json:"system_prompt"`
		Messages     []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if req.SystemPrompt != nil {
		t.Errorf("provider-level system_prompt leaked as a raw field: %s", body)
	}
	if len(req.Messages) != 2 || req.Messages[0].Role != "system" {
		t.Errorf("expected leading system message from provider default, got %+v", req.Messages)
	}
}
