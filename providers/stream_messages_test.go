package providers

import (
	"encoding/json"
	"testing"

	"github.com/teilomillet/gollm/types"
)

// streamWithMessages is the optional capability llm.LLMImpl.Stream type-asserts
// for to preserve multi-turn structure while streaming.
type streamWithMessages interface {
	PrepareStreamRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error)
}

func structuredFixture() ([]types.MemoryMessage, map[string]interface{}) {
	msgs := []types.MemoryMessage{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi there"},
		{Role: "user", Content: "what did I just say?"},
	}
	opts := map[string]interface{}{
		"system_prompt": "You are a helpful assistant.",
		"max_tokens":    256,
		"strict_tools":  true,
	}
	return msgs, opts
}

// TestPrepareStreamRequestWithMessages_OpenAI verifies the structured streaming
// body keeps every turn (no flattening) and enables streaming. GoogleProvider
// inherits this method via embedding, so this also covers google-openai.
func TestPrepareStreamRequestWithMessages_OpenAI(t *testing.T) {
	p, ok := NewOpenAIProvider("test-key", "gpt-4o-mini", nil).(streamWithMessages)
	if !ok {
		t.Fatal("OpenAIProvider does not implement PrepareStreamRequestWithMessages")
	}

	msgs, opts := structuredFixture()
	body, err := p.PrepareStreamRequestWithMessages(msgs, opts)
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}

	var req struct {
		Stream   bool `json:"stream"`
		Messages []struct {
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
	// system (developer) + 3 turns = 4 messages; flattening would yield 1.
	if len(req.Messages) != 4 {
		t.Fatalf("expected 4 structured messages, got %d: %s", len(req.Messages), body)
	}
	if req.Messages[0].Role != "developer" {
		t.Errorf("expected system as developer message, got role %q", req.Messages[0].Role)
	}
	if req.Messages[2].Role != "assistant" {
		t.Errorf("expected turn 2 to be assistant, got %q", req.Messages[2].Role)
	}
}

// streamRequestPreparer is the base streaming-request capability.
type streamRequestPreparer interface {
	PrepareStreamRequest(prompt string, options map[string]interface{}) ([]byte, error)
}

// TestStreamUsageOption verifies stream_options.include_usage defaults on, can be
// switched off via the "stream_usage" option, and that the control key never
// leaks into the wire body — on both the flattened and structured paths.
func TestStreamUsageOption(t *testing.T) {
	type body struct {
		StreamOptions *struct {
			IncludeUsage bool `json:"include_usage"`
		} `json:"stream_options"`
		StreamUsage *bool `json:"stream_usage"` // must always be nil (never serialized)
	}

	parse := func(raw []byte, err error) body {
		t.Helper()
		if err != nil {
			t.Fatalf("prepare: %v", err)
		}
		var b body
		if err := json.Unmarshal(raw, &b); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if b.StreamUsage != nil {
			t.Fatalf("stream_usage control key leaked into body: %s", raw)
		}
		return b
	}

	flat := NewOpenAIProvider("k", "gpt-4o-mini", nil).(streamRequestPreparer)
	msgs, _ := structuredFixture()
	structured := NewOpenAIProvider("k", "gpt-4o-mini", nil).(streamWithMessages)

	// Default: include_usage present and true.
	b := parse(flat.PrepareStreamRequest("hi", map[string]interface{}{}))
	if b.StreamOptions == nil || !b.StreamOptions.IncludeUsage {
		t.Errorf("flattened default: expected include_usage=true, got %+v", b.StreamOptions)
	}
	b = parse(structured.PrepareStreamRequestWithMessages(msgs, map[string]interface{}{}))
	if b.StreamOptions == nil || !b.StreamOptions.IncludeUsage {
		t.Errorf("structured default: expected include_usage=true, got %+v", b.StreamOptions)
	}

	// Disabled: stream_options omitted entirely.
	b = parse(flat.PrepareStreamRequest("hi", map[string]interface{}{"stream_usage": false}))
	if b.StreamOptions != nil {
		t.Errorf("flattened disabled: expected no stream_options, got %+v", b.StreamOptions)
	}
	b = parse(structured.PrepareStreamRequestWithMessages(msgs, map[string]interface{}{"stream_usage": false}))
	if b.StreamOptions != nil {
		t.Errorf("structured disabled: expected no stream_options, got %+v", b.StreamOptions)
	}
}

// A provider-level stream_usage default (set via SetOption, not passed per call)
// must be honored on the structured streaming path, matching the flattened path.
func TestStreamUsageProviderDefault(t *testing.T) {
	p := NewOpenAIProvider("k", "gpt-4o-mini", nil)
	p.SetOption("stream_usage", false) // provider-level default, no per-call option

	sp := p.(streamWithMessages)
	msgs, _ := structuredFixture()
	body, err := sp.PrepareStreamRequestWithMessages(msgs, map[string]interface{}{})
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	var b struct {
		StreamOptions *struct {
			IncludeUsage bool `json:"include_usage"`
		} `json:"stream_options"`
		StreamUsage *bool `json:"stream_usage"`
	}
	if err := json.Unmarshal(body, &b); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if b.StreamOptions != nil {
		t.Errorf("provider-level stream_usage=false ignored: got stream_options %+v", b.StreamOptions)
	}
	if b.StreamUsage != nil {
		t.Errorf("stream_usage control key leaked into body: %s", body)
	}
}

// TestStreamSystemRole_GoogleVsOpenAI verifies the system prompt is emitted with
// the role each endpoint understands: "developer" for OpenAI, "system" for
// Google's OpenAI-compatible endpoint (which maps it to system_instruction).
func TestStreamSystemRole_GoogleVsOpenAI(t *testing.T) {
	firstRole := func(body []byte) string {
		var req struct {
			Messages []struct {
				Role string `json:"role"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if len(req.Messages) == 0 {
			t.Fatal("no messages in body")
		}
		return req.Messages[0].Role
	}

	msgs, opts := structuredFixture()

	gp := NewGoogleProvider("k", "gemini-2.5-flash", nil).(streamWithMessages)
	gbody, err := gp.PrepareStreamRequestWithMessages(msgs, opts)
	if err != nil {
		t.Fatalf("google prepare: %v", err)
	}
	if role := firstRole(gbody); role != "system" {
		t.Errorf("google: expected system-prompt role %q, got %q", "system", role)
	}

	op := NewOpenAIProvider("k", "gpt-4o-mini", nil).(streamWithMessages)
	obody, err := op.PrepareStreamRequestWithMessages(msgs, opts)
	if err != nil {
		t.Fatalf("openai prepare: %v", err)
	}
	if role := firstRole(obody); role != "developer" {
		t.Errorf("openai: expected system-prompt role %q, got %q", "developer", role)
	}
}

func TestPrepareStreamRequestWithMessages_Anthropic(t *testing.T) {
	p, ok := NewAnthropicProvider("test-key", "claude-haiku-4-5", nil).(streamWithMessages)
	if !ok {
		t.Fatal("AnthropicProvider does not implement PrepareStreamRequestWithMessages")
	}

	msgs, opts := structuredFixture()
	body, err := p.PrepareStreamRequestWithMessages(msgs, opts)
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}

	var req struct {
		Stream   bool          `json:"stream"`
		System   interface{}   `json:"system"` // Anthropic carries system as a top-level field
		Messages []interface{} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if !req.Stream {
		t.Error("stream not enabled")
	}
	if req.System == nil {
		t.Error("system prompt not carried into request")
	}
	// 3 conversation turns preserved (system is separate on Anthropic).
	if len(req.Messages) != 3 {
		t.Fatalf("expected 3 structured messages, got %d: %s", len(req.Messages), body)
	}
}
