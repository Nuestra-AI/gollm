package providers

import (
	"encoding/json"
	"testing"

	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

// LLMImpl.Stream copies l.Options (which includes strict_tools, set via
// SetOption) into the per-call options map. strict_tools is a per-tool flag, not
// a top-level chat-completions argument: OpenAI 400s on a top-level strict_tools.
// Both stream request builders must keep it out of the request body — the
// flattened path used to leak it, breaking single-turn tool streams.
func TestStreamRequestsDoNotLeakStrictTools(t *testing.T) {
	p := NewOpenAIProvider("k", "gpt-4o", nil).(*OpenAIProvider)

	tools := []utils.Tool{{
		Type: "function",
		Function: utils.Function{
			Name:        "get_weather",
			Description: "gets weather",
			Parameters:  map[string]any{"type": "object", "properties": map[string]any{}},
		},
	}}
	opts := func() map[string]interface{} {
		return map[string]interface{}{
			"strict_tools": true,
			"temperature":  0.0,
			"max_tokens":   1024,
			"tools":        tools,
			"stream":       true,
		}
	}

	assertNoTopLevelStrictTools := func(t *testing.T, label string, body []byte) {
		t.Helper()
		var m map[string]interface{}
		if err := json.Unmarshal(body, &m); err != nil {
			t.Fatalf("%s: unmarshal: %v", label, err)
		}
		if _, ok := m["strict_tools"]; ok {
			t.Errorf("%s: request body leaks top-level strict_tools", label)
		}
		if _, ok := m["tools"]; !ok {
			t.Errorf("%s: tools missing from request body", label)
		}
	}

	// Flattened path: single plain user turn (no history) — persona/knowledge
	// live in the system prompt, so nothing lands in prompt.Messages.
	flat, err := p.PrepareStreamRequest("hello", opts())
	if err != nil {
		t.Fatalf("PrepareStreamRequest: %v", err)
	}
	assertNoTopLevelStrictTools(t, "flattened", flat)

	// Structured path: multi-turn conversation.
	msgs := []types.MemoryMessage{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
		{Role: "user", Content: "weather?"},
	}
	structured, err := p.PrepareStreamRequestWithMessages(msgs, opts())
	if err != nil {
		t.Fatalf("PrepareStreamRequestWithMessages: %v", err)
	}
	assertNoTopLevelStrictTools(t, "structured", structured)
}

// The tool-array assertion must not panic when strict_tools is absent from the
// options (previously an unchecked type assertion on a missing key).
func TestPrepareRequestWithMessagesNoStrictToolsKey(t *testing.T) {
	p := NewOpenAIProvider("k", "gpt-4o", nil).(*OpenAIProvider)
	tools := []utils.Tool{{
		Type:     "function",
		Function: utils.Function{Name: "f", Description: "d", Parameters: map[string]any{"type": "object"}},
	}}
	msgs := []types.MemoryMessage{{Role: "user", Content: "hi"}}

	// No strict_tools key at all — must not panic.
	if _, err := p.PrepareRequestWithMessages(msgs, map[string]interface{}{"tools": tools}); err != nil {
		t.Fatalf("PrepareRequestWithMessages: %v", err)
	}
	if _, err := p.PrepareRequest("hi", map[string]interface{}{"tools": tools}); err != nil {
		t.Fatalf("PrepareRequest: %v", err)
	}
}

// Control/meta keys set as provider-level defaults via SetOption (not per-call)
// must not leak into the request body either. Both stream builders merge
// p.options into the body; the provider (p.options) merge loops previously
// excluded strict_tools/images only on the per-call path, so a SetOption default
// leaked a top-level strict_tools/images and 400d the request.
func TestStreamRequestsDoNotLeakProviderDefaultControlKeys(t *testing.T) {
	p := NewOpenAIProvider("k", "gpt-4o", nil).(*OpenAIProvider)
	p.SetOption("strict_tools", true)
	p.SetOption("images", []types.ContentPart{})

	tools := []utils.Tool{{
		Type: "function",
		Function: utils.Function{
			Name:        "get_weather",
			Description: "gets weather",
			Parameters:  map[string]any{"type": "object", "properties": map[string]any{}},
		},
	}}
	// tools travel per-call (mirroring llm.Stream); strict_tools/images are
	// provider-level defaults, the path this test exercises.
	opts := func() map[string]interface{} {
		return map[string]interface{}{"tools": tools, "stream": true}
	}

	assertNoLeak := func(t *testing.T, label string, body []byte) {
		t.Helper()
		var m map[string]interface{}
		if err := json.Unmarshal(body, &m); err != nil {
			t.Fatalf("%s: unmarshal: %v", label, err)
		}
		for _, k := range []string{"strict_tools", "images"} {
			if _, ok := m[k]; ok {
				t.Errorf("%s: request body leaks top-level %s from provider defaults", label, k)
			}
		}
		if _, ok := m["tools"]; !ok {
			t.Errorf("%s: tools missing from request body", label)
		}
	}

	flat, err := p.PrepareStreamRequest("hello", opts())
	if err != nil {
		t.Fatalf("PrepareStreamRequest: %v", err)
	}
	assertNoLeak(t, "flattened", flat)

	msgs := []types.MemoryMessage{{Role: "user", Content: "weather?"}}
	structured, err := p.PrepareStreamRequestWithMessages(msgs, opts())
	if err != nil {
		t.Fatalf("PrepareStreamRequestWithMessages: %v", err)
	}
	assertNoLeak(t, "structured", structured)
}
