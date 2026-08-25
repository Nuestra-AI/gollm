package providers

import (
	"strings"
	"testing"
)

// TestRefusalsSurfaceAsErrors: a refusal is a completed call with no completion,
// and previously parsed as empty and read as success.
func TestRefusalsSurfaceAsErrors(t *testing.T) {
	t.Run("chat", func(t *testing.T) {
		body := []byte(`{"id":"1","model":"gpt-4o","choices":[{"message":
			{"content":"","refusal":"I can't help with that."},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":0}}`)
		p := NewOpenAIProvider("sk-t", "gpt-4o", nil)
		text, _, err := p.ParseResponseWithUsage(body)
		if err == nil {
			t.Fatalf("refusal parsed as success, text=%q", text)
		}
		if !strings.Contains(err.Error(), "I can't help with that.") {
			t.Errorf("error does not carry the refusal reason: %v", err)
		}
	})

	t.Run("responses", func(t *testing.T) {
		body := []byte(`{"id":"1","model":"gpt-5.6-sol","status":"completed","output":
			[{"type":"message","content":[{"type":"refusal","refusal":"I can't help with that."}]}],
			"usage":{"input_tokens":1,"output_tokens":0}}`)
		p := NewOpenAIResponsesProvider("sk-t", "gpt-5.6-sol", nil)
		text, _, err := p.ParseResponseWithUsage(body)
		if err == nil {
			t.Fatalf("refusal parsed as success, text=%q", text)
		}
		if !strings.Contains(err.Error(), "I can't help with that.") {
			t.Errorf("error does not carry the refusal reason: %v", err)
		}
	})

	t.Run("responses: real output is not suppressed", func(t *testing.T) {
		body := []byte(`{"id":"1","model":"gpt-5.6-sol","status":"completed","output":
			[{"type":"message","content":[{"type":"output_text","text":"here you go"},
			{"type":"refusal","refusal":"partially declined"}]}],
			"usage":{"input_tokens":1,"output_tokens":1}}`)
		p := NewOpenAIResponsesProvider("sk-t", "gpt-5.6-sol", nil)
		text, _, err := p.ParseResponseWithUsage(body)
		if err != nil {
			t.Fatalf("a response carrying real text must not error: %v", err)
		}
		if text != "here you go" {
			t.Errorf("text = %q, want %q", text, "here you go")
		}
	})
}

// TestWebSearchRefusalSurfacesAsError: the web-search path returns before the
// refusal check below it, so a search that ends in a refusal would otherwise
// parse as an empty success — the exact failure this file exists to prevent.
func TestWebSearchRefusalSurfacesAsError(t *testing.T) {
	p := NewOpenAIResponsesProvider("sk-t", "gpt-5.6-sol", nil)

	t.Run("refusal after a search", func(t *testing.T) {
		body := []byte(`{"id":"1","model":"gpt-5.6-sol","status":"completed","output":
			[{"type":"web_search_call","id":"ws_1","status":"completed"},
			{"type":"message","role":"assistant","content":
			[{"type":"refusal","refusal":"I can't help with that."}]}],
			"usage":{"input_tokens":1,"output_tokens":0}}`)
		text, _, err := p.ParseResponseWithUsage(body)
		if err == nil {
			t.Fatalf("refusal parsed as success, text=%q", text)
		}
		if !strings.Contains(err.Error(), "I can't help with that.") {
			t.Errorf("error does not carry the refusal reason: %v", err)
		}
	})

	t.Run("search results are not suppressed", func(t *testing.T) {
		body := []byte(`{"id":"1","model":"gpt-5.6-sol","status":"completed","output":
			[{"type":"web_search_call","id":"ws_1","status":"completed"},
			{"type":"message","role":"assistant","content":
			[{"type":"output_text","text":"here you go"},
			{"type":"refusal","refusal":"partially declined"}]}],
			"usage":{"input_tokens":1,"output_tokens":1}}`)
		text, _, err := p.ParseResponseWithUsage(body)
		if err != nil {
			t.Fatalf("a response carrying real text must not error: %v", err)
		}
		if !strings.Contains(text, "here you go") {
			t.Errorf("text = %q, want it to contain %q", text, "here you go")
		}
	})
}
