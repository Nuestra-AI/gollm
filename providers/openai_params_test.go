package providers

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/teilomillet/gollm/types"
)

func bodyKeys(t *testing.T, b []byte) map[string]interface{} {
	t.Helper()
	var m map[string]interface{}
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("body is not valid JSON: %v", err)
	}
	return m
}

// chatOnlyParams are valid on /v1/chat/completions and absent from the Responses
// schema. Sending one to /v1/responses fails the whole request.
var chatOnlyParams = []string{
	"seed", "stop", "n", "frequency_penalty", "presence_penalty",
	"logit_bias", "logprobs", "max_tokens", "max_completion_tokens",
}

// notOpenAIParams are accepted by neither endpoint. gollm exposes setters for them
// because other providers take them, so they can reach an OpenAI request.
var notOpenAIParams = []string{"top_k", "min_p", "repeat_penalty", "mirostat"}

func kitchenSinkOptions() map[string]interface{} {
	opts := map[string]interface{}{}
	for _, k := range chatOnlyParams {
		opts[k] = 1
	}
	for _, k := range notOpenAIParams {
		opts[k] = 1
	}
	return opts
}

// TestResponsesRejectsChatOnlyParams pins the allowlist: nothing outside the
// documented /v1/responses body may be forwarded, however the request was built.
func TestResponsesRejectsChatOnlyParams(t *testing.T) {
	schema := map[string]interface{}{"type": "object", "properties": map[string]interface{}{}}
	msgs := []types.MemoryMessage{{Role: "user", Content: "hi"}}

	paths := map[string]func(p Provider, o map[string]interface{}) ([]byte, error){
		"PrepareRequest": func(p Provider, o map[string]interface{}) ([]byte, error) { return p.PrepareRequest("hi", o) },
		"PrepareRequestWithSchema": func(p Provider, o map[string]interface{}) ([]byte, error) {
			return p.PrepareRequestWithSchema("hi", o, schema)
		},
		"PrepareRequestWithMessages": func(p Provider, o map[string]interface{}) ([]byte, error) {
			return p.PrepareRequestWithMessages(msgs, o)
		},
		"PrepareStreamRequest": func(p Provider, o map[string]interface{}) ([]byte, error) { return p.PrepareStreamRequest("hi", o) },
	}

	for name, build := range paths {
		t.Run(name, func(t *testing.T) {
			p := NewOpenAIResponsesProvider("sk-t", "gpt-5.6-sol", nil)
			body, err := build(p, kitchenSinkOptions())
			if err != nil {
				t.Fatalf("%s returned error: %v", name, err)
			}
			request := bodyKeys(t, body)
			for _, key := range append(append([]string{}, chatOnlyParams...), notOpenAIParams...) {
				if _, present := request[key]; present {
					t.Errorf("%s forwarded %q, which /v1/responses does not accept", name, key)
				}
			}
			// The request must still be well formed after filtering.
			if request["model"] == nil || request["input"] == nil {
				t.Errorf("%s: filtering damaged the request: %s", name, body)
			}
		})
	}
}

// TestChatRejectsNonOpenAIParams covers the mirror case: parameters that are not
// OpenAI's on either endpoint were previously forwarded to Chat Completions.
func TestChatRejectsNonOpenAIParams(t *testing.T) {
	p := NewOpenAIProvider("sk-t", "gpt-4o", nil)
	body, err := p.PrepareRequest("hi", kitchenSinkOptions())
	if err != nil {
		t.Fatalf("PrepareRequest returned error: %v", err)
	}
	request := bodyKeys(t, body)

	for _, key := range notOpenAIParams {
		if _, present := request[key]; present {
			t.Errorf("forwarded %q, which /v1/chat/completions does not accept", key)
		}
	}
	// Chat-valid parameters must survive — the allowlist must not overreach.
	for _, key := range []string{"seed", "n", "frequency_penalty", "logit_bias", "logprobs"} {
		if _, present := request[key]; !present {
			t.Errorf("dropped %q, which /v1/chat/completions does accept", key)
		}
	}
}

// TestChatFilterSkipsNonOpenAICatalogues guards the embedding providers. DeepSeek
// and Google reuse OpenAIProvider's wire format while serving their own models and
// parameters, so OpenAI's vocabulary must not be imposed on them.
func TestChatFilterSkipsNonOpenAICatalogues(t *testing.T) {
	for _, model := range []string{"gemini-2.5-pro", "deepseek-chat"} {
		t.Run(model, func(t *testing.T) {
			p := NewOpenAIProvider("sk-t", model, nil)
			body, err := p.PrepareRequest("hi", map[string]interface{}{
				"extra_body": map[string]interface{}{"google": map[string]interface{}{"thinking_config": 1}},
			})
			if err != nil {
				t.Fatalf("PrepareRequest returned error: %v", err)
			}
			if _, present := bodyKeys(t, body)["extra_body"]; !present {
				t.Errorf("extra_body was stripped for %q; the filter must not apply to non-OpenAI catalogues", model)
			}
		})
	}
}

// TestResponsesTranslatesResponseFormat verifies structured output survives a
// transport change: Chat's response_format becomes Responses' text.format rather
// than being dropped, which would silently downgrade the call to free text.
func TestResponsesTranslatesResponseFormat(t *testing.T) {
	t.Run("json_object", func(t *testing.T) {
		p := NewOpenAIResponsesProvider("sk-t", "gpt-5.6-sol", nil)
		body, err := p.PrepareRequest("hi", map[string]interface{}{
			"response_format": map[string]interface{}{"type": "json_object"},
		})
		if err != nil {
			t.Fatalf("PrepareRequest returned error: %v", err)
		}
		request := bodyKeys(t, body)
		if _, leaked := request["response_format"]; leaked {
			t.Error("response_format reached the body; /v1/responses does not accept it")
		}
		text, ok := request["text"].(map[string]interface{})
		if !ok {
			t.Fatalf("no text object in body: %s", body)
		}
		format, ok := text["format"].(map[string]interface{})
		if !ok || format["type"] != "json_object" {
			t.Errorf("text.format = %v, want type json_object", text["format"])
		}
	})

	t.Run("json_schema is flattened", func(t *testing.T) {
		p := NewOpenAIResponsesProvider("sk-t", "gpt-5.6-sol", nil)
		body, err := p.PrepareRequest("hi", map[string]interface{}{
			"response_format": map[string]interface{}{
				"type": "json_schema",
				"json_schema": map[string]interface{}{
					"name":   "out",
					"strict": true,
					"schema": map[string]interface{}{"type": "object"},
				},
			},
		})
		if err != nil {
			t.Fatalf("PrepareRequest returned error: %v", err)
		}
		text := bodyKeys(t, body)["text"].(map[string]interface{})
		format, ok := text["format"].(map[string]interface{})
		if !ok {
			t.Fatalf("no text.format in body: %s", body)
		}
		// Responses hoists the schema fields; it has no nested json_schema object.
		if _, stillNested := format["json_schema"]; stillNested {
			t.Error("json_schema was left nested; /v1/responses expects it flattened")
		}
		if format["name"] != "out" || format["strict"] != true || format["schema"] == nil {
			t.Errorf("flattened format lost fields: %v", format)
		}
	})

	t.Run("an explicit schema wins", func(t *testing.T) {
		p := NewOpenAIResponsesProvider("sk-t", "gpt-5.6-sol", nil)
		body, err := p.PrepareRequestWithSchema("hi", map[string]interface{}{
			"response_format": map[string]interface{}{"type": "json_object"},
		}, map[string]interface{}{"type": "object"})
		if err != nil {
			t.Fatalf("PrepareRequestWithSchema returned error: %v", err)
		}
		text := bodyKeys(t, body)["text"].(map[string]interface{})
		format := text["format"].(map[string]interface{})
		if format["type"] != "json_schema" {
			t.Errorf("text.format.type = %v; the schema requested for this call must win", format["type"])
		}
	})
}

// TestRefusalsSurfaceAsErrors covers both endpoints. A refusal is a completed call
// with no completion; previously it parsed as empty and read as success.
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
