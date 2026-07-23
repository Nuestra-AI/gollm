package providers

import (
	"encoding/json"
	"testing"
)

func TestVerbosityIsModelGatedAndShapedPerAPI(t *testing.T) {
	t.Run("chat completions sends it at the top level", func(t *testing.T) {
		p := NewOpenAIProvider("key", "gpt-5", nil).(*OpenAIProvider)
		p.SetOption("verbosity", "low")
		body, err := p.PrepareRequest("hi", nil)
		if err != nil {
			t.Fatalf("PrepareRequest: %v", err)
		}
		var req map[string]interface{}
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if req["verbosity"] != "low" {
			t.Errorf("verbosity = %v, want %q at the top level: %s", req["verbosity"], "low", body)
		}
	})

	t.Run("non-GPT-5 models never see it", func(t *testing.T) {
		p := NewOpenAIProvider("key", "gpt-4o", nil).(*OpenAIProvider)
		p.SetOption("verbosity", "low")
		body, err := p.PrepareRequest("hi", map[string]interface{}{"verbosity": "high"})
		if err != nil {
			t.Fatalf("PrepareRequest: %v", err)
		}
		var req map[string]interface{}
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if _, present := req["verbosity"]; present {
			t.Errorf("verbosity reached a model that rejects it: %s", body)
		}
	})

	t.Run("invalid values are dropped rather than sent", func(t *testing.T) {
		p := NewOpenAIProvider("key", "gpt-5", nil).(*OpenAIProvider)
		p.SetOption("verbosity", "extremely")
		body, err := p.PrepareRequest("hi", nil)
		if err != nil {
			t.Fatalf("PrepareRequest: %v", err)
		}
		var req map[string]interface{}
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if _, present := req["verbosity"]; present {
			t.Errorf("an out-of-range verbosity would 400 the whole request: %s", body)
		}
	})

	t.Run("responses API nests it under text without losing the schema format", func(t *testing.T) {
		p := NewOpenAIResponsesProvider("key", "gpt-5", nil).(*OpenAIResponsesProvider)
		p.SetOption("verbosity", "high")
		schema := map[string]interface{}{"type": "object", "properties": map[string]interface{}{}}
		body, err := p.PrepareRequestWithSchema("hi", nil, schema)
		if err != nil {
			t.Fatalf("PrepareRequestWithSchema: %v", err)
		}
		var req map[string]interface{}
		if err := json.Unmarshal(body, &req); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if _, flat := req["verbosity"]; flat {
			t.Errorf("verbosity sent flat: the Responses API rejects it there: %s", body)
		}
		text, ok := req["text"].(map[string]interface{})
		if !ok {
			t.Fatalf("no text object: %s", body)
		}
		if text["verbosity"] != "high" {
			t.Errorf("text.verbosity = %v, want %q", text["verbosity"], "high")
		}
		if text["format"] == nil {
			t.Errorf("structured-output format was clobbered by verbosity: %s", body)
		}
	})
}
