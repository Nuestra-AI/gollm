package providers

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/teilomillet/gollm/config"

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

// chatOnlyParams are valid on Chat Completions and absent from the Responses
// schema; sending one there fails the whole request.
var chatOnlyParams = []string{
	"seed", "stop", "n", "frequency_penalty", "presence_penalty",
	"logit_bias", "logprobs", "max_tokens", "max_completion_tokens",
}

// notOpenAIParams are accepted by neither endpoint, but gollm has setters for them
// because other providers do.
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

// TestResponsesRejectsChatOnlyParams: nothing outside the documented /v1/responses
// body may be forwarded, however the request was built.
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

// TestChatRejectsNonOpenAIParams: parameters OpenAI takes on neither endpoint were
// previously forwarded to Chat Completions.
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

// TestChatFilterSkipsNonOpenAICatalogues guards the embedding providers: DeepSeek
// and Google reuse the wire format with their own parameters.
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

// TestResponsesTranslatesResponseFormat: structured output must survive a transport
// change rather than silently downgrading to free text.
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

// TestSetDefaultOptionsForwardsSupportedSampling: each endpoint must receive the
// sampling parameters it accepts and none of the Ollama-family ones, which OpenAI
// takes on neither.
func TestSetDefaultOptionsForwardsSupportedSampling(t *testing.T) {
	minP, tfsZ := 0.05, 1.0
	seed := 7
	cfg := &config.Config{
		Temperature: 0.3, MaxTokens: 256, TopP: 0.8,
		FrequencyPenalty: 0.2, PresencePenalty: 0.4,
		Seed: &seed, MinP: &minP, TfsZ: &tfsZ,
	}

	ollamaOnly := []string{"min_p", "top_k", "repeat_penalty", "repeat_last_n", "mirostat", "tfs_z"}

	t.Run("chat", func(t *testing.T) {
		p := NewOpenAIProvider("sk-t", "gpt-4o", nil)
		p.SetDefaultOptions(cfg)
		body, err := p.PrepareRequest("hi", map[string]interface{}{})
		if err != nil {
			t.Fatalf("PrepareRequest returned error: %v", err)
		}
		request := bodyKeys(t, body)
		for key, want := range map[string]interface{}{
			"temperature": 0.3, "top_p": 0.8,
			"frequency_penalty": 0.2, "presence_penalty": 0.4, "seed": float64(7),
		} {
			got, present := request[key]
			if !present {
				t.Errorf("%q was not forwarded", key)
				continue
			}
			if fmt.Sprint(got) != fmt.Sprint(want) {
				t.Errorf("%q = %v, want %v", key, got, want)
			}
		}
		for _, key := range ollamaOnly {
			if _, present := request[key]; present {
				t.Errorf("forwarded %q, which OpenAI does not accept", key)
			}
		}
	})

	t.Run("responses", func(t *testing.T) {
		p := NewOpenAIResponsesProvider("sk-t", "gpt-4o", nil)
		p.SetDefaultOptions(cfg)
		body, err := p.PrepareRequest("hi", map[string]interface{}{})
		if err != nil {
			t.Fatalf("PrepareRequest returned error: %v", err)
		}
		request := bodyKeys(t, body)
		if request["top_p"] == nil || request["temperature"] == nil {
			t.Errorf("top_p/temperature not forwarded: %s", body)
		}
		// This API has none of these.
		for _, key := range append([]string{"seed", "frequency_penalty", "presence_penalty"}, ollamaOnly...) {
			if _, present := request[key]; present {
				t.Errorf("forwarded %q, which /v1/responses does not accept", key)
			}
		}
	})

	t.Run("reasoning models still drop the sampling family", func(t *testing.T) {
		p := NewOpenAIProvider("sk-t", "gpt-5.6-sol", nil)
		p.SetDefaultOptions(cfg)
		body, err := p.PrepareRequest("hi", map[string]interface{}{})
		if err != nil {
			t.Fatalf("PrepareRequest returned error: %v", err)
		}
		request := bodyKeys(t, body)
		for _, key := range []string{"temperature", "top_p", "frequency_penalty", "presence_penalty"} {
			if _, present := request[key]; present {
				t.Errorf("forwarded %q to a reasoning model, which rejects it", key)
			}
		}
	})
}

// TestSystemHistoryMessageUsesSystemRole: a system-role history message must take
// the same role as the system prompt, so one request cannot carry both.
func TestSystemHistoryMessageUsesSystemRole(t *testing.T) {
	p := NewOpenAIProvider("sk-t", "gpt-4o", nil)
	body, err := p.PrepareRequestWithMessages([]types.MemoryMessage{
		{Role: "system", Content: "history sys"},
		{Role: "user", Content: "hi"},
	}, map[string]interface{}{"system_prompt": "prompt sys"})
	if err != nil {
		t.Fatalf("PrepareRequestWithMessages returned error: %v", err)
	}

	roles := map[string]bool{}
	for _, m := range bodyKeys(t, body)["messages"].([]interface{}) {
		roles[m.(map[string]interface{})["role"].(string)] = true
	}
	if roles["system"] && roles["developer"] {
		t.Errorf("request carries both system and developer roles: %s", body)
	}
	if !roles["developer"] {
		t.Errorf("expected the system role to be normalized to developer: %s", body)
	}
}

// TestZeroTopPIsTreatedAsUnset: NewConfig leaves TopP at Go's zero, which passes
// the gte=0 validation. top_p 0 is degenerate sampling, not "unset", so it must
// never reach a request.
func TestZeroTopPIsTreatedAsUnset(t *testing.T) {
	cfg := &config.Config{Temperature: 0.7, MaxTokens: 100} // TopP zero, as NewConfig leaves it

	t.Run("chat", func(t *testing.T) {
		p := NewOpenAIProvider("sk-t", "gpt-4o", nil)
		p.SetDefaultOptions(cfg)
		body, _ := p.PrepareRequest("hi", map[string]interface{}{})
		if v, present := bodyKeys(t, body)["top_p"]; present {
			t.Errorf("sent top_p=%v; zero must be treated as unset", v)
		}
	})

	t.Run("responses", func(t *testing.T) {
		p := NewOpenAIResponsesProvider("sk-t", "gpt-4o", nil)
		p.SetDefaultOptions(cfg)
		body, _ := p.PrepareRequest("hi", map[string]interface{}{})
		if v, present := bodyKeys(t, body)["top_p"]; present {
			t.Errorf("sent top_p=%v; zero must be treated as unset", v)
		}
	})

	t.Run("a real value still passes through", func(t *testing.T) {
		p := NewOpenAIProvider("sk-t", "gpt-4o", nil)
		p.SetDefaultOptions(&config.Config{Temperature: 0.7, MaxTokens: 100, TopP: 0.8})
		body, _ := p.PrepareRequest("hi", map[string]interface{}{})
		if bodyKeys(t, body)["top_p"] != 0.8 {
			t.Errorf("top_p = %v, want 0.8", bodyKeys(t, body)["top_p"])
		}
	})
}
