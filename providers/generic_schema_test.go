package providers

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/teilomillet/gollm/utils"
)

func newGenericFor(t *testing.T, provider string) *GenericProvider {
	t.Helper()
	cfg, ok := GetDefaultRegistry().GetProviderConfig(provider)
	if !ok {
		t.Fatalf("no registry config for %q", provider)
	}
	return &GenericProvider{
		model:   "m",
		config:  cfg,
		options: map[string]interface{}{},
		logger:  utils.NewLogger(utils.LogLevelWarn),
	}
}

// TestGenericSchemaIsNormalized covers the single-prompt schema path, which embedded
// the schema exactly as handed over. A schema given as a string or []byte — both of
// which normalizeSchema exists to accept — went out as a quoted string, so the API
// received text where it expected a schema object. The messages variant already
// normalized, so the two paths disagreed on the same input.
func TestGenericSchemaIsNormalized(t *testing.T) {
	const schemaJSON = `{"type":"object","properties":{"x":{"type":"string"}}}`

	inputs := map[string]interface{}{
		"string": schemaJSON,
		"bytes":  []byte(schemaJSON),
		"map": map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{"x": map[string]interface{}{"type": "string"}},
		},
	}

	for name, schema := range inputs {
		t.Run(name, func(t *testing.T) {
			p := newGenericFor(t, "aliyun")

			body, err := p.PrepareRequestWithSchema("hi", map[string]interface{}{}, schema)
			if err != nil {
				t.Fatalf("PrepareRequestWithSchema failed: %v", err)
			}

			var decoded map[string]interface{}
			if err := json.Unmarshal(body, &decoded); err != nil {
				t.Fatalf("body is not valid JSON: %v", err)
			}
			functions, ok := decoded["functions"].([]interface{})
			if !ok || len(functions) != 1 {
				t.Fatalf("functions = %v, want one entry", decoded["functions"])
			}
			fn, _ := functions[0].(map[string]interface{})
			params, ok := fn["parameters"].(map[string]interface{})
			if !ok {
				t.Fatalf("parameters = %#v (%T), want an object", fn["parameters"], fn["parameters"])
			}
			if params["type"] != "object" {
				t.Errorf("parameters.type = %v, want object", params["type"])
			}
		})
	}
}

// TestGenericSchemaPathsAgree pins the two paths to the same shape for one input.
func TestGenericSchemaPathsAgree(t *testing.T) {
	const schemaJSON = `{"type":"object","properties":{"x":{"type":"string"}}}`
	p := newGenericFor(t, "aliyun")

	single, err := p.PrepareRequestWithSchema("hi", map[string]interface{}{}, schemaJSON)
	if err != nil {
		t.Fatalf("PrepareRequestWithSchema failed: %v", err)
	}
	messages, err := p.PrepareRequestWithMessagesAndSchema(nil, map[string]interface{}{}, schemaJSON)
	if err != nil {
		t.Fatalf("PrepareRequestWithMessagesAndSchema failed: %v", err)
	}

	got, want := functionParams(t, single), functionParams(t, messages)
	if !jsonEqual(t, got, want) {
		t.Errorf("the two schema paths disagree:\n single-prompt: %v\n messages:      %v", got, want)
	}
}

func functionParams(t *testing.T, body []byte) interface{} {
	t.Helper()
	var decoded map[string]interface{}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("body is not valid JSON: %v", err)
	}
	functions, ok := decoded["functions"].([]interface{})
	if !ok || len(functions) == 0 {
		t.Fatalf("functions = %#v, want a non-empty array\n%s", decoded["functions"], body)
	}
	fn, ok := functions[0].(map[string]interface{})
	if !ok {
		t.Fatalf("functions[0] = %#v, want an object\n%s", functions[0], body)
	}
	return fn["parameters"]
}

func jsonEqual(t *testing.T, a, b interface{}) bool {
	t.Helper()
	x, err := json.Marshal(a)
	if err != nil {
		t.Fatalf("marshalling %v: %v", a, err)
	}
	y, err := json.Marshal(b)
	if err != nil {
		t.Fatalf("marshalling %v: %v", b, err)
	}
	return string(x) == string(y)
}

// TestGenericAnthropicStructuredSchemaIsNormalized covers the same defect on the
// Anthropic-shaped structured path, which marshalled the raw value into the prompt —
// a string schema arrived as an escaped blob rather than readable JSON.
func TestGenericAnthropicStructuredSchemaIsNormalized(t *testing.T) {
	p := newGenericFor(t, "anthropic")

	body, err := p.PrepareRequestWithSchema("hi", map[string]interface{}{},
		`{"type":"object","properties":{"x":{"type":"string"}}}`)
	if err != nil {
		t.Fatalf("PrepareRequestWithSchema failed: %v", err)
	}

	var decoded map[string]interface{}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("body is not valid JSON: %v", err)
	}
	messages, ok := decoded["messages"].([]interface{})
	if !ok || len(messages) == 0 {
		t.Fatalf("messages = %#v, want a non-empty array\n%s", decoded["messages"], body)
	}
	first, ok := messages[0].(map[string]interface{})
	if !ok {
		t.Fatalf("messages[0] = %#v, want an object\n%s", messages[0], body)
	}
	content, ok := first["content"].(string)
	if !ok {
		t.Fatalf("messages[0].content = %#v (%T), want a string\n%s", first["content"], first["content"], body)
	}

	if !strings.Contains(content, `{"properties":`) && !strings.Contains(content, `{"type":"object"`) {
		t.Errorf("prompt should carry readable JSON, got %q", content)
	}
	if strings.Contains(content, `\"type\"`) {
		t.Errorf("prompt carries an escaped schema blob, got %q", content)
	}
}
