package llm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/providers"
	"github.com/teilomillet/gollm/utils"
)

// ValidateSchema accepts what the validator can read and rejects a schema that would fail every
// response, wherever in the tree the fault is.
func TestValidateSchema(t *testing.T) {
	tests := []struct {
		name    string
		schema  interface{}
		wantErr bool
	}{
		{"Go map with nullable type list", map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{"q": map[string]interface{}{"type": []string{"string", "null"}}},
			"required":   []string{"q"},
		}, false},
		{"JSON string", `{"type":"object","properties":{"items":{"type":"array","items":{"type":"integer"}}}}`, false},
		{"free-form object and array", `{"type":"object","properties":{"o":{"type":["object","null"]},"a":{"type":"array"}}}`, false},
		{"draft-3 required: true", `{"type":"object","properties":{"a":{"type":"string","required":true}}}`, false},
		{"untyped property accepts any value", `{"type":"object","properties":{"note":{"description":"free text"}}}`, false},
		{"anyOf property is rejected", `{"type":"object","properties":{"q":{"anyOf":[{"type":"string"},{"type":"null"}]}}}`, true},
		{"root $ref is rejected", `{"$ref":"#/$defs/A","$defs":{"A":{"type":"object"}}}`, true},
		{"allOf on a typed field is rejected", `{"type":"object","properties":{"q":{"type":"string","allOf":[{"pattern":"x"}]}}}`, true},
		{"if/then is rejected", `{"type":"object","if":{"required":["a"]},"then":{"required":["b"]}}`, true},
		{"$defs alone is an annotation", `{"type":"object","$defs":{"A":{"type":"string"}},"properties":{"q":{"type":"string"}}}`, false},
		{"pattern Go cannot compile", `{"type":"string","pattern":"^(?=.*[a-z]).{8,}$"}`, true},
		{"enum that is not a list", `{"type":"string","enum":"a"}`, true},
		{"negative maxLength", `{"type":"string","maxLength":-1}`, true},
		{"zero multipleOf", `{"type":"number","multipleOf":0}`, true},
		{"additionalProperties that is a string", `{"type":"object","additionalProperties":"no"}`, true},
		{"additionalProperties schema with a bad type", `{"type":"object","additionalProperties":{"type":"strng"}}`, true},
		{"untyped items", `{"type":"array","items":{}}`, false},
		{"nil schema", nil, true},
		{"JSON null schema", `null`, true},
		{"type that is a number", `{"type":"object","properties":{"q":{"type":5}}}`, true},
		{"misspelled type nested in items", `{"type":"array","items":{"type":"object","properties":{"q":{"type":["strng","null"]}}}}`, true},
		{"properties of the wrong type", `{"type":"object","properties":[]}`, true},
		{"property schema that is not an object", `{"type":"object","properties":{"q":true}}`, true},
		{"items of the wrong type", `{"type":"array","items":"string"}`, true},
		{"required list holds non-string", `{"type":"object","properties":{},"required":[1]}`, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ValidateSchema(tt.schema); (err != nil) != tt.wantErr {
				t.Errorf("ValidateSchema() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// A node with no "type" accepts any value, while its siblings and "required" are still checked.
func TestValidateAgainstSchemaUntypedNodes(t *testing.T) {
	schema := `{"type":"object","properties":{"id":{"type":"integer"},"note":{"description":"free text"}},"required":["id","note"]}`
	tests := []struct {
		name     string
		response string
		wantErr  bool
	}{
		{"untyped field holds anything", `{"id": 1, "note": {"any": ["thing"]}}`, false},
		{"untyped field still required", `{"id": 1}`, true},
		{"typed sibling still checked", `{"id": "one", "note": "x"}`, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ValidateAgainstSchema(tt.response, schema); (err != nil) != tt.wantErr {
				t.Errorf("ValidateAgainstSchema() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// A schema fault is reported even when the response never reaches the faulty part, so a schema
// cannot pass for some responses and fail for others.
func TestValidateAgainstSchemaReportsFaultsTheDataSkips(t *testing.T) {
	schema := `{"type":["object","null"],"properties":{"q":{"type":"strng"}}}`
	if err := ValidateAgainstSchema(`null`, schema); err == nil {
		t.Error("a null response passed a schema with a misspelled property type")
	}
}

// A schema the validator cannot read is rejected before any request: no billed attempt, no retry,
// and an ErrorTypeInvalidInput error the caller can tell apart from a response that failed.
func TestInvalidSchemaSendsNoRequest(t *testing.T) {
	var calls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"1","model":"gpt-4o","choices":[{"message":{"content":"{\"q\":\"x\"}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
	}))
	defer server.Close()

	registry := providers.NewProviderRegistry()
	registry.Register("openai", func(apiKey, model string, extraHeaders map[string]string) providers.Provider {
		return &endpointOverride{
			Provider: providers.NewOpenAIProvider(apiKey, model, extraHeaders),
			endpoint: server.URL,
		}
	})
	cfg := &config.Config{
		Provider: "openai", Model: "gpt-4o", MaxRetries: 3,
		RetryDelay: time.Millisecond, Timeout: 5 * time.Second,
		APIKeys: map[string]string{"openai": "sk-test-key-that-is-long-enough"},
	}
	client, err := NewLLM(cfg, utils.NewLogger(utils.LogLevelWarn), registry)
	if err != nil {
		t.Fatalf("NewLLM returned error: %v", err)
	}

	invalid := `{"type":"object","properties":{"q":{"type":"strng"}}}`
	valid := `{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`

	entryPoints := []struct {
		name string
		call func(schema interface{}) error
	}{
		{"GenerateWithSchema", func(schema interface{}) error {
			_, err := client.GenerateWithSchema(context.Background(), NewPrompt("hi"), schema)
			return err
		}},
		{"GenerateWithSchemaAndUsage", func(schema interface{}) error {
			_, _, err := client.GenerateWithSchemaAndUsage(context.Background(), NewPrompt("hi"), schema)
			return err
		}},
	}
	for _, ep := range entryPoints {
		t.Run(ep.name, func(t *testing.T) {
			atomic.StoreInt32(&calls, 0)
			err := ep.call(invalid)
			var llmErr *LLMError
			if !errors.As(err, &llmErr) || llmErr.Type != ErrorTypeInvalidInput {
				t.Errorf("err = %v, want an ErrorTypeInvalidInput LLMError", err)
			}
			if n := atomic.LoadInt32(&calls); n != 0 {
				t.Errorf("made %d requests for a schema that fails every response, want 0", n)
			}

			if err := ep.call(valid); err != nil {
				t.Errorf("valid schema: %v", err)
			}
			if n := atomic.LoadInt32(&calls); n != 1 {
				t.Errorf("valid schema made %d requests, want 1", n)
			}
		})
	}
}

// Constraint keywords are enforced on the response, so a provider that sees the schema only as
// instructions (Anthropic, DeepSeek) cannot return a value the schema rules out. Each keyword applies
// to the kind of value it constrains, typed or not.
func TestValidateAgainstSchemaConstraints(t *testing.T) {
	field := func(spec string) string { return `{"type":"object","properties":{"v":` + spec + `}}` }
	tests := []struct {
		name     string
		schema   string
		response string
		wantErr  bool
	}{
		{"enum allows a listed value", field(`{"type":"string","enum":["approved"]}`), `{"v":"approved"}`, false},
		{"enum rejects another value", field(`{"type":"string","enum":["approved"]}`), `{"v":"rejected"}`, true},
		{"enum on an untyped field", field(`{"enum":[1,"a",null]}`), `{"v":2}`, true},
		{"nullable enum allows null", field(`{"type":["string","null"],"enum":["a",null]}`), `{"v":null}`, false},
		{"const", field(`{"const":"x"}`), `{"v":"y"}`, true},

		{"minimum", field(`{"type":"number","minimum":0}`), `{"v":-1}`, true},
		{"maximum", field(`{"type":"number","maximum":1}`), `{"v":7}`, true},
		{"within bounds", field(`{"type":"number","minimum":0,"maximum":1}`), `{"v":0.9}`, false},
		{"exclusiveMinimum at the bound", field(`{"type":"integer","exclusiveMinimum":0}`), `{"v":0}`, true},
		{"exclusiveMaximum at the bound", field(`{"type":"integer","exclusiveMaximum":10}`), `{"v":10}`, true},
		{"multipleOf 0.1 allows 0.3", field(`{"type":"number","multipleOf":0.1}`), `{"v":0.3}`, false},
		{"multipleOf 5 rejects 7", field(`{"type":"integer","multipleOf":5}`), `{"v":7}`, true},
		{"integer rejects a fraction", field(`{"type":"integer"}`), `{"v":3.7}`, true},
		{"integer accepts 3.0", field(`{"type":"integer"}`), `{"v":3.0}`, false},
		{"minimum on an untyped field applies to numbers", field(`{"minimum":10}`), `{"v":3}`, true},
		{"minimum on an untyped field ignores strings", field(`{"minimum":10}`), `{"v":"x"}`, false},

		{"maxLength counts code points", field(`{"type":"string","maxLength":2}`), `{"v":"日本"}`, false},
		{"maxLength exceeded", field(`{"type":"string","maxLength":2}`), `{"v":"abc"}`, true},
		{"minLength", field(`{"type":"string","minLength":3}`), `{"v":"ab"}`, true},
		{"pattern matches", field(`{"type":"string","pattern":"^[A-Z]{3}$"}`), `{"v":"ABC"}`, false},
		{"pattern does not match", field(`{"type":"string","pattern":"^[A-Z]{3}$"}`), `{"v":"hello"}`, true},

		{"format date", field(`{"type":"string","format":"date"}`), `{"v":"2026-11-10"}`, false},
		{"format date rejects words", field(`{"type":"string","format":"date"}`), `{"v":"next tuesday"}`, true},
		{"format date-time", field(`{"type":"string","format":"date-time"}`), `{"v":"2026-11-10T12:00:00Z"}`, false},
		{"format date-time without an offset", field(`{"type":"string","format":"date-time"}`), `{"v":"2026-11-10 12:00"}`, true},
		{"format time with or without an offset", field(`{"type":"string","format":"time"}`), `{"v":"14:30:00"}`, false},
		{"format email", field(`{"type":"string","format":"email"}`), `{"v":"a@example.com"}`, false},
		{"format email rejects a display name", field(`{"type":"string","format":"email"}`), `{"v":"Al <a@example.com>"}`, true},
		{"format uuid", field(`{"type":"string","format":"uuid"}`), `{"v":"123e4567-e89b-12d3-a456-426614174000"}`, false},
		{"format uuid rejects other text", field(`{"type":"string","format":"uuid"}`), `{"v":"not-a-uuid"}`, true},
		{"format ipv4", field(`{"type":"string","format":"ipv4"}`), `{"v":"192.168.0.1"}`, false},
		{"format ipv4 rejects ipv6", field(`{"type":"string","format":"ipv4"}`), `{"v":"::1"}`, true},
		{"format ipv6", field(`{"type":"string","format":"ipv6"}`), `{"v":"::1"}`, false},
		{"unknown format is an annotation", field(`{"type":"string","format":"uri"}`), `{"v":"not a uri"}`, false},

		{"minItems", field(`{"type":"array","minItems":2}`), `{"v":["a"]}`, true},
		{"maxItems", field(`{"type":"array","maxItems":1}`), `{"v":["a","b"]}`, true},
		{"uniqueItems", field(`{"type":"array","uniqueItems":true}`), `{"v":[{"a":1},{"a":1}]}`, true},
		{"uniqueItems allows distinct items", field(`{"type":"array","uniqueItems":true}`), `{"v":[1,"1"]}`, false},

		{"additionalProperties false rejects an extra field", `{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":false}`, `{"a":"x","b":1}`, true},
		{"additionalProperties false allows listed fields", `{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":false}`, `{"a":"x"}`, false},
		{"additionalProperties schema checks extra fields", `{"type":"object","additionalProperties":{"type":"string"}}`, `{"a":"x","b":1}`, true},
		{"additionalProperties schema allows matching fields", `{"type":"object","additionalProperties":{"type":"string"}}`, `{"a":"x","b":"y"}`, false},
		{"maxProperties", `{"type":"object","maxProperties":1}`, `{"a":1,"b":2}`, true},
		{"minProperties", `{"type":"object","minProperties":1}`, `{}`, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ValidateAgainstSchema(tt.response, tt.schema); (err != nil) != tt.wantErr {
				t.Errorf("ValidateAgainstSchema() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// Every schema GenerateJSONSchema returns passes ValidateSchema, and its rules are enforced on
// responses: min and max bound a string's length and an array's items, as in the validator package.
func TestGenerateJSONSchemaIsEnforceable(t *testing.T) {
	type inner struct {
		Code string `json:"code" validate:"required,len=3"`
	}
	type sample struct {
		Name  string   `json:"name" validate:"required,min=2,max=5"`
		Age   int      `json:"age" validate:"min=0,max=150"`
		Tags  []string `json:"tags" validate:"min=1,max=2,unique=true"`
		Email string   `json:"email" validate:"email"`
		Note  string   `json:"note" validate:"contains=ok"`
		Mode  string   `json:"mode" validate:"enum=a|b"`
		Inner inner    `json:"inner" validate:"required"`
	}

	for _, v := range []interface{}{sample{}, &sample{}} {
		schema, err := GenerateJSONSchema(v)
		if err != nil {
			t.Fatalf("GenerateJSONSchema(%T): %v", v, err)
		}
		if err := ValidateSchema(schema); err != nil {
			t.Fatalf("generated schema fails ValidateSchema: %v\n%s", err, schema)
		}

		valid := `{"name":"Al","age":30,"tags":["x"],"email":"a@example.com","note":"is ok","mode":"a","inner":{"code":"abc"}}`
		if err := ValidateAgainstSchema(valid, schema); err != nil {
			t.Errorf("valid response rejected: %v", err)
		}
		for name, response := range map[string]string{
			"name too long":        `{"name":"Alexander","inner":{"code":"abc"}}`,
			"too many tags":        `{"name":"Al","tags":["x","y","z"],"inner":{"code":"abc"}}`,
			"duplicate tags":       `{"name":"Al","tags":["x","x"],"inner":{"code":"abc"}}`,
			"age out of range":     `{"name":"Al","age":200,"inner":{"code":"abc"}}`,
			"note missing ok":      `{"name":"Al","note":"nope","inner":{"code":"abc"}}`,
			"mode not in enum":     `{"name":"Al","mode":"c","inner":{"code":"abc"}}`,
			"nested code too long": `{"name":"Al","inner":{"code":"abcd"}}`,
		} {
			if err := ValidateAgainstSchema(response, schema); err == nil {
				t.Errorf("%s: accepted", name)
			}
		}
	}
}

// A rule a schema cannot hold in an enforceable form goes into the field's description, so the
// schema still passes ValidateSchema, the model still sees the rule, and ExtractStructuredData can
// enforce it with Validate. Only a non-struct is an error.
func TestGenerateJSONSchemaDescribesRulesASchemaCannotHold(t *testing.T) {
	type sample struct {
		Excludes  string `json:"excludes" validate:"excludes=x"`
		Password  string `json:"password" validate:"password=strong"`
		Lookahead string `json:"lookahead" validate:"regex=^(?=a)"`
		Two       string `json:"two" validate:"regex=^a,contains=b"`
	}
	raw, err := GenerateJSONSchema(sample{})
	if err != nil {
		t.Fatalf("GenerateJSONSchema: %v", err)
	}
	if err := ValidateSchema(raw); err != nil {
		t.Fatalf("generated schema fails ValidateSchema: %v\n%s", err, raw)
	}

	var schema struct {
		Properties map[string]struct {
			Pattern     string `json:"pattern"`
			Description string `json:"description"`
		} `json:"properties"`
	}
	if err := json.Unmarshal(raw, &schema); err != nil {
		t.Fatal(err)
	}
	for field, want := range map[string]struct{ pattern, description string }{
		"excludes":  {"", `Must not contain "x".`},
		"password":  {`^[A-Za-z\d@$!%*?&]{8,}$`, "Must include an uppercase letter"},
		"lookahead": {"", `Must match the regular expression "^(?=a)".`},
		"two":       {"^a", `Must contain "b".`},
	} {
		got := schema.Properties[field]
		if got.Pattern != want.pattern {
			t.Errorf("%s: pattern = %q, want %q", field, got.Pattern, want.pattern)
		}
		if !strings.Contains(got.Description, want.description) {
			t.Errorf("%s: description = %q, want it to contain %q", field, got.Description, want.description)
		}
	}

	if err := ValidateAgainstSchema(`{"password":"short"}`, raw); err == nil {
		t.Error("password=strong accepted a 5-character password")
	}
	if err := ValidateAgainstSchema(`{"password":"has space 1!"}`, raw); err == nil {
		t.Error("password=strong accepted a character outside its set")
	}

	for _, v := range []interface{}{"not a struct", nil} {
		if _, err := GenerateJSONSchema(v); err == nil {
			t.Errorf("GenerateJSONSchema(%T) succeeded, want an error", v)
		}
	}
}

// The generator reads tags as the validator package does: rules after dive apply to each item, and
// gte/lte bound like min/max. MovieReviewValidated in examples/data_extraction relies on both.
func TestGenerateJSONSchemaFollowsValidatorTags(t *testing.T) {
	type review struct {
		Genres []string `json:"genres" validate:"required,min=1,max=5,dive,min=3,max=20"`
		Age    int      `json:"age" validate:"gte=0,lte=150"`
		Name   string   `json:"name" validate:"gte=2,lte=4"`
	}
	schema, err := GenerateJSONSchema(review{})
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateSchema(schema); err != nil {
		t.Fatalf("generated schema fails ValidateSchema: %v\n%s", err, schema)
	}
	tests := []struct {
		name     string
		response string
		wantErr  bool
	}{
		{"within every bound", `{"genres":["drama","comedy"],"age":30,"name":"Ann"}`, false},
		{"six genres", `{"genres":["aaa","bbb","ccc","ddd","eee","fff"]}`, true},
		{"no genres", `{"genres":[]}`, true},
		{"genre too short", `{"genres":["ab"]}`, true},
		{"genre too long", `{"genres":["abcdefghijklmnopqrstu"]}`, true},
		{"age above lte", `{"genres":["drama"],"age":151}`, true},
		{"name below gte", `{"genres":["drama"],"name":"A"}`, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ValidateAgainstSchema(tt.response, schema); (err != nil) != tt.wantErr {
				t.Errorf("ValidateAgainstSchema() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// omitempty means an empty value skips the field's rules, so they become description text rather
// than keywords an empty response would fail. enum values take the field's type, and only a
// list-level "required" marks the field required.
func TestGenerateJSONSchemaOmitEmptyEnumAndRequired(t *testing.T) {
	type sample struct {
		Nick     string   `json:"nick" validate:"omitempty,min=3,email"`
		Priority int      `json:"priority" validate:"required,enum=1|2|3"`
		Level    int      `json:"level" validate:"enum=low|high"`
		Tags     []string `json:"tags" validate:"omitempty,dive,required"`
		Partner  string   `json:"partner" validate:"required_if=Nick x"`
	}
	raw, err := GenerateJSONSchema(sample{})
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateSchema(raw); err != nil {
		t.Fatalf("generated schema fails ValidateSchema: %v\n%s", err, raw)
	}
	var schema struct {
		Properties map[string]map[string]interface{} `json:"properties"`
		Required   []string                          `json:"required"`
	}
	if err := json.Unmarshal(raw, &schema); err != nil {
		t.Fatal(err)
	}

	nick := schema.Properties["nick"]
	for _, k := range []string{"minLength", "format"} {
		if _, ok := nick[k]; ok {
			t.Errorf("nick has %s under omitempty: %v", k, nick)
		}
	}
	if d, _ := nick["description"].(string); !strings.HasPrefix(d, "If not empty: ") || !strings.Contains(d, "At least 3 characters.") {
		t.Errorf("nick description = %q", d)
	}
	if d, _ := schema.Properties["level"]["description"].(string); d != "Must be one of: low, high." {
		t.Errorf("level description = %q, want the enum described since its values are not integers", d)
	}
	if strings.Join(schema.Required, ",") != "priority" {
		t.Errorf("required = %v, want only priority", schema.Required)
	}

	tests := []struct {
		name     string
		response string
		wantErr  bool
	}{
		{"empty nick passes", `{"nick":"","priority":2}`, false},
		{"integer enum matches a number", `{"priority":2}`, false},
		{"integer enum rejects another number", `{"priority":4}`, true},
		{"tags may be left out", `{"priority":1}`, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ValidateAgainstSchema(tt.response, raw); (err != nil) != tt.wantErr {
				t.Errorf("ValidateAgainstSchema() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// Draft-04's boolean exclusiveMinimum/exclusiveMaximum modify minimum/maximum and are accepted.
func TestValidateAgainstSchemaDraft4ExclusiveBounds(t *testing.T) {
	schema := `{"type":"object","properties":{"n":{"type":"number","minimum":0,"exclusiveMinimum":true,"maximum":10,"exclusiveMaximum":false}}}`
	if err := ValidateSchema(schema); err != nil {
		t.Fatalf("ValidateSchema: %v", err)
	}
	for response, wantErr := range map[string]bool{`{"n":0}`: true, `{"n":0.5}`: false, `{"n":10}`: false, `{"n":10.5}`: true} {
		if err := ValidateAgainstSchema(response, schema); (err != nil) != wantErr {
			t.Errorf("%s: error = %v, wantErr %v", response, err, wantErr)
		}
	}
}
