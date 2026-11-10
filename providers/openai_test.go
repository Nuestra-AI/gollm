package providers

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestNeedsMaxCompletionTokens verifies that the needsMaxCompletionTokens function
// correctly identifies models that require max_completion_tokens instead of max_tokens
func TestNeedsMaxCompletionTokens(t *testing.T) {
	testCases := []struct {
		modelName      string
		expectedResult bool
		description    string
	}{
		{"gpt-4", false, "Standard GPT-4 model should not use max_completion_tokens"},
		{"gpt-4-turbo", false, "GPT-4 Turbo model should not use max_completion_tokens"},
		{"gpt-3.5-turbo", false, "GPT-3.5 Turbo model should not use max_completion_tokens"},
		{"o1-preview", true, "o1-preview model should use max_completion_tokens"},
		{"o3-mini", true, "o3-mini model should use max_completion_tokens"},
		{"o4-mini", true, "o4-mini model should use max_completion_tokens"},
		{"gpt-4o", true, "GPT-4o model should use max_completion_tokens"},
		{"gpt-4o-mini", true, "GPT-4o mini model should use max_completion_tokens"},
		{"gpt-5", true, "GPT-5 model should use max_completion_tokens"},
		{"gpt-5.4-pro", true, "GPT-5.4 Pro model should use max_completion_tokens"},
		// The o-series is o<digit>. A bare "o" prefix is not the o-series, and treating it as
		// one sends max_completion_tokens to models that only accept max_tokens. This case
		// previously expected true, from a substring match that also caught "omni-moderation"
		// and "opus-3".
		{"o-preview", false, "a bare o- prefix is not an o-series model"},
		{"omni-moderation-latest", false, "omni-moderation is not an o-series model"},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			// Create a provider with the test model
			provider := NewOpenAIProvider("fake-api-key", tc.modelName, nil)

			// Get the OpenAIProvider concrete type from the Provider interface
			openAIProvider, ok := provider.(*OpenAIProvider)
			assert.True(t, ok, "Provider should be of type *OpenAIProvider")

			// Test the needsMaxCompletionTokens function
			result := openAIProvider.needsMaxCompletionTokens()
			assert.Equal(t, tc.expectedResult, result, "needsMaxCompletionTokens returned unexpected result for model %s", tc.modelName)
		})
	}
}

// Strict mode requires additionalProperties: false on every object, including a nullable one
// whose type is a list such as ["object", "null"].
func TestCleanSchemaForOpenAINullableObject(t *testing.T) {
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"fromJSON": map[string]interface{}{
				"type":       []interface{}{"object", "null"},
				"properties": map[string]interface{}{"a": map[string]interface{}{"type": "string"}},
			},
			"fromGo": map[string]interface{}{
				"type":       []string{"null", "object"},
				"properties": map[string]interface{}{"b": map[string]interface{}{"type": "string"}},
			},
			"text": map[string]interface{}{"type": []interface{}{"string", "null"}},
		},
	}

	cleaned := cleanSchemaForOpenAI(schema, openAISchemaKeywords).(map[string]interface{})
	props := cleaned["properties"].(map[string]interface{})

	assert.Equal(t, false, cleaned["additionalProperties"], "root object")
	assert.Equal(t, false, props["fromJSON"].(map[string]interface{})["additionalProperties"], "nullable object from JSON")
	assert.Equal(t, false, props["fromGo"].(map[string]interface{})["additionalProperties"], "nullable object from a Go literal")
	assert.NotContains(t, props["text"].(map[string]interface{}), "additionalProperties", "nullable string")
}

// An object with no "properties" allows any keys, which strict mode cannot express. Closing it with
// additionalProperties: false would force the model to return {}, so it is left open: the cleaner
// drops the unsupported additionalProperties schema and adds no false in its place.
func TestCleanSchemaForOpenAIFreeFormObject(t *testing.T) {
	schema := map[string]interface{}{
		"type":                 "object",
		"additionalProperties": map[string]interface{}{"type": "string"},
	}

	cleaned := cleanSchemaForOpenAI(schema, openAISchemaKeywords).(map[string]interface{})

	assert.NotContains(t, cleaned, "additionalProperties")
}

// Each provider keeps only the keywords its docs say it enforces. OpenAI's own models keep the value
// limits; xAI keeps them except multipleOf, which its docs do not list; fine-tuned OpenAI models
// reject them; Gemini's compat endpoint gets none (see baseSchemaKeywords). Every provider keeps the
// structure, "enum" and "description". None keeps an unsupported keyword such as minLength, a
// pattern, a format outside strictFormats, or a value limit on a type it does not constrain.
func TestCleanSchemaForOpenAIKeywordsPerProvider(t *testing.T) {
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"sentiment": map[string]interface{}{"type": "string", "enum": []string{"positive", "negative", "neutral"}},
			"score":     map[string]interface{}{"type": "number", "minimum": 0, "maximum": 1, "description": "confidence, 0-1"},
			"count":     map[string]interface{}{"type": []string{"integer", "null"}, "maximum": 5},
			"step":      map[string]interface{}{"type": "number", "multipleOf": 5},
			"code":      map[string]interface{}{"type": "string", "pattern": "^[A-Z]{3}$", "minLength": 3},
			"when":      map[string]interface{}{"type": "string", "format": "date"},
			"link":      map[string]interface{}{"type": "string", "format": "uri"},
			"name":      map[string]interface{}{"type": "string", "minimum": 1},
			"tags":      map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "string"}, "maxItems": 3},
			"label":     map[string]interface{}{"type": "string", "maxItems": 2},
		},
		"required": []string{"sentiment", "score", "count", "step", "code", "when", "link", "name", "tags", "label"},
	}
	field := func(keywords map[string]bool, name string) map[string]interface{} {
		cleaned := cleanSchemaForOpenAI(schema, keywords).(map[string]interface{})
		return cleaned["properties"].(map[string]interface{})[name].(map[string]interface{})
	}

	tests := []struct {
		name                       string
		keywords                   map[string]bool
		wantLimits, wantMultipleOf bool
	}{
		{"OpenAI model", strictSchemaKeywords(nil, "gpt-5.4-mini"), true, true},
		{"fine-tuned OpenAI model", strictSchemaKeywords(nil, "ft:gpt-4o-mini:acme::abc123"), false, false},
		{"xAI", strictSchemaKeywords(NewXAIProvider("k", "grok-4.7", nil).(*XAIProvider).schemaKeywords, "grok-4.7"), true, false},
		{"Gemini", strictSchemaKeywords(NewGoogleProvider("k", "gemini-3.5-flash", nil).(*GoogleProvider).schemaKeywords, "gemini-3.5-flash"), false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, []string{"positive", "negative", "neutral"}, field(tt.keywords, "sentiment")["enum"])
			assert.True(t, strings.HasPrefix(field(tt.keywords, "score")["description"].(string), "confidence, 0-1"), "caller's description kept first")
			for _, f := range []struct{ name, keyword string }{
				{"code", "minLength"}, {"code", "pattern"}, {"link", "format"}, {"name", "minimum"}, {"label", "maxItems"},
			} {
				assert.NotContains(t, field(tt.keywords, f.name), f.keyword, "%s.%s is never sent", f.name, f.keyword)
			}

			for _, f := range []struct{ name, keyword string }{
				{"score", "minimum"}, {"score", "maximum"}, {"count", "maximum"}, {"when", "format"}, {"tags", "maxItems"},
			} {
				_, kept := field(tt.keywords, f.name)[f.keyword]
				assert.Equal(t, tt.wantLimits, kept, "%s.%s", f.name, f.keyword)
			}
			_, kept := field(tt.keywords, "step")["multipleOf"]
			assert.Equal(t, tt.wantMultipleOf, kept, "step.multipleOf")
		})
	}
}

// The compat endpoints set their own keyword list; nil means OpenAI's.
func TestCompatProvidersSetSchemaKeywords(t *testing.T) {
	assert.Equal(t, baseSchemaKeywords, NewGoogleProvider("k", "gemini-3.5-flash", nil).(*GoogleProvider).schemaKeywords)
	assert.Equal(t, xaiSchemaKeywords, NewXAIProvider("k", "grok-4.7", nil).(*XAIProvider).schemaKeywords)
	assert.Nil(t, NewOpenAIProvider("k", "gpt-5.4-mini", nil).(*OpenAIProvider).schemaKeywords)
	assert.Nil(t, NewOpenAIResponsesProvider("k", "gpt-5.4-pro", nil).(*OpenAIResponsesProvider).schemaKeywords)
}

// A constraint the endpoint does not take is written into the field's description, so the model is
// told about a rule gollm's validator then checks; "const" becomes a one-value enum. A constraint on
// the wrong type for the field is neither sent nor described.
func TestCleanSchemaForOpenAIDescribesDroppedRules(t *testing.T) {
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"code":   map[string]interface{}{"type": "string", "maxLength": 5, "pattern": "^[A-Z]+$", "description": "A code."},
			"tags":   map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "string"}, "uniqueItems": true},
			"kind":   map[string]interface{}{"type": "string", "const": "fixed"},
			"score":  map[string]interface{}{"type": "number", "minimum": 0, "maximum": 1},
			"name":   map[string]interface{}{"type": "string", "minimum": 1},
			"rating": map[string]interface{}{"type": "number", "minimum": 0, "exclusiveMinimum": true},
		},
	}
	field := func(keywords map[string]bool, name string) map[string]interface{} {
		cleaned := cleanSchemaForOpenAI(schema, keywords).(map[string]interface{})
		return cleaned["properties"].(map[string]interface{})[name].(map[string]interface{})
	}
	openAI := strictSchemaKeywords(nil, "gpt-5.4-mini")
	gemini := baseSchemaKeywords

	code := field(openAI, "code")
	assert.Equal(t, `A code. At most 5 characters. Must match the regular expression "^[A-Z]+$".`, code["description"])
	assert.NotContains(t, code, "maxLength")
	assert.NotContains(t, code, "pattern")

	assert.Equal(t, "Items must be unique.", field(openAI, "tags")["description"])
	assert.Equal(t, []interface{}{"fixed"}, field(openAI, "kind")["enum"])
	assert.NotContains(t, field(openAI, "kind"), "const")

	assert.Equal(t, 0, field(openAI, "score")["minimum"], "OpenAI takes value limits as keywords")
	assert.NotContains(t, field(openAI, "score"), "description")
	assert.Equal(t, "At most 1. At least 0.", field(gemini, "score")["description"], "Gemini gets them as text")

	assert.NotContains(t, field(openAI, "name"), "minimum")
	assert.NotContains(t, field(openAI, "name"), "description", "a number limit on a string is not described")

	rating := field(openAI, "rating")
	assert.Equal(t, 0, rating["exclusiveMinimum"], "draft-04 boolean form becomes the numeric form")
	assert.NotContains(t, rating, "minimum")
}
