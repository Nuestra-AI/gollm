package providers

import (
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

	cleaned := cleanSchemaForOpenAI(schema).(map[string]interface{})
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

	cleaned := cleanSchemaForOpenAI(schema).(map[string]interface{})

	assert.NotContains(t, cleaned, "additionalProperties")
}

// Strict mode enforces "enum" and passes "description" to the model, so both survive cleaning;
// keywords it does not support, such as "minimum", are still removed.
func TestCleanSchemaForOpenAIKeepsEnumAndDescription(t *testing.T) {
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"sentiment": map[string]interface{}{"type": "string", "enum": []string{"positive", "negative", "neutral"}},
			"score":     map[string]interface{}{"type": "number", "minimum": 0, "description": "confidence, 0-1"},
		},
		"required": []string{"sentiment", "score"},
	}

	cleaned := cleanSchemaForOpenAI(schema).(map[string]interface{})
	props := cleaned["properties"].(map[string]interface{})
	sentiment := props["sentiment"].(map[string]interface{})
	score := props["score"].(map[string]interface{})

	assert.Equal(t, []string{"positive", "negative", "neutral"}, sentiment["enum"])
	assert.Equal(t, "confidence, 0-1", score["description"])
	assert.NotContains(t, score, "minimum")
}
