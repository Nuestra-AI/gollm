package providers

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// ---------------------------------------------------------------------------
// toInt
// ---------------------------------------------------------------------------

func TestToInt(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expected int
	}{
		{"int", 42, 42},
		{"float64", float64(99), 99},
		{"int64", int64(123), 123},
		{"string returns 0", "hello", 0},
		{"nil returns 0", nil, 0},
		{"bool returns 0", true, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, toInt(tt.input))
		})
	}
}

// ---------------------------------------------------------------------------
// mergeOpenAIResponsesOptions
// ---------------------------------------------------------------------------

func TestMergeResponsesOptionsBasic(t *testing.T) {
	provider := map[string]interface{}{"temperature": 0.7, "max_tokens": 100}
	request := map[string]interface{}{"top_p": 0.9}

	merged := mergeOpenAIResponsesOptions("gpt-4o", provider, request, nil)

	assert.Equal(t, 0.7, merged["temperature"])
	assert.Equal(t, 0.9, merged["top_p"])
	// max_tokens should be converted to max_output_tokens
	assert.Equal(t, 100, merged["max_output_tokens"])
	assert.Nil(t, merged["max_tokens"])
}

func TestMergeResponsesOptionsRequestOverridesProvider(t *testing.T) {
	provider := map[string]interface{}{"temperature": 0.5}
	request := map[string]interface{}{"temperature": 0.9}

	merged := mergeOpenAIResponsesOptions("gpt-4o", provider, request, nil)

	assert.Equal(t, 0.9, merged["temperature"])
}

func TestMergeResponsesOptionsMaxCompletionTokensConverted(t *testing.T) {
	provider := map[string]interface{}{"max_completion_tokens": 200}
	request := map[string]interface{}{}

	merged := mergeOpenAIResponsesOptions("gpt-4o", provider, request, nil)

	assert.Equal(t, 200, merged["max_output_tokens"])
	assert.Nil(t, merged["max_completion_tokens"])
}

func TestMergeResponsesOptionsExcludeKeys(t *testing.T) {
	provider := map[string]interface{}{"temperature": 0.7, "system_prompt": "be nice"}
	request := map[string]interface{}{"tools": []string{"web_search"}}

	merged := mergeOpenAIResponsesOptions("gpt-4o", provider, request, []string{"system_prompt", "tools"})

	assert.Equal(t, 0.7, merged["temperature"])
	assert.Nil(t, merged["system_prompt"])
	assert.Nil(t, merged["tools"])
}

func TestMergeResponsesOptionsReasoningEffortFilteredForGPT4o(t *testing.T) {
	provider := map[string]interface{}{"reasoning_effort": "medium"}
	request := map[string]interface{}{}

	merged := mergeOpenAIResponsesOptions("gpt-4o", provider, request, nil)

	// GPT-4o does not support reasoning_effort
	assert.Nil(t, merged["reasoning_effort"])
}

func TestMergeResponsesOptionsReasoningEffortKeptForO3(t *testing.T) {
	provider := map[string]interface{}{"reasoning_effort": "medium"}
	request := map[string]interface{}{}

	merged := mergeOpenAIResponsesOptions("o3-mini", provider, request, nil)

	assert.Equal(t, "medium", merged["reasoning_effort"])
}

func TestMergeResponsesOptionsTemperatureRemovedForOSeries(t *testing.T) {
	provider := map[string]interface{}{"temperature": 0.7}
	request := map[string]interface{}{}

	merged := mergeOpenAIResponsesOptions("o3-mini", provider, request, nil)

	assert.Nil(t, merged["temperature"])
}

func TestMergeResponsesOptionsTemperatureRemovedForGPT5(t *testing.T) {
	provider := map[string]interface{}{"temperature": 0.7}
	request := map[string]interface{}{}

	merged := mergeOpenAIResponsesOptions("gpt-5", provider, request, nil)

	assert.Nil(t, merged["temperature"])
}

func TestMergeResponsesOptionsTemperatureKeptForGPT4o(t *testing.T) {
	provider := map[string]interface{}{"temperature": 0.7}
	request := map[string]interface{}{}

	merged := mergeOpenAIResponsesOptions("gpt-4o", provider, request, nil)

	assert.Equal(t, 0.7, merged["temperature"])
}
