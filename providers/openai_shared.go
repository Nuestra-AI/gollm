package providers

import "strings"

// mergeOpenAIResponsesOptions combines provider defaults with per-request options
// for the Responses API, using max_output_tokens instead of max_completion_tokens.
// max_output_tokens (Responses API naming) instead of max_completion_tokens.
func mergeOpenAIResponsesOptions(model string, providerOpts, requestOpts map[string]interface{}, excludeKeys []string) map[string]interface{} {
	excluded := make(map[string]bool, len(excludeKeys))
	for _, k := range excludeKeys {
		excluded[k] = true
	}

	merged := make(map[string]interface{})

	// Provider defaults first
	for k, v := range providerOpts {
		if !excluded[k] {
			merged[k] = v
		}
	}

	// Request options override
	for k, v := range requestOpts {
		if !excluded[k] {
			merged[k] = v
		}
	}

	// Responses API uses max_output_tokens — convert from max_tokens or max_completion_tokens
	if v, has := merged["max_tokens"]; has {
		merged["max_output_tokens"] = v
		delete(merged, "max_tokens")
	}
	if v, has := merged["max_completion_tokens"]; has {
		merged["max_output_tokens"] = v
		delete(merged, "max_completion_tokens")
	}

	// reasoning_effort: only for models that support it
	if !modelNeedsReasoningEffort(model) {
		delete(merged, "reasoning_effort")
	}

	// temperature: some models don't support it
	if modelNeedsNoTemperature(model) {
		delete(merged, "temperature")
	}

	return merged
}

// toInt converts an interface{} value to int, handling both int and float64
// (the latter is what json.Unmarshal produces for numbers in interface{}).
func toInt(v interface{}) int {
	switch n := v.(type) {
	case int:
		return n
	case float64:
		return int(n)
	case int64:
		return int(n)
	default:
		return 0
	}
}

// isOSeriesModel returns true for OpenAI o-series reasoning models (o1, o3, o4-mini, etc.).
// It requires "o" followed by a digit to avoid false positives on unrelated models
// like "ollama-model", "opus-3", or "omni-moderation-latest".
func isOSeriesModel(model string) bool {
	return len(model) >= 2 && model[0] == 'o' && model[1] >= '0' && model[1] <= '9'
}

// isGPT4oOrNewerNonReasoning returns true for GPT-4o variants, GPT-4.1, and GPT-4.5
// — modern non-reasoning models that use max_completion_tokens but support temperature
// and tool_choice. This includes chatgpt-4o-latest.
//
// Known model IDs (as of 2026-03):
//
//	gpt-4o, gpt-4o-mini, gpt-4o-audio-preview, gpt-4o-search-preview, chatgpt-4o-latest
//	gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
//	gpt-4.5-preview
func isGPT4oOrNewerNonReasoning(model string) bool {
	return strings.HasPrefix(model, "gpt-4o") ||
		strings.HasPrefix(model, "gpt-4.1") ||
		strings.HasPrefix(model, "gpt-4.5") ||
		strings.HasPrefix(model, "chatgpt-4o")
}

// isGPT5Model returns true for GPT-5 family models (gpt-5, gpt-5.1, gpt-5.4, etc.).
// These are reasoning-capable models that support reasoning_effort.
//
// Known model IDs (as of 2026-03):
//
//	gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-pro, gpt-5-codex, gpt-5-chat-latest
//	gpt-5.1, gpt-5.1-codex, gpt-5.2, gpt-5.2-pro, gpt-5.3-codex
//	gpt-5.4, gpt-5.4-pro, gpt-5.4-mini, gpt-5.4-nano
func isGPT5Model(model string) bool {
	return strings.HasPrefix(model, "gpt-5")
}

// modelNeedsMaxCompletionTokens checks if the model requires max_completion_tokens
// instead of max_tokens. All modern models (o-series, GPT-4o+, GPT-5) use this.
func modelNeedsMaxCompletionTokens(model string) bool {
	return isOSeriesModel(model) || isGPT4oOrNewerNonReasoning(model) || isGPT5Model(model)
}

// modelNeedsReasoningEffort checks if a given model supports the reasoning_effort parameter.
// O-series reasoning models and GPT-5 family support it; GPT-4o/4.1 do not.
func modelNeedsReasoningEffort(model string) bool {
	return isOSeriesModel(model) || isGPT5Model(model)
}
