package providers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/teilomillet/gollm/types"
)

// parseOpenAICompatStreamChunk parses one SSE data payload from an OpenAI-shaped chat completions
// stream into a normalized chunk carrying text, tool-call fragments, finish reason, and usage.
//
// Every OpenAI-compatible backend in this package (OpenAI, Groq, Mistral, vLLM, Lambda, Google and
// DeepSeek via embedding, and the generic OpenAI-type provider) streams this same shape, so they
// share one parser rather than each reimplementing it — and, more to the point, rather than each
// silently omitting usage. Unlike the text-only path it does NOT end the stream on finish_reason:
// the usage chunk (choices:[] + usage) arrives after it, followed by [DONE].
func parseOpenAICompatStreamChunk(chunk []byte) (types.StreamChunk, error) {
	trimmed := bytes.TrimSpace(chunk)
	if len(trimmed) == 0 {
		return types.StreamChunk{}, types.ErrStreamSkip
	}
	if bytes.Equal(trimmed, []byte("[DONE]")) {
		return types.StreamChunk{}, io.EOF
	}

	var response struct {
		Choices []struct {
			Delta struct {
				Role      string `json:"role"`
				Content   string `json:"content"`
				ToolCalls []struct {
					Index    int    `json:"index"`
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"delta"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *openAICompatUsage `json:"usage"`
		// Every chunk repeats the model that served it, which is the resolved one — a
		// gateway or a moving alias answers with something other than what was asked for.
		Model string `json:"model"`
		// The tier the request was served on, which scales the price of every token.
		ServiceTier string `json:"service_tier"`
	}
	if err := json.Unmarshal(trimmed, &response); err != nil {
		return types.StreamChunk{}, fmt.Errorf("malformed response: %w", err)
	}

	// Final usage-only chunk (choices empty, usage populated).
	if len(response.Choices) == 0 {
		if response.Usage != nil {
			return types.StreamChunk{Kind: "usage", Usage: response.Usage.normalize(), Model: response.Model, ServiceTier: response.ServiceTier}, nil
		}
		return types.StreamChunk{}, types.ErrStreamSkip
	}

	choice := response.Choices[0]
	// Tool-call fragment: the opening delta carries id+name, later deltas carry
	// partial-JSON argument pieces. OpenAI typically streams one tool_calls entry
	// per chunk (each with its own index), but a chunk may carry several; the
	// extras beyond the first are returned as ExtraToolCallDeltas.
	if len(choice.Delta.ToolCalls) > 0 {
		deltas := make([]*types.ToolCallDelta, len(choice.Delta.ToolCalls))
		for i, tc := range choice.Delta.ToolCalls {
			deltas[i] = &types.ToolCallDelta{
				Index:        tc.Index,
				ID:           tc.ID,
				Name:         tc.Function.Name,
				ArgsFragment: tc.Function.Arguments,
			}
		}
		return types.StreamChunk{Kind: "tool_call_delta", ToolCallDelta: deltas[0], ExtraToolCallDeltas: deltas[1:], Model: response.Model, ServiceTier: response.ServiceTier}, nil
	}
	if choice.FinishReason != "" {
		// Some gateways co-locate usage on the finish chunk instead of a separate
		// trailing chunk; capture it here so it isn't dropped.
		finish := types.StreamChunk{Kind: "finish", FinishReason: choice.FinishReason, Model: response.Model, ServiceTier: response.ServiceTier}
		if response.Usage != nil {
			finish.Usage = response.Usage.normalize()
		}
		return finish, nil
	}
	if choice.Delta.Content == "" {
		return types.StreamChunk{}, types.ErrStreamSkip
	}
	return types.StreamChunk{Kind: "text", Text: choice.Delta.Content, Model: response.Model, ServiceTier: response.ServiceTier}, nil
}

// openAICompatUsage is the usage object shared by the OpenAI-shaped APIs, including the cached-input
// and reasoning-output breakdowns that are billed differently from the totals.
type openAICompatUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
		// Cache writes bill at 1.25x the uncached input rate on GPT-5.6 and later
		// families; on earlier ones they were free, so this is a cost that only
		// becomes visible by reading it.
		CacheWriteTokens int `json:"cache_write_tokens"`
		AudioTokens      int `json:"audio_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
		AudioTokens     int `json:"audio_tokens"`
		// Predicted-output accounting. Rejected predictions never appear in the
		// response and are billed at the full output rate anyway, which makes them
		// the easiest tokens in the API to pay for without noticing.
		AcceptedPredictionTokens int `json:"accepted_prediction_tokens"`
		RejectedPredictionTokens int `json:"rejected_prediction_tokens"`
	} `json:"completion_tokens_details"`
}

func (u *openAICompatUsage) normalize() *types.TokenUsage {
	if u == nil {
		return nil
	}
	return &types.TokenUsage{
		PromptTokens:             u.PromptTokens,
		CompletionTokens:         u.CompletionTokens,
		TotalTokens:              u.TotalTokens,
		CachedPromptTokens:       u.PromptTokensDetails.CachedTokens,
		ReasoningTokens:          u.CompletionTokensDetails.ReasoningTokens,
		CacheWritePromptTokens:   u.PromptTokensDetails.CacheWriteTokens,
		AudioPromptTokens:        u.PromptTokensDetails.AudioTokens,
		AudioCompletionTokens:    u.CompletionTokensDetails.AudioTokens,
		AcceptedPredictionTokens: u.CompletionTokensDetails.AcceptedPredictionTokens,
		RejectedPredictionTokens: u.CompletionTokensDetails.RejectedPredictionTokens,
	}
}

// prepareOpenAICompatStreamRequest turns a prepared non-stream request body into a streaming one,
// optionally asking for the trailing usage chunk via stream_options.include_usage.
//
// defaultInclude sets the behaviour when the caller expresses no preference. It is false for the
// providers that gained usage streaming here (Groq, Mistral, vLLM, Lambda, and the generic
// OpenAI-type provider): stream_options is a comparatively recent addition to the OpenAI API, and
// self-hosted backends and compat gateways that predate it reject unknown fields outright, which
// would turn every stream into a 400. Those providers opt in with the "stream_usage" option.
// Providers already known to accept it — OpenAI and OpenRouter — request it by default in their own
// PrepareStreamRequest implementations.
func prepareOpenAICompatStreamRequest(body []byte, options map[string]interface{}, defaultInclude bool) ([]byte, error) {
	var requestBody map[string]interface{}
	if err := json.Unmarshal(body, &requestBody); err != nil {
		return nil, err
	}
	delete(requestBody, "stream_usage") // control key: must never reach the wire
	requestBody["stream"] = true

	include := defaultInclude
	if v, ok := options["stream_usage"].(bool); ok {
		include = v
	}
	if include {
		requestBody["stream_options"] = map[string]interface{}{"include_usage": true}
	}
	return json.Marshal(requestBody)
}

// optionString extracts a string from an option value, accepting either a plain
// string or a types.ReasoningEffort (the published cross-provider effort enum),
// so callers can pass either form to SetOption("reasoning_effort", ...).
func optionString(v interface{}) (string, bool) {
	switch s := v.(type) {
	case string:
		return s, true
	case types.ReasoningEffort:
		return string(s), true
	}
	return "", false
}

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
