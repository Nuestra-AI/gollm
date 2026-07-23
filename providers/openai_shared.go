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
// and reasoning-output breakdowns that are billed differently from the totals containing them.
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

	// reasoning_effort: dropped for models that don't take it, clamped to a level the rest
	// actually accept. Nesting under reasoning.effort happens in applyResponsesReasoning.
	applyOpenAIReasoningEffort(model, merged)

	// verbosity: GPT-5 only. Normalized to a plain string here; nesting it under text — where
	// this API expects it — happens in applyResponsesVerbosity once the request body exists.
	applyOpenAIVerbosity(model, merged)

	// Reasoning models reject the whole sampling family, not just temperature.
	stripUnsupportedReasoningParams(model, merged)

	// parallel_tool_calls is unsupported on the o-series and on GPT-5 at minimal effort. This
	// matters more here than on Chat Completions: o3-pro and codex-mini are Responses-only.
	if modelNeedsNoParallelToolCalls(model, merged) {
		delete(merged, "parallel_tool_calls")
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

// isGPT5ChatModel returns true for the non-reasoning chat variants gpt-5-chat and
// gpt-5-chat-latest. They are GPT-5 models that reject reasoning_effort outright
// ("Invalid 'reasoning_effort' for non-reasoning model: gpt-5-chat-latest"), so every
// reasoning-only parameter has to be withheld from them.
//
// Deliberately anchored to the "gpt-5-chat" prefix rather than "chat" anywhere: gpt-5.1-chat
// and later chat variants DO support reasoning effort, so a looser match would strip a
// parameter those models accept.
func isGPT5ChatModel(model string) bool {
	return strings.HasPrefix(model, "gpt-5-chat")
}

// isCodexMiniModel returns true for the codex-mini reasoning model, which supports
// reasoning_effort but matches none of the naming patterns above — it is neither o<digit>
// nor gpt-prefixed.
func isCodexMiniModel(model string) bool {
	return strings.HasPrefix(model, "codex-mini")
}

// isO1MiniModel returns true for o1-mini, the one reasoning model that does not accept
// reasoning_effort at all.
func isO1MiniModel(model string) bool {
	return strings.HasPrefix(model, "o1-mini")
}

// gpt5MinorVersion returns the minor version of a GPT-5 model id — 0 for "gpt-5" and
// "gpt-5-mini", 1 for "gpt-5.1-codex", 6 for "gpt-5.6-sol" — and whether the model is a
// GPT-5 at all. Which reasoning_effort values a model accepts depends on this number, not
// on the family alone.
func gpt5MinorVersion(model string) (int, bool) {
	if !isGPT5Model(model) {
		return 0, false
	}
	rest := model[len("gpt-5"):]
	if !strings.HasPrefix(rest, ".") {
		return 0, true // "gpt-5", "gpt-5-mini", "gpt-5-pro", …
	}
	minor := 0
	digits := 0
	for _, c := range rest[1:] {
		if c < '0' || c > '9' {
			break
		}
		minor = minor*10 + int(c-'0')
		digits++
	}
	if digits == 0 {
		return 0, true
	}
	return minor, true
}

// isGPT5Model returns true for GPT-5 family models (gpt-5, gpt-5.1, gpt-5.4, etc.).
// Most are reasoning-capable; see isGPT5ChatModel for the exception.
//
// Known model IDs (as of 2026-03):
//
//	gpt-5, gpt-5-mini, gpt-5-nano, gpt-5-pro, gpt-5-codex, gpt-5-chat-latest
//	gpt-5.1, gpt-5.1-codex, gpt-5.2, gpt-5.2-pro, gpt-5.3-codex
//	gpt-5.4, gpt-5.4-pro, gpt-5.4-mini, gpt-5.4-nano
func isGPT5Model(model string) bool {
	return strings.HasPrefix(model, "gpt-5")
}

// isOpenAIFamilyModel reports whether a model id belongs to OpenAI's own catalogue.
//
// It exists because GoogleProvider and DeepSeekProvider embed OpenAIProvider to reuse the
// OpenAI-shaped wire format, and therefore inherit its SetOption. Their model ids
// ("gemini-2.5-flash", "deepseek-reasoner") must not be run through OpenAI's per-model
// parameter rules, which would strip parameters those providers accept.
func isOpenAIFamilyModel(model string) bool {
	return strings.HasPrefix(model, "gpt-") ||
		strings.HasPrefix(model, "chatgpt-") ||
		isOSeriesModel(model) ||
		isCodexMiniModel(model)
}

// modelNeedsMaxCompletionTokens checks if the model requires max_completion_tokens
// instead of max_tokens. All modern models (o-series, GPT-4o+, GPT-5) use this.
func modelNeedsMaxCompletionTokens(model string) bool {
	return isOSeriesModel(model) || isGPT4oOrNewerNonReasoning(model) || isGPT5Model(model)
}

// modelNeedsReasoningEffort checks if a given model supports the reasoning_effort parameter.
// The o-series (except o1-mini), codex-mini, and the GPT-5 family support it; GPT-4o/4.1 and
// the non-reasoning gpt-5-chat variants do not.
func modelNeedsReasoningEffort(model string) bool {
	if isO1MiniModel(model) || isGPT5ChatModel(model) {
		return false
	}
	return isOSeriesModel(model) || isCodexMiniModel(model) || isGPT5Model(model)
}

// modelSupportsVerbosity reports whether the model accepts the verbosity parameter, which hints
// how expansive the prose should be ("low", "medium", "high"). It is a GPT-5 reasoning feature:
// the o-series, every earlier family, and the non-reasoning gpt-5-chat variants reject it, so it
// must be stripped rather than passed through.
func modelSupportsVerbosity(model string) bool {
	return isGPT5Model(model) && !isGPT5ChatModel(model)
}

// applyOpenAIVerbosity drops a verbosity option the model cannot take, leaving a valid one in
// place. Like applyOpenAIReasoningEffort it is a no-op for non-OpenAI catalogues: Google and
// DeepSeek embed OpenAIProvider for its wire format, and OpenAI's per-model rules say nothing
// about their model ids, so judging their options by those rules would delete a caller's setting
// on a provider OpenAI has no say over.
func applyOpenAIVerbosity(model string, opts map[string]interface{}) {
	if !isOpenAIFamilyModel(model) {
		return
	}
	v, has := opts["verbosity"]
	if !has {
		return
	}
	// Only GPT-5 reasoning models accept it, and only at low/medium/high — an out-of-range
	// value is a 400 for the whole request, so it is dropped rather than sent.
	str, ok := optionString(v)
	if !modelSupportsVerbosity(model) || !ok || !validVerbosity(str) {
		delete(opts, "verbosity")
		return
	}
	opts["verbosity"] = str
}

// validVerbosity reports whether v is one of the values the API accepts. Anything else is dropped
// rather than sent, since an unknown value is a 400 for the whole request.
func validVerbosity(v string) bool {
	return v == "low" || v == "medium" || v == "high"
}

// applyResponsesVerbosity moves a flat verbosity option into text.verbosity, which is where the
// Responses API expects it — sent at the top level it is an unknown parameter and rejects the whole
// request. It merges into an existing text object rather than replacing it, because that object may
// already carry the structured-output format.
func applyResponsesVerbosity(request map[string]interface{}) {
	v, has := request["verbosity"]
	if !has {
		return
	}
	delete(request, "verbosity")
	str, ok := v.(string)
	if !ok || str == "" {
		return
	}
	text, _ := request["text"].(map[string]interface{})
	if text == nil {
		text = map[string]interface{}{}
	}
	text["verbosity"] = str
	request["text"] = text
}

// Which reasoning_effort values a model actually accepts, per the published support matrix.
// Sending an unsupported level is a 400 for the whole request, so callers pass the canonical
// cross-provider level (types.ReasoningEffort) and it is clamped here to something the model
// takes — the enum has always documented this clamping; these predicates are what implement it.

// supportsEffortNone reports whether "none" is accepted, i.e. reasoning can be switched off.
// GPT-5.1 and later only.
func supportsEffortNone(model string) bool {
	minor, ok := gpt5MinorVersion(model)
	return ok && minor >= 1
}

// supportsEffortMinimal reports whether "minimal" is accepted. It exists only on the original
// GPT-5 reasoning models and was dropped from gpt-5.1 onward; gpt-5-codex never had it.
func supportsEffortMinimal(model string) bool {
	minor, ok := gpt5MinorVersion(model)
	return ok && minor == 0 && !strings.HasPrefix(model, "gpt-5-codex")
}

// supportsEffortXHigh reports whether "xhigh" is accepted: gpt-5.4 and later, plus
// gpt-5.1-codex-max which introduced the level.
func supportsEffortXHigh(model string) bool {
	if strings.HasPrefix(model, "gpt-5.1-codex-max") {
		return true
	}
	minor, ok := gpt5MinorVersion(model)
	return ok && minor >= 4
}

// supportsEffortMax reports whether "max" is accepted: gpt-5.6 and later.
func supportsEffortMax(model string) bool {
	minor, ok := gpt5MinorVersion(model)
	return ok && minor >= 6
}

// normalizeOpenAIReasoningEffort maps a canonical effort level onto one the model accepts,
// clamping the ends of the scale inward rather than rejecting them. It reports false when the
// value cannot be used at all, in which case the caller drops the parameter.
//
// gpt-5-pro is the one model with a fixed level: it only runs at "high", and passing anything
// else is an error, so every level resolves there.
func normalizeOpenAIReasoningEffort(model, effort string) (string, bool) {
	// Not an OpenAI model: an embedded provider (Google, DeepSeek) is using the OpenAI wire
	// format with its own catalogue, and OpenAI's per-model rules say nothing about it.
	if !isOpenAIFamilyModel(model) {
		return effort, true
	}
	if !modelNeedsReasoningEffort(model) {
		return "", false
	}

	resolved, ok := resolveEffortLevel(model, effort)
	if !ok {
		return "", false
	}
	// A pinned model overrides the resolution, but only once the level is known to be a level
	// at all — an unrecognized value is dropped here exactly as it is everywhere else, rather
	// than silently honored as the pinned level.
	if pinned, isPinned := pinnedEffortModel(model); isPinned {
		return pinned, true
	}
	return resolved, true
}

// resolveEffortLevel clamps a canonical level to the nearest one the model accepts, reporting
// false for anything that is not a level.
func resolveEffortLevel(model, effort string) (string, bool) {
	switch effort {
	case string(types.ReasoningEffortMax):
		if supportsEffortMax(model) {
			return "max", true
		}
		if supportsEffortXHigh(model) {
			return "xhigh", true
		}
		return "high", true
	case string(types.ReasoningEffortXHigh):
		if supportsEffortXHigh(model) {
			return "xhigh", true
		}
		return "high", true
	case string(types.ReasoningEffortHigh), string(types.ReasoningEffortMedium), string(types.ReasoningEffortLow):
		// Accepted by every model that takes the parameter at all.
		return effort, true
	case string(types.ReasoningEffortMinimal):
		if supportsEffortMinimal(model) {
			return "minimal", true
		}
		// The nearest level that still runs: "none" would change the semantics more.
		return "low", true
	case string(types.ReasoningEffortNone):
		if supportsEffortNone(model) {
			return "none", true
		}
		if supportsEffortMinimal(model) {
			return "minimal", true
		}
		return "low", true
	default:
		// Unrecognized value: dropping it beats sending a 400 for the whole request.
		return "", false
	}
}

// pinnedEffortModel returns the single level a model is locked to, if it has one. gpt-5-pro only
// runs at "high" and rejects anything else, default included.
func pinnedEffortModel(model string) (string, bool) {
	if strings.HasPrefix(model, "gpt-5-pro") {
		return "high", true
	}
	return "", false
}

// applyOpenAIReasoningEffort normalizes or removes the reasoning_effort entry in a prepared
// option map, so every request-building path gates the parameter the same way.
func applyOpenAIReasoningEffort(model string, opts map[string]interface{}) {
	v, has := opts["reasoning_effort"]
	if !has {
		return
	}
	str, ok := optionString(v)
	if !ok {
		delete(opts, "reasoning_effort")
		return
	}
	normalized, ok := normalizeOpenAIReasoningEffort(model, str)
	if !ok {
		delete(opts, "reasoning_effort")
		return
	}
	opts["reasoning_effort"] = normalized
}

// reasoningUnsupportedParams are the sampling controls reasoning models reject outright. Sending
// any of them fails the whole request, and they arrive by way of generic per-request options, so
// they are stripped rather than forwarded. (max_tokens is handled separately, by conversion to
// max_completion_tokens/max_output_tokens rather than removal.)
var reasoningUnsupportedParams = []string{
	"temperature", "top_p", "presence_penalty", "frequency_penalty",
	"logprobs", "top_logprobs", "logit_bias",
}

// stripUnsupportedReasoningParams removes the sampling controls a reasoning model rejects.
func stripUnsupportedReasoningParams(model string, opts map[string]interface{}) {
	if !isOpenAIFamilyModel(model) || !modelNeedsNoTemperature(model) {
		return
	}
	for _, key := range reasoningUnsupportedParams {
		delete(opts, key)
	}
}

// applyResponsesReasoning moves a flat reasoning_effort option into reasoning.effort, which is
// where the Responses API expects it — sent flat it is an unknown parameter and rejects the whole
// request. It merges into an existing reasoning object rather than replacing it, so a caller's
// reasoning.summary survives.
func applyResponsesReasoning(request map[string]interface{}) {
	v, has := request["reasoning_effort"]
	if !has {
		return
	}
	delete(request, "reasoning_effort")
	str, ok := v.(string)
	if !ok || str == "" {
		return
	}
	reasoning, _ := request["reasoning"].(map[string]interface{})
	if reasoning == nil {
		reasoning = map[string]interface{}{}
	}
	reasoning["effort"] = str
	request["reasoning"] = reasoning
}
