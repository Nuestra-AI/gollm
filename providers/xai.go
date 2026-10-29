// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"strconv"
	"strings"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/types"
)

// XAIProvider implements the Provider interface for xAI's Grok models.
//
// It embeds OpenAIResponsesProvider because xAI's primary API is a Responses API
// with OpenAI's request and response shapes: instructions/input, text.format for
// structured output, flat function tools, function_call/function_call_output
// items and the same typed stream events. What differs is the parameter
// vocabulary and the reasoning-effort levels, which are supplied through the
// embedded provider's allowedParams and adaptOptions hooks.
//
// Image input is stricter than OpenAI's and is passed through unconverted: xAI
// accepts JPG, PNG, WebP and ICO only (no GIF), and rejects images under 512
// pixels in total. Verified against the live API 2026-10-29.
type XAIProvider struct {
	OpenAIResponsesProvider
}

// xaiResponsesAllowedParams is the accepted top-level body of xAI's POST
// /v1/responses, per https://docs.x.ai/docs/api-reference (2026-10-27).
//
// Narrower than OpenAI's in places — no conversation, prompt, moderation or
// max_tool_calls — and wider in others: top_k, min_p, max_turns, guidelines and
// search_parameters. "background" is listed by xAI but marked unsupported, so it
// is omitted. reasoning_effort is absent because it is nested under reasoning
// before the filter runs.
var xaiResponsesAllowedParams = map[string]bool{
	"context_management": true, "guidelines": true, "include": true,
	"input": true, "instructions": true, "logprobs": true,
	"max_output_tokens": true, "max_turns": true, "metadata": true,
	"min_p": true, "model": true, "parallel_tool_calls": true,
	"previous_response_id": true, "prompt_cache_key": true, "reasoning": true,
	"safety_identifier": true, "search_parameters": true, "service_tier": true,
	"store": true, "stream": true, "temperature": true, "text": true,
	"tool_choice": true, "tools": true, "top_k": true, "top_logprobs": true,
	"top_p": true, "truncation": true, "user": true,
}

// xaiBuiltinTools are xAI's server-side tools that OpenAI lacks, sent as bare
// {"type": ...}. x_search searches posts on X. web_search is shared with OpenAI
// and handled by the embedded provider.
var xaiBuiltinTools = map[string]bool{"x_search": true}

// NewXAIProvider creates a new xAI provider instance.
//
// Parameters:
//   - apiKey: xAI API key for authentication
//   - model: The model to use (e.g., "grok-4.7")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured xAI Provider instance
func NewXAIProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	provider := &XAIProvider{
		OpenAIResponsesProvider: *NewOpenAIResponsesProvider(apiKey, model, extraHeaders).(*OpenAIResponsesProvider),
	}
	provider.allowedParams = xaiResponsesAllowedParams
	provider.adaptOptions = adaptXAIOptions
	provider.builtinTools = xaiBuiltinTools
	return provider
}

// Name returns "xai" as the provider identifier.
func (p *XAIProvider) Name() string {
	return "xai"
}

// Endpoint returns the xAI Responses API endpoint URL.
func (p *XAIProvider) Endpoint() string {
	return "https://api.x.ai/v1/responses"
}

// SetDefaultOptions forwards the sampling parameters xAI's Responses API accepts.
// See xaiSamplingParams.
func (p *XAIProvider) SetDefaultOptions(cfg *config.Config) {
	applySamplingDefaults(p, cfg, xaiSamplingParams)
}

// adaptXAIOptions brings merged options in line with xAI before they are nested
// into the Responses shape.
//
// reasoning_effort is clamped to a level the model takes, or dropped for a model
// that takes none — an unsupported value fails the whole request, as on OpenAI.
// verbosity is dropped: xAI's text object accepts only format. So is web_search's
// search_context_size, which xAI rejects ("Argument not supported") where OpenAI
// takes it.
func adaptXAIOptions(model string, request map[string]interface{}) {
	delete(request, "verbosity")
	if tools, ok := request["tools"].([]map[string]interface{}); ok {
		for _, tool := range tools {
			if tool["type"] == "web_search" {
				delete(tool, "search_context_size")
			}
		}
	}

	v, has := request["reasoning_effort"]
	if !has {
		return
	}
	str, ok := optionString(v)
	if !ok {
		delete(request, "reasoning_effort")
		return
	}
	normalized, ok := normalizeXAIReasoningEffort(model, str)
	if !ok {
		delete(request, "reasoning_effort")
		return
	}
	request["reasoning_effort"] = normalized
}

// normalizeXAIReasoningEffort maps a canonical effort level onto one the model
// accepts, clamping the ends of the scale inward like normalizeOpenAIReasoningEffort.
//
// xAI accepts low, medium, high and xhigh (grok-4.5 treats xhigh as high), and
// reasoning cannot be disabled, so none and minimal resolve to low and max to
// xhigh. https://docs.x.ai/docs/guides/reasoning (2026-10-27).
func normalizeXAIReasoningEffort(model, effort string) (string, bool) {
	if !xaiModelTakesReasoningEffort(model) {
		return "", false
	}
	switch effort {
	case string(types.ReasoningEffortMax), string(types.ReasoningEffortXHigh):
		return "xhigh", true
	case string(types.ReasoningEffortHigh), string(types.ReasoningEffortMedium), string(types.ReasoningEffortLow):
		return effort, true
	case string(types.ReasoningEffortMinimal), string(types.ReasoningEffortNone):
		return "low", true
	}
	return "", false
}

// xaiModelTakesReasoningEffort reports whether a Grok model accepts reasoning_effort.
//
// xAI documents it for grok-4.5 and later and for grok-4.20-multi-agent, where it
// sets the agent count. grok-4.3 is undocumented but accepts low through xhigh. The
// rest of the grok-4.20 line and grok-build-0.1 reject it with a 400 ("does not
// support parameter reasoningEffort"). Verified against the live API 2026-10-29.
// grok-4.20 predates grok-4.3, so it is excluded by name rather than by version order.
func xaiModelTakesReasoningEffort(model string) bool {
	if strings.Contains(model, "multi-agent") {
		return true
	}
	major, minor, ok := grokVersion(model)
	if !ok {
		return false
	}
	if major > 4 {
		return true
	}
	return major == 4 && minor >= 3 && minor != 20
}

// grokVersion parses the major and minor version from an id like "grok-4.7" or
// "grok-4.20-0309-reasoning". A bare major ("grok-5") has minor 0.
func grokVersion(model string) (major, minor int, ok bool) {
	rest, found := strings.CutPrefix(model, "grok-")
	if !found {
		return 0, 0, false
	}
	version, _, _ := strings.Cut(rest, "-")
	majorStr, minorStr, hasMinor := strings.Cut(version, ".")
	major, err := strconv.Atoi(majorStr)
	if err != nil {
		return 0, 0, false
	}
	if !hasMinor {
		return major, 0, true
	}
	minor, err = strconv.Atoi(minorStr)
	if err != nil {
		return 0, 0, false
	}
	return major, minor, true
}
