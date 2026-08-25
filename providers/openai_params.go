package providers

import "github.com/teilomillet/gollm/utils"

// Parameter allowlists for the two OpenAI endpoints.
//
// Both providers build their request body by copying caller-supplied options into
// it. That was previously a denylist — anything not explicitly excluded was
// forwarded — which sends OpenAI parameters it does not accept and fails the whole
// request with "Unknown parameter". The failure mode was worst across transports:
// an option valid on Chat Completions (seed, stop, n, response_format) is not
// valid on Responses, so selecting a different endpoint could break a request that
// had always worked.
//
// An allowlist inverts that: only parameters the endpoint documents are sent, and
// anything else is dropped. Unknown options are dropped silently — a caller
// carrying settings across providers should not have to prune them per endpoint —
// with the detail available at debug level for when a setting appears not to take
// effect.
//
// Both lists are transcribed from the CreateResponse and CreateChatCompletionRequest
// schemas in OpenAI's published OpenAPI specification
// (github.com/openai/openai-openapi), read on 2026-08-25. They are the API's
// vocabulary, not gollm's: gollm's own control keys (tools, system_prompt, images,
// structured_messages, strict_tools) are absent by design, because each provider
// converts them into real parameters before the filter runs.

// responsesAllowedParams is the accepted top-level body of POST /v1/responses.
var responsesAllowedParams = map[string]bool{
	"background": true, "context_management": true, "conversation": true,
	"include": true, "input": true, "instructions": true,
	"max_output_tokens": true, "max_tool_calls": true, "metadata": true,
	"model": true, "moderation": true, "parallel_tool_calls": true,
	"previous_response_id": true, "prompt": true, "prompt_cache_key": true,
	"prompt_cache_options": true, "prompt_cache_retention": true,
	"reasoning": true, "safety_identifier": true, "service_tier": true,
	"store": true, "stream": true, "stream_options": true, "temperature": true,
	"text": true, "tool_choice": true, "tools": true, "top_logprobs": true,
	"top_p": true, "truncation": true, "user": true,
}

// chatAllowedParams is the accepted body of POST /v1/chat/completions.
//
// Note what is here that Responses lacks — frequency_penalty, presence_penalty,
// logit_bias, logprobs, n, response_format, seed, stop, max_tokens — and what is
// absent from both: min_p and top_k are not OpenAI parameters on either endpoint,
// though gollm exposes setters for them because other providers accept them.
var chatAllowedParams = map[string]bool{
	"audio": true, "frequency_penalty": true, "function_call": true,
	"functions": true, "logit_bias": true, "logprobs": true,
	"max_completion_tokens": true, "max_tokens": true, "messages": true,
	"metadata": true, "modalities": true, "model": true, "moderation": true,
	"n": true, "parallel_tool_calls": true, "prediction": true,
	"presence_penalty": true, "prompt_cache_key": true,
	"prompt_cache_options": true, "prompt_cache_retention": true,
	"reasoning_effort": true, "response_format": true, "safety_identifier": true,
	"seed": true, "service_tier": true, "stop": true, "store": true,
	"stream": true, "stream_options": true, "temperature": true,
	"tool_choice": true, "tools": true, "top_logprobs": true, "top_p": true,
	"user": true, "verbosity": true, "web_search_options": true,
}

// filterToAllowedParams drops every key the endpoint does not accept, in place.
//
// Dropping is silent by design; the debug line exists so a caller chasing a
// setting that appears to have no effect can see where it went.
func filterToAllowedParams(request map[string]interface{}, allowed map[string]bool, endpoint string, logger utils.Logger) {
	for key := range request {
		if allowed[key] {
			continue
		}
		delete(request, key)
		if logger != nil {
			logger.Debug("Dropped parameter not accepted by this endpoint",
				"parameter", key, "endpoint", endpoint)
		}
	}
}

// filterToOpenAIChatParams applies the Chat Completions allowlist, but only to
// OpenAI's own catalogue.
//
// DeepSeek and Google embed OpenAIProvider for its wire format while serving their
// own models and their own parameters — google-openai carries Gemini's thinking
// budget in an extra_body.google object, which is meaningful to that endpoint and
// absent from OpenAI's schema. Filtering those against OpenAI's vocabulary would
// silently delete settings the target API accepts, so the filter is scoped by
// model the same way stripUnsupportedReasoningParams and applyOpenAIVerbosity are.
//
// The Responses provider needs no such guard: nothing embeds it, and its endpoint
// is OpenAI's by construction.
func filterToOpenAIChatParams(model string, request map[string]interface{}, logger utils.Logger) {
	if !isOpenAIFamilyModel(model) {
		return
	}
	filterToAllowedParams(request, chatAllowedParams, "/v1/chat/completions", logger)
}

// translateResponseFormatToText rewrites a Chat Completions response_format into
// the Responses equivalent, text.format.
//
// The two APIs express structured output differently: Chat takes a top-level
// response_format, wrapping a JSON schema in a nested "json_schema" object, while
// Responses takes text.format with the schema's fields hoisted to the same level
// as the type. Translating rather than dropping means a caller who set
// response_format still gets structured output after a transport change, instead
// of silently receiving free text.
//
// A format already built by PrepareRequestWithSchema wins: that one was requested
// for this call, whereas response_format may be a stale provider-level default.
func translateResponseFormatToText(request map[string]interface{}) {
	raw, ok := request["response_format"]
	if !ok {
		return
	}
	delete(request, "response_format")

	format, ok := raw.(map[string]interface{})
	if !ok {
		return
	}
	if text, exists := request["text"].(map[string]interface{}); exists {
		if _, hasFormat := text["format"]; hasFormat {
			return
		}
	}

	// {"type":"json_schema","json_schema":{name,schema,strict}} → flattened.
	if nested, isSchema := format["json_schema"].(map[string]interface{}); isSchema {
		flat := map[string]interface{}{"type": "json_schema"}
		for _, key := range []string{"name", "schema", "strict", "description"} {
			if v, has := nested[key]; has {
				flat[key] = v
			}
		}
		format = flat
	}

	text, _ := request["text"].(map[string]interface{})
	if text == nil {
		text = map[string]interface{}{}
	}
	text["format"] = format
	request["text"] = text
}
