package providers

import "github.com/teilomillet/gollm/utils"

// Parameter allowlists for the two OpenAI endpoints.
//
// Both providers copy caller options into the request body. On a denylist that
// forwards parameters the endpoint rejects, failing the whole request with
// "Unknown parameter" — and the sets differ, so seed or response_format survives
// a transport change and breaks a request that always worked. Unknown parameters
// are dropped silently; the debug line is for chasing a setting that seems to
// have no effect.
//
// Both lists are transcribed from the CreateResponse and CreateChatCompletionRequest
// schemas in github.com/openai/openai-openapi, read 2026-08-25. They are the API's
// vocabulary, not gollm's: control keys (tools, system_prompt, images,
// structured_messages, strict_tools) are absent because each provider converts
// them into real parameters before the filter runs.

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

// chatAllowedParams is the accepted body of POST /v1/chat/completions. Note what
// Responses lacks (frequency_penalty, logit_bias, n, response_format, seed, stop,
// max_tokens) and what neither takes: min_p and top_k, which gollm exposes setters
// for because other providers accept them.
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

// filterToOpenAIChatParams applies the allowlist to OpenAI's catalogue only.
//
// DeepSeek and Google embed OpenAIProvider for its wire format while serving their
// own models and parameters — google-openai carries Gemini's thinking budget in
// extra_body.google — so OpenAI's vocabulary would delete settings those endpoints
// accept. Scoped by model like stripUnsupportedReasoningParams. The Responses
// provider needs no such guard: nothing embeds it.
func filterToOpenAIChatParams(model string, request map[string]interface{}, logger utils.Logger) {
	if !isOpenAIFamilyModel(model) {
		return
	}
	filterToAllowedParams(request, chatAllowedParams, "/v1/chat/completions", logger)
}

// translateResponseFormatToText rewrites Chat's response_format as text.format,
// hoisting the nested json_schema fields the way this API expects. Translating
// rather than dropping keeps structured output working across a transport change
// instead of silently returning free text. A format already built by
// PrepareRequestWithSchema wins: it was requested for this call.
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
