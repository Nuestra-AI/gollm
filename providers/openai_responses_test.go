package providers

import (
	"encoding/json"
	"io"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

func newTestResponsesProvider(model string) *OpenAIResponsesProvider {
	p := NewOpenAIResponsesProvider("test-key", model, nil)
	return p.(*OpenAIResponsesProvider)
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

func TestResponsesName(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")
	assert.Equal(t, "openai-responses", p.Name())
}

func TestResponsesEndpoint(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")
	assert.Equal(t, "https://api.openai.com/v1/responses", p.Endpoint())
}

func TestResponsesSupportsJSONSchema(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")
	assert.True(t, p.SupportsJSONSchema())
}

func TestResponsesSupportsStreaming(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")
	assert.True(t, p.SupportsStreaming())
}

func TestResponsesHeaders(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")
	headers := p.Headers()
	assert.Equal(t, "application/json", headers["Content-Type"])
	assert.Equal(t, "Bearer test-key", headers["Authorization"])
}

func TestResponsesExtraHeaders(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")
	p.SetExtraHeaders(map[string]string{"X-Custom": "value"})
	headers := p.Headers()
	assert.Equal(t, "value", headers["X-Custom"])
}

// ---------------------------------------------------------------------------
// PrepareRequest
// ---------------------------------------------------------------------------

func TestResponsesPrepareRequest(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")
	p.SetOption("temperature", 0.7)
	p.SetOption("max_tokens", 100)

	body, err := p.PrepareRequest("Hello world", map[string]interface{}{
		"system_prompt": "You are helpful",
	})
	require.NoError(t, err)

	var req map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &req))

	assert.Equal(t, "gpt-4o", req["model"])
	assert.Equal(t, "Hello world", req["input"])
	assert.Equal(t, "You are helpful", req["instructions"])
	assert.Equal(t, float64(0.7), req["temperature"])
	assert.Equal(t, float64(100), req["max_output_tokens"])
	assert.Nil(t, req["max_tokens"], "max_tokens should be converted to max_output_tokens")
}

func TestResponsesPrepareRequestNoTemperatureForO3(t *testing.T) {
	p := newTestResponsesProvider("o3-mini")
	p.SetOption("temperature", 0.7)
	p.SetOption("max_tokens", 100)

	body, err := p.PrepareRequest("Hello", map[string]interface{}{})
	require.NoError(t, err)

	var req map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &req))

	assert.Nil(t, req["temperature"], "o3 models should not have temperature")
	assert.Equal(t, float64(100), req["max_output_tokens"])
}

// ---------------------------------------------------------------------------
// PrepareRequestWithSchema
// ---------------------------------------------------------------------------

func TestResponsesPrepareRequestWithSchema(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")

	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"answer": map[string]interface{}{"type": "string"},
		},
		"required": []string{"answer"},
	}

	body, err := p.PrepareRequestWithSchema("What is 2+2?", map[string]interface{}{}, schema)
	require.NoError(t, err)

	var req map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &req))

	assert.Equal(t, "gpt-4o", req["model"])
	assert.Equal(t, "What is 2+2?", req["input"])

	// Verify text.format structure
	textObj, ok := req["text"].(map[string]interface{})
	require.True(t, ok, "text should be an object")
	formatObj, ok := textObj["format"].(map[string]interface{})
	require.True(t, ok, "text.format should be an object")
	assert.Equal(t, "json_schema", formatObj["type"])
	assert.Equal(t, "structured_response", formatObj["name"])
	assert.Equal(t, true, formatObj["strict"])
	assert.NotNil(t, formatObj["schema"])
}

// ---------------------------------------------------------------------------
// PrepareRequestWithMessages
// ---------------------------------------------------------------------------

func TestResponsesPrepareRequestWithMessages(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")

	messages := []types.MemoryMessage{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there!"},
		{Role: "user", Content: "Follow up"},
	}

	body, err := p.PrepareRequestWithMessages(messages, map[string]interface{}{
		"system_prompt": "Be helpful",
	})
	require.NoError(t, err)

	var req map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &req))

	assert.Equal(t, "Be helpful", req["instructions"])
	input, ok := req["input"].([]interface{})
	require.True(t, ok, "input should be an array for multi-turn")
	assert.Len(t, input, 3)

	// Check first message
	msg0 := input[0].(map[string]interface{})
	assert.Equal(t, "user", msg0["role"])
	assert.Equal(t, "Hello", msg0["content"])
}

func TestResponsesPrepareRequestWithMessagesToolCalls(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")

	messages := []types.MemoryMessage{
		{Role: "user", Content: "What's the weather?"},
		{
			Role: "assistant",
			ToolCalls: []types.ToolCall{
				{
					ID:   "call_123",
					Type: "function",
					Function: struct {
						Name      string          `json:"name"`
						Arguments json.RawMessage `json:"arguments"`
					}{
						Name:      "get_weather",
						Arguments: json.RawMessage(`{"city":"NYC"}`),
					},
				},
			},
		},
		{Role: "tool", ToolCallID: "call_123", Content: `{"temp": 72}`},
	}

	body, err := p.PrepareRequestWithMessages(messages, map[string]interface{}{})
	require.NoError(t, err)

	var req map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &req))

	input, ok := req["input"].([]interface{})
	require.True(t, ok)

	// Should have: user msg, function_call item, function_call_output item
	assert.Len(t, input, 3)

	// Function call item
	callItem := input[1].(map[string]interface{})
	assert.Equal(t, "function_call", callItem["type"])
	assert.Equal(t, "call_123", callItem["call_id"])
	assert.Equal(t, "get_weather", callItem["name"])

	// Function call output
	outputItem := input[2].(map[string]interface{})
	assert.Equal(t, "function_call_output", outputItem["type"])
	assert.Equal(t, "call_123", outputItem["call_id"])
	assert.Equal(t, `{"temp": 72}`, outputItem["output"])
}

// ---------------------------------------------------------------------------
// ParseResponse
// ---------------------------------------------------------------------------

func TestResponsesParseResponse(t *testing.T) {
	responseJSON := `{
		"id": "resp_abc123",
		"status": "completed",
		"model": "gpt-4o",
		"output": [
			{
				"type": "message",
				"role": "assistant",
				"content": [{"type": "output_text", "text": "The answer is 4."}]
			}
		],
		"usage": {
			"input_tokens": 50,
			"output_tokens": 10,
			"total_tokens": 60
		}
	}`

	p := newTestResponsesProvider("gpt-4o")
	text, err := p.ParseResponse([]byte(responseJSON))
	require.NoError(t, err)
	assert.Equal(t, "The answer is 4.", text)
}

func TestResponsesParseResponseWithUsage(t *testing.T) {
	responseJSON := `{
		"id": "resp_abc123",
		"status": "completed",
		"model": "gpt-4o",
		"output": [
			{
				"type": "message",
				"role": "assistant",
				"content": [{"type": "output_text", "text": "Hello!"}]
			}
		],
		"usage": {
			"input_tokens": 50,
			"output_tokens": 10,
			"total_tokens": 60
		}
	}`

	p := newTestResponsesProvider("gpt-4o")
	text, details, err := p.ParseResponseWithUsage([]byte(responseJSON))
	require.NoError(t, err)
	assert.Equal(t, "Hello!", text)
	require.NotNil(t, details)
	assert.Equal(t, "resp_abc123", details.ID)
	assert.Equal(t, "gpt-4o", details.Model)
	assert.Equal(t, 50, details.TokenUsage.PromptTokens)
	assert.Equal(t, 10, details.TokenUsage.CompletionTokens)
	assert.Equal(t, 60, details.TokenUsage.TotalTokens)
}

// ---------------------------------------------------------------------------
// ParseResponse — function calls
// ---------------------------------------------------------------------------

func TestResponsesParseFunctionCall(t *testing.T) {
	responseJSON := `{
		"id": "resp_fc",
		"status": "completed",
		"model": "gpt-4o",
		"output": [
			{
				"type": "function_call",
				"id": "fc_1",
				"call_id": "call_abc",
				"name": "get_weather",
				"arguments": "{\"city\":\"NYC\"}"
			}
		],
		"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
	}`

	p := newTestResponsesProvider("gpt-4o")
	text, err := p.ParseResponse([]byte(responseJSON))
	require.NoError(t, err)
	assert.Contains(t, text, "get_weather")
	assert.Contains(t, text, "NYC")
}

// ---------------------------------------------------------------------------
// ParseResponse — web search
// ---------------------------------------------------------------------------

func TestResponsesParseWebSearch(t *testing.T) {
	responseJSON := `{
		"id": "resp_ws",
		"status": "completed",
		"model": "gpt-4o",
		"output": [
			{
				"type": "web_search_call",
				"id": "ws_1",
				"status": "completed",
				"action": {
					"type": "search",
					"query": "weather NYC"
				}
			},
			{
				"type": "message",
				"role": "assistant",
				"content": [
					{
						"type": "output_text",
						"text": "It's sunny in NYC.",
						"annotations": [
							{
								"type": "url_citation",
								"start_index": 0,
								"end_index": 18,
								"url": "https://weather.com",
								"title": "Weather"
							}
						]
					}
				]
			}
		],
		"usage": {"input_tokens": 20, "output_tokens": 15, "total_tokens": 35}
	}`

	p := newTestResponsesProvider("gpt-4o")
	text, details, err := p.ParseResponseWithUsage([]byte(responseJSON))
	require.NoError(t, err)
	assert.Equal(t, "It's sunny in NYC.", text)
	require.NotNil(t, details)
	require.NotNil(t, details.Metadata)

	// Verify web search calls in metadata
	calls, ok := details.Metadata["web_search_calls"].([]types.WebSearchCall)
	require.True(t, ok)
	assert.Len(t, calls, 1)
	assert.Equal(t, "ws_1", calls[0].ID)

	// Verify citations
	citations, ok := details.Metadata["citations"].([]types.URLCitation)
	require.True(t, ok)
	assert.Len(t, citations, 1)
	assert.Equal(t, "https://weather.com", citations[0].URL)
}

// ---------------------------------------------------------------------------
// Option merging
// ---------------------------------------------------------------------------

func TestResponsesOptionMerging(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")
	p.SetOption("max_tokens", 200)

	// Verify it was stored as max_output_tokens
	assert.Equal(t, 200, p.options["max_output_tokens"])
	assert.Nil(t, p.options["max_tokens"])
}

func TestResponsesReasoningEffortFiltered(t *testing.T) {
	// GPT-4o does not support reasoning_effort
	p := newTestResponsesProvider("gpt-4o")
	p.SetOption("reasoning_effort", "medium")
	assert.Nil(t, p.options["reasoning_effort"])

	// o3-mini does support it
	p2 := newTestResponsesProvider("o3-mini")
	p2.SetOption("reasoning_effort", "medium")
	assert.Equal(t, "medium", p2.options["reasoning_effort"])
}

// ---------------------------------------------------------------------------
// Built-in tools
// ---------------------------------------------------------------------------

func TestResponsesBuiltinTools(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")

	tools := []utils.Tool{
		{Type: "web_search", SearchContextSize: "medium"},
		{Type: "file_search", VectorStoreIDs: []string{"vs_123"}},
		{
			Type: "function",
			Function: utils.Function{
				Name:        "get_weather",
				Description: "Get weather",
				Parameters:  map[string]interface{}{"type": "object"},
			},
		},
	}

	body, err := p.PrepareRequest("test", map[string]interface{}{
		"tools":        tools,
		"strict_tools": false,
	})
	require.NoError(t, err)

	var req map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &req))

	apiTools, ok := req["tools"].([]interface{})
	require.True(t, ok)
	assert.Len(t, apiTools, 3)

	// web_search tool
	ws := apiTools[0].(map[string]interface{})
	assert.Equal(t, "web_search", ws["type"])
	assert.Equal(t, "medium", ws["search_context_size"])

	// file_search tool
	fs := apiTools[1].(map[string]interface{})
	assert.Equal(t, "file_search", fs["type"])
	vsIDs, ok := fs["vector_store_ids"].([]interface{})
	require.True(t, ok)
	assert.Equal(t, "vs_123", vsIDs[0])

	// function tool
	fn := apiTools[2].(map[string]interface{})
	assert.Equal(t, "function", fn["type"])
	assert.Equal(t, "get_weather", fn["name"])
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

func TestResponsesPrepareStreamRequest(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")
	p.SetOption("temperature", 0.5)

	body, err := p.PrepareStreamRequest("Hello", map[string]interface{}{})
	require.NoError(t, err)

	var req map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &req))

	assert.Equal(t, true, req["stream"])
	assert.Equal(t, "gpt-4o", req["model"])
	assert.Equal(t, "Hello", req["input"])
}

func TestResponsesParseStreamDelta(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")

	chunk := `{"type": "response.output_text.delta", "delta": "Hello"}`
	text, err := p.ParseStreamResponse([]byte(chunk))
	require.NoError(t, err)
	assert.Equal(t, "Hello", text)
}

func TestResponsesParseStreamCompleted(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")

	chunk := `{"type": "response.completed"}`
	_, err := p.ParseStreamResponse([]byte(chunk))
	assert.ErrorIs(t, err, io.EOF)
}

func TestResponsesParseStreamDone(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")

	_, err := p.ParseStreamResponse([]byte("[DONE]"))
	assert.ErrorIs(t, err, io.EOF)
}

func TestResponsesParseStreamSkipsNonDelta(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")

	chunk := `{"type": "response.created"}`
	_, err := p.ParseStreamResponse([]byte(chunk))
	assert.Error(t, err)
	assert.NotErrorIs(t, err, io.EOF)
}

// ---------------------------------------------------------------------------
// PrepareRequestWithMessagesAndSchema
// ---------------------------------------------------------------------------

func TestResponsesPrepareRequestWithMessagesAndSchema(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")

	messages := []types.MemoryMessage{
		{Role: "user", Content: "Summarize this"},
		{Role: "assistant", Content: "Sure, here it is."},
		{Role: "user", Content: "Now in JSON"},
	}

	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"summary": map[string]interface{}{"type": "string"},
		},
		"required": []string{"summary"},
	}

	body, err := p.PrepareRequestWithMessagesAndSchema(messages, map[string]interface{}{
		"system_prompt": "Be concise",
	}, schema)
	require.NoError(t, err)

	var req map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &req))

	assert.Equal(t, "gpt-4o", req["model"])
	assert.Equal(t, "Be concise", req["instructions"])

	// Verify input is an array
	input, ok := req["input"].([]interface{})
	require.True(t, ok)
	assert.Len(t, input, 3)

	// Verify text.format structure
	textObj, ok := req["text"].(map[string]interface{})
	require.True(t, ok)
	formatObj, ok := textObj["format"].(map[string]interface{})
	require.True(t, ok)
	assert.Equal(t, "json_schema", formatObj["type"])
	assert.Equal(t, "structured_response", formatObj["name"])
	assert.NotNil(t, formatObj["schema"])
}

// ---------------------------------------------------------------------------
// Error / edge-case tests
// ---------------------------------------------------------------------------

func TestResponsesParseResponseMalformedJSON(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")
	_, err := p.ParseResponse([]byte(`{not json`))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "error parsing response")
}

func TestResponsesParseResponseEmptyOutput(t *testing.T) {
	responseJSON := `{
		"id": "resp_empty",
		"status": "completed",
		"model": "gpt-4o",
		"output": [],
		"usage": {"input_tokens": 5, "output_tokens": 0, "total_tokens": 5}
	}`

	p := newTestResponsesProvider("gpt-4o")
	_, err := p.ParseResponse([]byte(responseJSON))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no content in response")
}

func TestResponsesParseResponseWithUsageEmptyOutput(t *testing.T) {
	responseJSON := `{
		"id": "resp_empty",
		"status": "completed",
		"model": "gpt-4o",
		"output": [],
		"usage": {"input_tokens": 5, "output_tokens": 0, "total_tokens": 5}
	}`

	p := newTestResponsesProvider("gpt-4o")
	_, details, err := p.ParseResponseWithUsage([]byte(responseJSON))
	assert.Error(t, err)
	// Details should still be populated even on error
	require.NotNil(t, details)
	assert.Equal(t, "resp_empty", details.ID)
	assert.Equal(t, 5, details.TokenUsage.PromptTokens)
}

func TestResponsesParseStreamMalformedJSON(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")
	_, err := p.ParseStreamResponse([]byte(`{broken`))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "malformed response")
}

func TestResponsesParseStreamEmpty(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")
	_, err := p.ParseStreamResponse([]byte("  "))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "empty chunk")
}

func TestResponsesStreamExcludesStreamKey(t *testing.T) {
	p := newTestResponsesProvider("gpt-4o")

	// Passing stream in options should not double-set it
	body, err := p.PrepareRequest("Hello", map[string]interface{}{
		"stream": true,
	})
	require.NoError(t, err)

	var req map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &req))

	// stream should NOT appear in a non-streaming request
	assert.Nil(t, req["stream"])
}

// ---------------------------------------------------------------------------
// Model detection helpers
// ---------------------------------------------------------------------------

func TestIsOSeriesModel(t *testing.T) {
	// Should match o-series reasoning models
	for _, model := range []string{"o1", "o1-mini", "o1-preview", "o1-pro", "o3", "o3-mini", "o3-pro", "o3-deep-research", "o4-mini", "o4-mini-deep-research"} {
		assert.True(t, isOSeriesModel(model), "expected %q to be o-series", model)
	}
	// Should NOT match non-o-series models
	for _, model := range []string{"omni-moderation-latest", "ollama-llama3", "opus-3", "openrouter-model", "gpt-4o", "gpt-5", ""} {
		assert.False(t, isOSeriesModel(model), "expected %q to NOT be o-series", model)
	}
}

func TestIsGPT4oOrNewerNonReasoning(t *testing.T) {
	// Should match GPT-4o variants
	for _, model := range []string{"gpt-4o", "gpt-4o-mini", "gpt-4o-audio-preview", "gpt-4o-search-preview", "chatgpt-4o-latest"} {
		assert.True(t, isGPT4oOrNewerNonReasoning(model), "expected %q to match", model)
	}
	// Should match GPT-4.1 family
	for _, model := range []string{"gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"} {
		assert.True(t, isGPT4oOrNewerNonReasoning(model), "expected %q to match", model)
	}
	// Should match GPT-4.5
	assert.True(t, isGPT4oOrNewerNonReasoning("gpt-4.5-preview"), "expected gpt-4.5-preview to match")
	// Should NOT match older GPT-4 or other models
	for _, model := range []string{"gpt-4", "gpt-4-turbo", "gpt-4-turbo-preview", "gpt-5", "gpt-3.5-turbo", "o3"} {
		assert.False(t, isGPT4oOrNewerNonReasoning(model), "expected %q to NOT match", model)
	}
}

func TestIsGPT5Model(t *testing.T) {
	for _, model := range []string{"gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro", "gpt-5-codex", "gpt-5-chat-latest", "gpt-5.1", "gpt-5.1-codex", "gpt-5.2", "gpt-5.2-pro", "gpt-5.3-codex", "gpt-5.4", "gpt-5.4-pro", "gpt-5.4-mini", "gpt-5.4-nano"} {
		assert.True(t, isGPT5Model(model), "expected %q to be GPT-5", model)
	}
	for _, model := range []string{"gpt-4o", "gpt-4.1", "o3", "gpt-3.5-turbo"} {
		assert.False(t, isGPT5Model(model), "expected %q to NOT be GPT-5", model)
	}
}

func TestModelNeedsNoTemperature(t *testing.T) {
	// All o-series should reject temperature
	for _, model := range []string{"o1", "o1-mini", "o3", "o3-mini", "o4-mini"} {
		assert.True(t, modelNeedsNoTemperature(model), "expected %q to need no temperature", model)
	}
	// GPT-5 family should reject temperature
	for _, model := range []string{"gpt-5", "gpt-5.4", "gpt-5-mini"} {
		assert.True(t, modelNeedsNoTemperature(model), "expected %q to need no temperature", model)
	}
	// GPT-4o, GPT-4.1, GPT-4 should support temperature
	for _, model := range []string{"gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4", "gpt-3.5-turbo"} {
		assert.False(t, modelNeedsNoTemperature(model), "expected %q to support temperature", model)
	}
}

func TestModelNeedsNoToolChoice(t *testing.T) {
	// O-series should reject tool_choice
	for _, model := range []string{"o1", "o3", "o3-mini", "o4-mini"} {
		assert.True(t, modelNeedsNoToolChoice(model), "expected %q to need no tool_choice", model)
	}
	// GPT-4o, GPT-4.1 should support tool_choice
	for _, model := range []string{"gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4"} {
		assert.False(t, modelNeedsNoToolChoice(model), "expected %q to support tool_choice", model)
	}
}

func TestModelNeedsMaxCompletionTokens(t *testing.T) {
	// Modern models should use max_completion_tokens
	for _, model := range []string{"o1", "o3", "o4-mini", "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.5-preview", "chatgpt-4o-latest", "gpt-5", "gpt-5.4"} {
		assert.True(t, modelNeedsMaxCompletionTokens(model), "expected %q to need max_completion_tokens", model)
	}
	// Legacy models should use max_tokens
	for _, model := range []string{"gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "babbage-002", "davinci-002"} {
		assert.False(t, modelNeedsMaxCompletionTokens(model), "expected %q to use max_tokens", model)
	}
}

func TestModelNeedsReasoningEffort(t *testing.T) {
	// Reasoning models should support reasoning_effort
	for _, model := range []string{"o1", "o3", "o4-mini", "gpt-5", "gpt-5.4", "gpt-5-mini"} {
		assert.True(t, modelNeedsReasoningEffort(model), "expected %q to support reasoning_effort", model)
	}
	// Non-reasoning models should not
	for _, model := range []string{"gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4", "gpt-3.5-turbo"} {
		assert.False(t, modelNeedsReasoningEffort(model), "expected %q to NOT support reasoning_effort", model)
	}
}
