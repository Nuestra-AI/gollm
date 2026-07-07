package providers

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/teilomillet/gollm/types"
)

// decodeReq unmarshals a prepared request body and returns the decoded
// top-level request map plus the messages array as typed maps.
func decodeReq(t *testing.T, body []byte) (map[string]interface{}, []map[string]interface{}) {
	t.Helper()
	var req map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &req))
	var msgs []map[string]interface{}
	if raw, ok := req["messages"].([]interface{}); ok {
		for i, m := range raw {
			mm, ok := m.(map[string]interface{})
			require.Truef(t, ok, "messages[%d] is not a JSON object: %T", i, m)
			msgs = append(msgs, mm)
		}
	}
	return req, msgs
}

// TestOpenRouterSystemPromptInjection verifies that system_prompt (the gollm
// control key set by LLMImpl) is injected as a leading system message and never
// leaks to the wire as a raw field — across every request-building path.
func TestOpenRouterSystemPromptInjection(t *testing.T) {
	const sp = "You are a helpful assistant."

	t.Run("PrepareRequest injects system_prompt and strips control key", func(t *testing.T) {
		p := NewOpenRouterProvider("k", "openai/gpt-4o", nil).(*OpenRouterProvider)
		body, err := p.PrepareRequest("hi", map[string]interface{}{"system_prompt": sp})
		require.NoError(t, err)
		req, msgs := decodeReq(t, body)

		require.NotContains(t, req, "system_prompt", "control key must not leak to the wire")
		require.NotContains(t, req, "system_message")
		require.Len(t, msgs, 2)
		require.Equal(t, "system", msgs[0]["role"])
		require.Equal(t, sp, msgs[0]["content"])
		require.Equal(t, "user", msgs[1]["role"])
		require.Equal(t, "hi", msgs[1]["content"])
	})

	t.Run("system_prompt from p.options is honored", func(t *testing.T) {
		p := NewOpenRouterProvider("k", "openai/gpt-4o", nil).(*OpenRouterProvider)
		p.SetOption("system_prompt", sp) // mirrors LLMImpl.SetOption
		body, err := p.PrepareRequest("hi", nil)
		require.NoError(t, err)
		req, msgs := decodeReq(t, body)

		require.NotContains(t, req, "system_prompt")
		require.Len(t, msgs, 2)
		require.Equal(t, "system", msgs[0]["role"])
		require.Equal(t, sp, msgs[0]["content"])
	})

	t.Run("system_message is honored when system_prompt absent", func(t *testing.T) {
		p := NewOpenRouterProvider("k", "openai/gpt-4o", nil).(*OpenRouterProvider)
		body, err := p.PrepareRequest("hi", map[string]interface{}{"system_message": sp})
		require.NoError(t, err)
		req, msgs := decodeReq(t, body)

		require.NotContains(t, req, "system_message")
		require.Len(t, msgs, 2)
		require.Equal(t, "system", msgs[0]["role"])
		require.Equal(t, sp, msgs[0]["content"])
	})

	t.Run("system_prompt preferred over system_message", func(t *testing.T) {
		p := NewOpenRouterProvider("k", "openai/gpt-4o", nil).(*OpenRouterProvider)
		body, err := p.PrepareRequest("hi", map[string]interface{}{
			"system_prompt":  sp,
			"system_message": "IGNORED",
		})
		require.NoError(t, err)
		_, msgs := decodeReq(t, body)
		require.Equal(t, sp, msgs[0]["content"])
	})

	t.Run("PrepareRequestWithSchema injects system_prompt", func(t *testing.T) {
		p := NewOpenRouterProvider("k", "openai/gpt-4o", nil).(*OpenRouterProvider)
		schema := map[string]interface{}{"type": "object"}
		body, err := p.PrepareRequestWithSchema("hi", map[string]interface{}{"system_prompt": sp}, schema)
		require.NoError(t, err)
		req, msgs := decodeReq(t, body)

		require.NotContains(t, req, "system_prompt")
		require.Equal(t, "system", msgs[0]["role"])
		require.Equal(t, sp, msgs[0]["content"])
	})

	t.Run("PrepareRequestWithMessages injects system_prompt as leading message", func(t *testing.T) {
		p := NewOpenRouterProvider("k", "openai/gpt-4o", nil).(*OpenRouterProvider)
		msgsIn := []types.MemoryMessage{{Role: "user", Content: "hi"}}
		body, err := p.PrepareRequestWithMessages(msgsIn, map[string]interface{}{"system_prompt": sp})
		require.NoError(t, err)
		req, msgs := decodeReq(t, body)

		require.NotContains(t, req, "system_prompt")
		require.Len(t, msgs, 2)
		require.Equal(t, "system", msgs[0]["role"])
		require.Equal(t, sp, msgs[0]["content"])
		require.Equal(t, "user", msgs[1]["role"])
	})

	t.Run("PrepareStreamRequestWithMessages injects exactly one system message (no double-injection)", func(t *testing.T) {
		p := NewOpenRouterProvider("k", "openai/gpt-4o", nil).(*OpenRouterProvider)
		p.SetOption("system_prompt", sp) // present in p.options AND resolvable per-call
		msgsIn := []types.MemoryMessage{{Role: "user", Content: "hi"}}
		body, err := p.PrepareStreamRequestWithMessages(msgsIn, map[string]interface{}{"system_prompt": sp})
		require.NoError(t, err)
		req, msgs := decodeReq(t, body)

		require.NotContains(t, req, "system_prompt")
		require.Equal(t, true, req["stream"])
		// Exactly one system message — regression guard against the old prepend
		// (stream path) + re-inject (base method) double-injection.
		systemCount := 0
		for _, m := range msgs {
			if m["role"] == "system" {
				systemCount++
			}
		}
		require.Equal(t, 1, systemCount, "system prompt must be injected exactly once")
	})

	t.Run("PrepareCompletionRequest strips control keys (no messages array)", func(t *testing.T) {
		p := NewOpenRouterProvider("k", "openai/gpt-4o", nil).(*OpenRouterProvider)
		body, err := p.PrepareCompletionRequest("hi", map[string]interface{}{"system_prompt": sp})
		require.NoError(t, err)
		req, _ := decodeReq(t, body)
		require.NotContains(t, req, "system_prompt", "control key must not leak on the completions endpoint")
		require.Equal(t, "hi", req["prompt"])
	})
}

// TestOpenRouterSchemaNormalizationParity verifies that the messages+schema path
// normalizes the schema the same way the single-prompt schema path does, so both
// accept the same shapes and neither forwards a raw schema.
func TestOpenRouterSchemaNormalizationParity(t *testing.T) {
	p := NewOpenRouterProvider("k", "openai/gpt-4o", nil).(*OpenRouterProvider)
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"name": map[string]interface{}{"type": "string"},
		},
	}

	singleBody, err := p.PrepareRequestWithSchema("hi", nil, schema)
	require.NoError(t, err)
	msgBody, err := p.PrepareRequestWithMessagesAndSchema(
		[]types.MemoryMessage{{Role: "user", Content: "hi"}}, nil, schema)
	require.NoError(t, err)

	singleReq, _ := decodeReq(t, singleBody)
	msgReq, _ := decodeReq(t, msgBody)

	singleRF := singleReq["response_format"].(map[string]interface{})
	msgRF := msgReq["response_format"].(map[string]interface{})

	// Both paths must produce OpenRouter's json_schema envelope, identically.
	require.Equal(t, "json_schema", singleRF["type"])
	require.Equal(t, singleRF, msgRF,
		"messages+schema path must produce the same response_format as the single-prompt path")

	js := msgRF["json_schema"].(map[string]interface{})
	require.Equal(t, "structured_response", js["name"])
	require.Equal(t, true, js["strict"])
	require.Equal(t, schema, js["schema"])
}
