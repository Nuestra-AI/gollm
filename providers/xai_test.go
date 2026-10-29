package providers

import (
	"io"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

func newTestXAIProvider(model string) *XAIProvider {
	return NewXAIProvider("test-key", model, nil).(*XAIProvider)
}

func TestXAIIdentity(t *testing.T) {
	p := newTestXAIProvider("grok-4.7")
	assert.Equal(t, "xai", p.Name())
	assert.Equal(t, "https://api.x.ai/v1/responses", p.Endpoint())
	assert.Equal(t, "Bearer test-key", p.Headers()["Authorization"])
	assert.True(t, p.SupportsJSONSchema())
	assert.True(t, p.SupportsStreaming())
}

// TestXAIRequestShape verifies the Responses wire format carries over: system
// prompt as instructions, max_tokens as max_output_tokens, and retention off by
// default exactly as on OpenAI.
func TestXAIRequestShape(t *testing.T) {
	p := newTestXAIProvider("grok-4.7")
	body, err := p.PrepareRequest("hello", map[string]interface{}{
		"system_prompt": "be brief",
		"max_tokens":    256,
	})
	require.NoError(t, err)
	req := decodeBody(t, body)

	assert.Equal(t, "grok-4.7", req["model"])
	assert.Equal(t, "be brief", req["instructions"])
	assert.Equal(t, "hello", req["input"])
	assert.Equal(t, float64(256), req["max_output_tokens"])
	assert.Equal(t, false, req["store"])
	assert.NotContains(t, req, "max_tokens")
	assert.NotContains(t, req, "system_prompt")
}

// TestXAISamplingDefaults verifies xAI gets top_k and min_p, which it accepts and
// OpenAI does not, and none of the Chat-only parameters.
func TestXAISamplingDefaults(t *testing.T) {
	p := newTestXAIProvider("grok-4.7")
	p.SetDefaultOptions(fullConfig())
	body, err := p.PrepareRequest("hi", nil)
	require.NoError(t, err)
	req := decodeBody(t, body)

	assert.Equal(t, 0.3, req["temperature"])
	assert.Equal(t, 0.8, req["top_p"])
	assert.Equal(t, float64(40), req["top_k"])
	assert.Equal(t, 0.05, req["min_p"])
	assert.Equal(t, float64(512), req["max_output_tokens"])
	for _, key := range []string{"seed", "frequency_penalty", "presence_penalty", "stop", "repeat_penalty"} {
		assert.NotContains(t, req, key)
	}
}

// TestXAIAllowlist verifies xAI's own vocabulary is used: its extensions survive
// and OpenAI-only parameters are dropped rather than sent to a 400.
func TestXAIAllowlist(t *testing.T) {
	p := newTestXAIProvider("grok-4.7")
	body, err := p.PrepareRequest("hi", map[string]interface{}{
		"search_parameters": map[string]interface{}{"mode": "auto"},
		"max_turns":         3,
		"conversation":      "conv_123",
		"moderation":        true,
		"background":        true,
	})
	require.NoError(t, err)
	req := decodeBody(t, body)

	assert.Contains(t, req, "search_parameters")
	assert.Contains(t, req, "max_turns")
	assert.NotContains(t, req, "conversation")
	assert.NotContains(t, req, "moderation")
	assert.NotContains(t, req, "background")
}

// TestXAIWebSearchDropsContextSize verifies the one OpenAI web_search field xAI
// rejects is stripped while the tool and its other settings survive.
func TestXAIWebSearchDropsContextSize(t *testing.T) {
	p := newTestXAIProvider("grok-4.7")
	body, err := p.PrepareRequest("hi", map[string]interface{}{
		"tools": []utils.Tool{{Type: "web_search", SearchContextSize: "low", UserLocation: map[string]interface{}{"country": "US"}}},
	})
	require.NoError(t, err)
	req := decodeBody(t, body)

	tools := req["tools"].([]interface{})
	require.Len(t, tools, 1)
	tool := tools[0].(map[string]interface{})
	assert.Equal(t, "web_search", tool["type"])
	assert.NotContains(t, tool, "search_context_size")
	assert.Contains(t, tool, "user_location")
}

// TestXAIXSearchPassthrough verifies x_search is sent as xAI's built-in tool, not
// wrapped as a function with an empty name.
func TestXAIXSearchPassthrough(t *testing.T) {
	p := newTestXAIProvider("grok-4.7")
	body, err := p.PrepareRequest("hi", map[string]interface{}{
		"tools": []utils.Tool{{Type: "x_search"}},
	})
	require.NoError(t, err)
	req := decodeBody(t, body)

	tools := req["tools"].([]interface{})
	require.Len(t, tools, 1)
	assert.Equal(t, map[string]interface{}{"type": "x_search"}, tools[0])
}

func TestXAIDropsVerbosity(t *testing.T) {
	p := newTestXAIProvider("grok-4.7")
	body, err := p.PrepareRequestWithSchema("hi", map[string]interface{}{"verbosity": "low"},
		map[string]interface{}{"type": "object", "properties": map[string]interface{}{"a": map[string]interface{}{"type": "string"}}})
	require.NoError(t, err)
	req := decodeBody(t, body)

	text := req["text"].(map[string]interface{})
	assert.NotContains(t, text, "verbosity")
	format := text["format"].(map[string]interface{})
	assert.Equal(t, "json_schema", format["type"])
	assert.Equal(t, true, format["strict"])
}

func TestXAIReasoningEffort(t *testing.T) {
	tests := []struct {
		model  string
		effort interface{}
		want   string // "" means the parameter is dropped
	}{
		{"grok-4.7", "high", "high"},
		{"grok-4.7", types.ReasoningEffortMedium, "medium"},
		{"grok-4.7", types.ReasoningEffortMax, "xhigh"},
		{"grok-4.7", "xhigh", "xhigh"},
		{"grok-4.6", types.ReasoningEffortNone, "low"},
		{"grok-4.5", types.ReasoningEffortMinimal, "low"},
		{"grok-5", "low", "low"},
		{"grok-4.20-multi-agent-0309", "xhigh", "xhigh"},
		{"grok-4.20-0309-reasoning", "high", ""},
		{"grok-4.20-0309-non-reasoning", "high", ""},
		{"grok-4.3", "high", "high"},
		{"grok-4.2", "high", ""},
		{"grok-build-0.1", "high", ""},
		{"grok-4.7", "bogus", ""},
	}
	for _, tt := range tests {
		t.Run(tt.model+"/"+optionStringOrEmpty(tt.effort), func(t *testing.T) {
			p := newTestXAIProvider(tt.model)
			body, err := p.PrepareRequest("hi", map[string]interface{}{"reasoning_effort": tt.effort})
			require.NoError(t, err)
			req := decodeBody(t, body)

			assert.NotContains(t, req, "reasoning_effort", "must be nested, never sent flat")
			if tt.want == "" {
				assert.NotContains(t, req, "reasoning")
				return
			}
			reasoning := req["reasoning"].(map[string]interface{})
			assert.Equal(t, tt.want, reasoning["effort"])
		})
	}
}

// TestXAIReasoningEffortFromSetOption covers the provider-level path, which reaches
// the request through the merge rather than per-request options.
func TestXAIReasoningEffortFromSetOption(t *testing.T) {
	p := newTestXAIProvider("grok-4.7")
	p.SetOption("reasoning_effort", types.ReasoningEffortMax)
	body, err := p.PrepareStreamRequest("hi", nil)
	require.NoError(t, err)
	req := decodeBody(t, body)

	assert.Equal(t, true, req["stream"])
	assert.Equal(t, "xhigh", req["reasoning"].(map[string]interface{})["effort"])
}

// TestXAIParseResponseSkipsReasoning verifies the encrypted reasoning item grok-4.7
// returns on every response does not leak into, or displace, the text.
func TestXAIParseResponseSkipsReasoning(t *testing.T) {
	p := newTestXAIProvider("grok-4.7")
	body := []byte(`{
		"id": "resp_1", "status": "completed", "model": "grok-4.7",
		"output": [
			{"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque", "summary": []},
			{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi there"}]}
		],
		"usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
			"output_tokens_details": {"reasoning_tokens": 15}}
	}`)
	text, details, err := p.ParseResponseWithUsage(body)
	require.NoError(t, err)
	assert.Equal(t, "Hi there", text)
	require.NotNil(t, details)
	assert.Equal(t, 15, details.TokenUsage.ReasoningTokens)
}

// TestXAIStreamDoneSentinel verifies the [DONE] terminator xAI sends ends the stream.
func TestXAIStreamDoneSentinel(t *testing.T) {
	p := newTestXAIProvider("grok-4.7")
	chunk, err := p.ParseStreamResponseRich([]byte(`{"type":"response.output_text.delta","delta":"Hi"}`))
	require.NoError(t, err)
	assert.Equal(t, "Hi", chunk.Text)

	_, err = p.ParseStreamResponseRich([]byte("[DONE]"))
	assert.ErrorIs(t, err, io.EOF)
}

func optionStringOrEmpty(v interface{}) string {
	s, _ := optionString(v)
	return s
}
