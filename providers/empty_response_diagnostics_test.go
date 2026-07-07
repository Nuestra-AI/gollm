package providers

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// emptyOpenAIBody is an OpenAI-shaped completion with no content and no tool
// calls, carrying a finish_reason and a non-zero completion_tokens count. The
// enriched empty-response error must surface BOTH diagnostics alongside the
// stable sentinel substring the backend classifies on. Groq, Mistral, and
// Gemini (via the OpenAI-compatible endpoint) all share this shape.
const emptyOpenAIBody = `{
	"id": "resp-123",
	"model": "test-model",
	"choices": [{"message": {"content": "", "tool_calls": []}, "finish_reason": "length"}],
	"usage": {"prompt_tokens": 10, "completion_tokens": 16384, "total_tokens": 16394}
}`

// emptyAnthropicBody is an Anthropic-shaped response with zero content blocks.
// Anthropic's native field is stop_reason; the diagnostic writes it under the
// finish_reason: label so classification is uniform. completion_tokens maps
// from usage.output_tokens.
const emptyAnthropicBody = `{
	"id": "msg-123",
	"model": "claude-test",
	"role": "assistant",
	"content": [],
	"stop_reason": "max_tokens",
	"usage": {"input_tokens": 10, "output_tokens": 16384}
}`

// emptyOpenRouterBody is an OpenRouter chat response with a choice present but
// empty content and no tool calls — the case that previously fell through to an
// empty string / mismatched error instead of the uniform sentinel.
const emptyOpenRouterBody = `{
	"id": "gen-123",
	"model": "test-model",
	"choices": [{"message": {"content": "", "tool_calls": []}, "finish_reason": "length"}],
	"usage": {"prompt_tokens": 10, "completion_tokens": 16384, "total_tokens": 16394}
}`

// assertEmptyDiagnostics checks that an empty-response error preserves the
// backend sentinel and appends the finish_reason + completion_tokens
// diagnostics, including the expected reason token and 16384 token count.
func assertEmptyDiagnostics(t *testing.T, err error, wantReason string) {
	t.Helper()
	require.Error(t, err)
	msg := err.Error()
	// Sentinel MUST remain intact and contiguous — the backend substring-matches it.
	require.Contains(t, msg, "no content or tool calls in response")
	require.Contains(t, msg, "finish_reason:")
	require.Contains(t, msg, `"`+wantReason+`"`)
	require.Contains(t, msg, "completion_tokens:")
	require.Contains(t, msg, "16384")
}

func TestEmptyResponseDiagnostics(t *testing.T) {
	cases := []struct {
		name       string
		provider   Provider
		body       string
		wantReason string
	}{
		{"openai", NewOpenAIProvider("k", "test-model", nil), emptyOpenAIBody, "length"},
		{"groq", NewGroqProvider("k", "test-model", nil), emptyOpenAIBody, "length"},
		{"mistral", NewMistralProvider("k", "test-model", nil), emptyOpenAIBody, "length"},
		// Gemini goes through the OpenAI-compatible endpoint and inherits
		// OpenAIProvider's parse methods, so it shares the OpenAI body/shape.
		{"gemini", NewGoogleProvider("k", "test-model", nil), emptyOpenAIBody, "length"},
		{"anthropic", NewAnthropicProvider("k", "claude-test", nil), emptyAnthropicBody, "max_tokens"},
		{"openrouter", NewOpenRouterProvider("k", "test-model", nil), emptyOpenRouterBody, "length"},
	}

	for _, tc := range cases {
		t.Run(tc.name+"/ParseResponse", func(t *testing.T) {
			_, err := tc.provider.ParseResponse([]byte(tc.body))
			assertEmptyDiagnostics(t, err, tc.wantReason)
		})
		t.Run(tc.name+"/ParseResponseWithUsage", func(t *testing.T) {
			_, _, err := tc.provider.ParseResponseWithUsage([]byte(tc.body))
			assertEmptyDiagnostics(t, err, tc.wantReason)
		})
	}
}

// TestEmptyResponseStopReasonZeroTokens covers the other diagnostic case: a
// plain stop with zero completion tokens (model declined / filtered), where
// bumping max_tokens would NOT help.
func TestEmptyResponseStopReasonZeroTokens(t *testing.T) {
	body := `{
		"choices": [{"message": {"content": "", "tool_calls": []}, "finish_reason": "content_filter"}],
		"usage": {"completion_tokens": 0}
	}`
	_, err := NewOpenAIProvider("k", "test-model", nil).ParseResponse([]byte(body))
	require.Error(t, err)
	require.Contains(t, err.Error(), "no content or tool calls in response")
	require.Contains(t, err.Error(), `"content_filter"`)
	require.Contains(t, err.Error(), "completion_tokens: 0")
	require.False(t, strings.Contains(err.Error(), "16384"))
}
