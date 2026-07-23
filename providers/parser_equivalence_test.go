package providers

import (
	"testing"

	"github.com/teilomillet/gollm/types"
)

// Generate and GenerateWithSchema now parse through ParseResponseWithUsage rather than
// ParseResponse, so that the round-trips they bill can be accounted for. That swap is only safe if
// the two parsers agree on the text they return — otherwise the busiest entrypoints in the library
// silently start producing different output.
//
// The two are separate implementations per provider, not one delegating to the other, so equivalence
// has to be asserted rather than assumed. Where they legitimately diverge it is because the usage
// parser understands content the text-only parser drops (server-side tool blocks, web search); those
// cases are pinned explicitly below rather than left to chance.
func TestParserEquivalenceAcrossEntrypoints(t *testing.T) {
	type parsers struct {
		text  func([]byte) (string, error)
		usage func([]byte) (string, *types.ResponseDetails, error)
	}

	openai := NewOpenAIProvider("key", "gpt-5", nil).(*OpenAIProvider)
	anthropic := NewAnthropicProvider("key", "claude-sonnet-4-5", nil).(*AnthropicProvider)
	openrouter := NewOpenRouterProvider("key", "anthropic/claude-3.5-sonnet", nil).(*OpenRouterProvider)
	cohere := NewCohereProvider("key", "command-r-plus-08-2024", nil).(*CohereProvider)

	cases := []struct {
		name string
		p    parsers
		body string
	}{
		{
			"openai plain text",
			parsers{openai.ParseResponse, openai.ParseResponseWithUsage},
			`{"id":"1","model":"gpt-5","choices":[{"message":{"content":"hello"},"finish_reason":"stop"}],
			  "usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
		},
		{
			"openai tool call",
			parsers{openai.ParseResponse, openai.ParseResponseWithUsage},
			`{"id":"1","model":"gpt-5","choices":[{"message":{"content":"","tool_calls":[
			   {"id":"call_1","type":"function","function":{"name":"get_weather","arguments":{"city":"Paris"}}}]},
			   "finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
		},
		{
			"anthropic plain text",
			parsers{anthropic.ParseResponse, anthropic.ParseResponseWithUsage},
			`{"id":"msg_1","model":"claude-sonnet-4-5","stop_reason":"end_turn",
			  "content":[{"type":"text","text":"hello"}],"usage":{"input_tokens":5,"output_tokens":2}}`,
		},
		{
			"anthropic text plus tool use",
			parsers{anthropic.ParseResponse, anthropic.ParseResponseWithUsage},
			`{"id":"msg_1","model":"claude-sonnet-4-5","stop_reason":"tool_use","content":[
			   {"type":"text","text":"let me check"},
			   {"type":"tool_use","id":"tu_1","name":"get_weather","input":{"city":"Paris"}}],
			  "usage":{"input_tokens":5,"output_tokens":2}}`,
		},
		{
			"openrouter chat completion",
			parsers{openrouter.ParseResponse, openrouter.ParseResponseWithUsage},
			`{"id":"gen_1","model":"anthropic/claude-3.5-sonnet",
			  "choices":[{"message":{"content":"hello"},"finish_reason":"stop"}],
			  "usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
		},
		{
			"openrouter text completion",
			parsers{openrouter.ParseResponse, openrouter.ParseResponseWithUsage},
			`{"id":"gen_1","model":"meta/llama","choices":[{"text":"hello"}],
			  "usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
		},
		{
			"cohere v2 message",
			parsers{cohere.ParseResponse, cohere.ParseResponseWithUsage},
			`{"id":"c1","message":{"role":"assistant","content":[{"type":"text","text":"hello"}]},
			  "usage":{"tokens":{"input_tokens":5,"output_tokens":2}}}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			textOut, textErr := tc.p.text([]byte(tc.body))
			usageOut, details, usageErr := tc.p.usage([]byte(tc.body))

			if (textErr == nil) != (usageErr == nil) {
				t.Fatalf("parsers disagree on success: ParseResponse err=%v, ParseResponseWithUsage err=%v", textErr, usageErr)
			}
			if textOut != usageOut {
				t.Errorf("text diverges between entrypoints:\n  ParseResponse:          %q\n  ParseResponseWithUsage: %q", textOut, usageOut)
			}
			if usageErr == nil && details == nil {
				t.Error("usage parser returned no details on a successful parse")
			}
		})
	}
}

// The two parsers diverge by design on server-side tool content: ParseResponseWithUsage handles
// blocks that ParseResponse ignores. Generate now takes the usage path, so this pins what that
// changes — the new output is the richer one, and this test is what would catch it changing again.
func TestUsageParserHandlesServerToolContent(t *testing.T) {
	anthropic := NewAnthropicProvider("key", "claude-sonnet-4-5", nil).(*AnthropicProvider)

	// A web-search turn: server_tool_use and web_search_tool_result blocks around the answer text.
	body := []byte(`{"id":"msg_1","model":"claude-sonnet-4-5","stop_reason":"end_turn","content":[
	  {"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"weather"}},
	  {"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[]},
	  {"type":"text","text":"it is sunny"}],
	  "usage":{"input_tokens":5,"output_tokens":2,"server_tool_use":{"web_search_requests":1}}}`)

	usageOut, details, err := anthropic.ParseResponseWithUsage(body)
	if err != nil {
		t.Fatalf("ParseResponseWithUsage: %v", err)
	}
	if usageOut == "" {
		t.Error("usage parser produced no text for a web-search turn")
	}
	if details == nil || details.Metadata["web_search_requests"] == nil {
		t.Error("web search request count not recorded: billed server-side tool use would be invisible")
	}
}
