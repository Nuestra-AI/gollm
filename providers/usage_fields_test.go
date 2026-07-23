package providers

import (
	"testing"

	"github.com/teilomillet/gollm/types"
)

// Anthropic reports extended-thinking output, the cache-write lifetime split, the served tier and
// server-side tool calls — all of which are billed, and none of which were read before. The
// lifetime split matters most: 5-minute writes bill at 1.25x input and 1-hour writes at 2x, so the
// aggregate cache_creation_input_tokens alone cannot price a cached call.
func TestAnthropicCapturesFullUsage(t *testing.T) {
	p := NewAnthropicProvider("key", "claude-sonnet-4-5", nil)
	body := `{"id":"msg_1","model":"claude-sonnet-4-5","stop_reason":"end_turn",
	  "content":[{"type":"text","text":"hi"}],
	  "usage":{"input_tokens":10,"output_tokens":50,
	           "cache_creation_input_tokens":30,"cache_read_input_tokens":7,
	           "cache_creation":{"ephemeral_5m_input_tokens":20,"ephemeral_1h_input_tokens":10},
	           "output_tokens_details":{"thinking_tokens":42},
	           "server_tool_use":{"web_search_requests":2,"web_fetch_requests":3},
	           "service_tier":"batch"}}`

	_, details, err := p.ParseResponseWithUsage([]byte(body))
	if err != nil {
		t.Fatalf("ParseResponseWithUsage: %v", err)
	}

	want := types.TokenUsage{
		PromptTokens: 10, CompletionTokens: 50, TotalTokens: 97,
		CacheCreationInputTokens: 30, CacheReadInputTokens: 7,
		CacheCreation5mInputTokens: 20, CacheCreation1hInputTokens: 10,
		ReasoningTokens: 42,
	}
	if details.TokenUsage != want {
		t.Errorf("usage = %+v,\n want %+v", details.TokenUsage, want)
	}
	// Batch is roughly half rate; without the tier the counts above price as standard.
	if details.ServiceTier != "batch" {
		t.Errorf("service tier = %q, want %q", details.ServiceTier, "batch")
	}
	if details.Metadata["web_search_requests"] != 2 || details.Metadata["web_fetch_requests"] != 3 {
		t.Errorf("server tool use not recorded: %+v", details.Metadata)
	}
}

// Newer responses send only the per-lifetime split. The aggregate has to be derived or cache writes
// vanish from the total entirely.
func TestAnthropicDerivesCacheCreationAggregate(t *testing.T) {
	p := NewAnthropicProvider("key", "claude-sonnet-4-5", nil)
	body := `{"id":"msg_1","model":"claude-sonnet-4-5","stop_reason":"end_turn",
	  "content":[{"type":"text","text":"hi"}],
	  "usage":{"input_tokens":5,"output_tokens":2,
	           "cache_creation":{"ephemeral_5m_input_tokens":100,"ephemeral_1h_input_tokens":40}}}`

	_, details, err := p.ParseResponseWithUsage([]byte(body))
	if err != nil {
		t.Fatalf("ParseResponseWithUsage: %v", err)
	}
	if got := details.TokenUsage.CacheCreationInputTokens; got != 140 {
		t.Errorf("cache creation aggregate = %d, want 140 (20+40 derived from the split)", got)
	}
	if got := details.TokenUsage.TotalTokens; got != 147 {
		t.Errorf("total = %d, want 147 — derived cache writes must reach the total", got)
	}
}

// Anthropic splits usage across stream events: message_start carries the input side and the cache
// breakdown, message_delta the output side and the thinking share.
func TestAnthropicStreamCapturesFullUsage(t *testing.T) {
	p := NewAnthropicProvider("key", "claude-sonnet-4-5", nil).(*AnthropicProvider)

	start, err := p.ParseStreamResponseRich([]byte(`{"type":"message_start","message":{"model":"claude-sonnet-4-5",
	  "usage":{"input_tokens":9,"cache_read_input_tokens":4,
	           "cache_creation":{"ephemeral_5m_input_tokens":6,"ephemeral_1h_input_tokens":2},
	           "service_tier":"priority"}}}`))
	if err != nil {
		t.Fatalf("message_start: %v", err)
	}
	if start.ServiceTier != "priority" {
		t.Errorf("service tier = %q, want %q", start.ServiceTier, "priority")
	}
	if start.Usage.CacheCreation5mInputTokens != 6 || start.Usage.CacheCreation1hInputTokens != 2 {
		t.Errorf("cache lifetime split lost: %+v", *start.Usage)
	}
	if start.Usage.CacheCreationInputTokens != 8 {
		t.Errorf("cache creation aggregate = %d, want 8", start.Usage.CacheCreationInputTokens)
	}

	delta, err := p.ParseStreamResponseRich([]byte(`{"type":"message_delta","delta":{"stop_reason":"end_turn"},
	  "usage":{"output_tokens":80,"output_tokens_details":{"thinking_tokens":64}}}`))
	if err != nil {
		t.Fatalf("message_delta: %v", err)
	}
	if delta.Usage.CompletionTokens != 80 || delta.Usage.ReasoningTokens != 64 {
		t.Errorf("output usage = %+v, want 80 completion of which 64 thinking", *delta.Usage)
	}
}

// Every one of these is billed and none was read before: cache writes at 1.25x, rejected
// predictions at the full output rate despite never appearing in the response, audio well above
// text, and the tier scaling all of it.
func TestOpenAICapturesFullUsage(t *testing.T) {
	p := NewOpenAIProvider("key", "gpt-5", nil)
	body := `{"id":"r1","model":"gpt-5","service_tier":"priority",
	  "choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],
	  "usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,
	           "prompt_tokens_details":{"cached_tokens":64,"cache_write_tokens":12,"audio_tokens":8},
	           "completion_tokens_details":{"reasoning_tokens":32,"audio_tokens":4,
	                                        "accepted_prediction_tokens":18,"rejected_prediction_tokens":10}}}`

	_, details, err := p.ParseResponseWithUsage([]byte(body))
	if err != nil {
		t.Fatalf("ParseResponseWithUsage: %v", err)
	}

	want := types.TokenUsage{
		PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150,
		CachedPromptTokens: 64, ReasoningTokens: 32,
		CacheWritePromptTokens: 12, AudioPromptTokens: 8, AudioCompletionTokens: 4,
		AcceptedPredictionTokens: 18, RejectedPredictionTokens: 10,
	}
	if details.TokenUsage != want {
		t.Errorf("usage = %+v,\n want %+v", details.TokenUsage, want)
	}
	if details.ServiceTier != "priority" {
		t.Errorf("service tier = %q, want %q", details.ServiceTier, "priority")
	}
}

// The same breakdowns arrive on the streaming usage chunk, and every OpenAI-compatible backend
// shares this parser.
func TestOpenAICompatStreamCapturesFullUsage(t *testing.T) {
	chunk := []byte(`{"model":"gpt-5","service_tier":"flex","choices":[],
	  "usage":{"prompt_tokens":11,"completion_tokens":22,"total_tokens":33,
	           "prompt_tokens_details":{"cached_tokens":5,"cache_write_tokens":3},
	           "completion_tokens_details":{"reasoning_tokens":7,"rejected_prediction_tokens":9}}}`)

	got, err := NewOpenAIProvider("key", "gpt-5", nil).(*OpenAIProvider).ParseStreamResponseRich(chunk)
	if err != nil {
		t.Fatalf("ParseStreamResponseRich: %v", err)
	}
	if got.ServiceTier != "flex" || got.Model != "gpt-5" {
		t.Errorf("model/tier = %q/%q, want gpt-5/flex", got.Model, got.ServiceTier)
	}
	want := types.TokenUsage{
		PromptTokens: 11, CompletionTokens: 22, TotalTokens: 33,
		CachedPromptTokens: 5, ReasoningTokens: 7,
		CacheWritePromptTokens: 3, RejectedPredictionTokens: 9,
	}
	if *got.Usage != want {
		t.Errorf("usage = %+v,\n want %+v", *got.Usage, want)
	}
}

// The Responses API names the same things differently (input_tokens_details, output_tokens_details).
func TestOpenAIResponsesCapturesFullUsage(t *testing.T) {
	p := NewOpenAIResponsesProvider("key", "gpt-5", nil)
	body := `{"id":"resp_1","model":"gpt-5","status":"completed","service_tier":"batch",
	  "output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}],
	  "usage":{"input_tokens":40,"output_tokens":20,"total_tokens":60,
	           "input_tokens_details":{"cached_tokens":16,"cache_write_tokens":4,"audio_tokens":2},
	           "output_tokens_details":{"reasoning_tokens":12,"audio_tokens":1}}}`

	_, details, err := p.ParseResponseWithUsage([]byte(body))
	if err != nil {
		t.Fatalf("ParseResponseWithUsage: %v", err)
	}
	want := types.TokenUsage{
		PromptTokens: 40, CompletionTokens: 20, TotalTokens: 60,
		CachedPromptTokens: 16, ReasoningTokens: 12,
		CacheWritePromptTokens: 4, AudioPromptTokens: 2, AudioCompletionTokens: 1,
	}
	if details.TokenUsage != want {
		t.Errorf("usage = %+v,\n want %+v", details.TokenUsage, want)
	}
	if details.ServiceTier != "batch" {
		t.Errorf("service tier = %q, want %q", details.ServiceTier, "batch")
	}
}

// verbosity is GPT-5-only and shaped differently per API: flat on Chat Completions, nested under
// text on Responses. Sent the wrong way, or to a model that doesn't take it, it fails the request.
