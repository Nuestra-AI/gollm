package providers

import (
	"encoding/json"
	"testing"

	"github.com/teilomillet/gollm/types"
)

// Ollama reports its counts as prompt_eval_count/eval_count on the terminal object rather than in a
// usage object. It previously returned nil details, so every Ollama call went unaccounted for.
func TestOllamaParsesTokenCounts(t *testing.T) {
	p := NewOllamaProvider("http://localhost:11434", "llama3", nil)
	body := `{"model":"llama3","response":"hi ","done":false}
{"model":"llama3","response":"there","done":true,"prompt_eval_count":18,"eval_count":7}`

	text, details, err := p.ParseResponseWithUsage([]byte(body))
	if err != nil {
		t.Fatalf("ParseResponseWithUsage: %v", err)
	}
	if text != "hi there" {
		t.Errorf("text = %q, want %q", text, "hi there")
	}
	if details == nil {
		t.Fatal("details is nil: Ollama usage would go unrecorded")
	}
	want := types.TokenUsage{PromptTokens: 18, CompletionTokens: 7, TotalTokens: 25}
	if details.TokenUsage != want {
		t.Errorf("usage = %+v, want %+v", details.TokenUsage, want)
	}
}

// Ollama's streamed usage arrives only on the done object; without a rich parser the whole cost of a
// streamed generation is invisible.
func TestOllamaStreamReportsUsageOnDone(t *testing.T) {
	p := NewOllamaProvider("http://localhost:11434", "llama3", nil).(*OllamaProvider)

	chunk, err := p.ParseStreamResponseRich([]byte(`{"response":"tok","done":false}`))
	if err != nil {
		t.Fatalf("text chunk: %v", err)
	}
	if chunk.Kind != "text" || chunk.Text != "tok" {
		t.Errorf("chunk = %+v, want a text chunk carrying %q", chunk, "tok")
	}

	chunk, err = p.ParseStreamResponseRich([]byte(`{"response":"","done":true,"done_reason":"stop","prompt_eval_count":4,"eval_count":9}`))
	if err != nil {
		t.Fatalf("done chunk: %v", err)
	}
	if chunk.Usage == nil {
		t.Fatal("terminal chunk carries no usage")
	}
	want := types.TokenUsage{PromptTokens: 4, CompletionTokens: 9, TotalTokens: 13}
	if *chunk.Usage != want {
		t.Errorf("usage = %+v, want %+v", *chunk.Usage, want)
	}
}

// Bedrock discarded usage entirely, across every model family. Each family reports counts under
// different keys, so all of them are checked.
func TestBedrockParsesUsagePerModelFamily(t *testing.T) {
	cases := []struct {
		name  string
		model string
		body  string
		want  types.TokenUsage
	}{
		{
			"anthropic", "anthropic.claude-3-sonnet-20240229-v1:0",
			`{"content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":30,"output_tokens":12}}`,
			types.TokenUsage{PromptTokens: 30, CompletionTokens: 12, TotalTokens: 42},
		},
		{
			"anthropic with cache", "anthropic.claude-3-sonnet-20240229-v1:0",
			`{"content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":5,"output_tokens":2,
			  "cache_creation_input_tokens":11,"cache_read_input_tokens":4}}`,
			// Cache writes and reads are billed on top of the uncached input, so the total covers
			// all four components — the rule types.TokenUsage.ComputedTotal defines.
			types.TokenUsage{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 22, CacheCreationInputTokens: 11, CacheReadInputTokens: 4},
		},
		{
			"meta", "meta.llama3-70b-instruct-v1:0",
			`{"generation":"hi","prompt_token_count":21,"generation_token_count":3}`,
			types.TokenUsage{PromptTokens: 21, CompletionTokens: 3, TotalTokens: 24},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := NewBedrockProvider("", tc.model, nil)
			_, details, err := p.ParseResponseWithUsage([]byte(tc.body))
			if err != nil {
				t.Fatalf("ParseResponseWithUsage: %v", err)
			}
			if details == nil {
				t.Fatal("details is nil: Bedrock usage would go unrecorded")
			}
			if details.TokenUsage != tc.want {
				t.Errorf("usage = %+v, want %+v", details.TokenUsage, tc.want)
			}
		})
	}
}

// Cohere v2 nests its counts under usage.tokens; reading the flat keys silently yielded zero.
func TestCohereParsesV2NestedUsage(t *testing.T) {
	p := NewCohereProvider("key", "command-r-plus-08-2024", nil)
	body := `{"id":"c1","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]},
	          "usage":{"billed_units":{"input_tokens":14,"output_tokens":6},
	                   "tokens":{"input_tokens":14,"output_tokens":6}}}`

	_, details, err := p.ParseResponseWithUsage([]byte(body))
	if err != nil {
		t.Fatalf("ParseResponseWithUsage: %v", err)
	}
	want := types.TokenUsage{PromptTokens: 14, CompletionTokens: 6, TotalTokens: 20}
	if details.TokenUsage != want {
		t.Errorf("usage = %+v, want %+v", details.TokenUsage, want)
	}
}

// Reasoning tokens are billed at the output rate and cached input at a discount, so a usage record
// without them can't produce a correct cost.
func TestOpenAICapturesCachedAndReasoningTokens(t *testing.T) {
	p := NewOpenAIProvider("key", "gpt-5", nil)
	body := `{"id":"r1","model":"gpt-5","choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],
	          "usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,
	                   "prompt_tokens_details":{"cached_tokens":64},
	                   "completion_tokens_details":{"reasoning_tokens":32}}}`

	_, details, err := p.ParseResponseWithUsage([]byte(body))
	if err != nil {
		t.Fatalf("ParseResponseWithUsage: %v", err)
	}
	if details.TokenUsage.CachedPromptTokens != 64 {
		t.Errorf("cached prompt tokens = %d, want 64", details.TokenUsage.CachedPromptTokens)
	}
	if details.TokenUsage.ReasoningTokens != 32 {
		t.Errorf("reasoning tokens = %d, want 32", details.TokenUsage.ReasoningTokens)
	}
}

// The OpenAI-compatible providers all stream the same shape, and all of them previously dropped the
// trailing usage chunk because none implemented a rich parser.
func TestOpenAICompatProvidersParseStreamUsage(t *testing.T) {
	usageChunk := []byte(`{"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":22,"total_tokens":33,
	                       "completion_tokens_details":{"reasoning_tokens":5}}}`)
	want := types.TokenUsage{PromptTokens: 11, CompletionTokens: 22, TotalTokens: 33, ReasoningTokens: 5}

	parsers := map[string]func([]byte) (types.StreamChunk, error){
		"groq":    NewGroqProvider("key", "llama-3.1-70b", nil).(*GroqProvider).ParseStreamResponseRich,
		"mistral": NewMistralProvider("key", "mistral-large", nil).(*MistralProvider).ParseStreamResponseRich,
		"vllm":    NewVLLMProvider("http://localhost:8000", "qwen", nil).(*VLLMProvider).ParseStreamResponseRich,
		"lambda":  NewLambdaProvider("key", "hermes-3", nil).(*LambdaProvider).ParseStreamResponseRich,
		"openai":  NewOpenAIProvider("key", "gpt-5", nil).(*OpenAIProvider).ParseStreamResponseRich,
	}

	for name, parse := range parsers {
		t.Run(name, func(t *testing.T) {
			chunk, err := parse(usageChunk)
			if err != nil {
				t.Fatalf("ParseStreamResponseRich: %v", err)
			}
			if chunk.Usage == nil {
				t.Fatal("usage chunk parsed without usage")
			}
			if *chunk.Usage != want {
				t.Errorf("usage = %+v, want %+v", *chunk.Usage, want)
			}
		})
	}
}

// streamRequestShape is the part of a prepared streaming body these tests assert on.
type streamRequestShape struct {
	Stream        bool `json:"stream"`
	StreamOptions *struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
}

// These providers front self-hosted backends and compat gateways that predate stream_options and
// reject unknown request fields, so asking for a usage chunk by default would turn every stream into
// a 400. Usage streaming is opt-in for them; the flag must be absent unless it is asked for.
func TestOpenAICompatStreamUsageIsOptIn(t *testing.T) {
	builders := map[string]func(string, map[string]interface{}) ([]byte, error){
		"groq":    NewGroqProvider("key", "llama-3.1-70b", nil).PrepareStreamRequest,
		"mistral": NewMistralProvider("key", "mistral-large", nil).PrepareStreamRequest,
		"vllm":    NewVLLMProvider("http://localhost:8000", "qwen", nil).PrepareStreamRequest,
		"lambda":  NewLambdaProvider("key", "hermes-3", nil).PrepareStreamRequest,
		"aliyun":  NewAliyunProvider("key", "qwen-max", nil).PrepareStreamRequest,
	}

	for name, build := range builders {
		t.Run(name, func(t *testing.T) {
			body, err := build("hi", map[string]interface{}{})
			if err != nil {
				t.Fatalf("PrepareStreamRequest: %v", err)
			}
			var req streamRequestShape
			if err := json.Unmarshal(body, &req); err != nil {
				t.Fatalf("unmarshal request: %v", err)
			}
			if !req.Stream {
				t.Error("stream flag missing from the streaming request")
			}
			if req.StreamOptions != nil {
				t.Errorf("stream_options sent without opt-in, which 400s on gateways that reject it: %s", body)
			}

			// Opting in produces the flag, and the control key never reaches the wire.
			body, err = build("hi", map[string]interface{}{"stream_usage": true})
			if err != nil {
				t.Fatalf("PrepareStreamRequest (opted in): %v", err)
			}
			var optedIn map[string]interface{}
			if err := json.Unmarshal(body, &optedIn); err != nil {
				t.Fatalf("unmarshal request: %v", err)
			}
			opts, ok := optedIn["stream_options"].(map[string]interface{})
			if !ok || opts["include_usage"] != true {
				t.Errorf("stream_usage=true did not request include_usage: %s", body)
			}
			if _, leaked := optedIn["stream_usage"]; leaked {
				t.Errorf("control key stream_usage leaked to the wire: %s", body)
			}
		})
	}
}

// OpenAI and OpenRouter are known to accept stream_options, and asked for usage before this change,
// so they must keep requesting it by default — these are the providers whose streaming usage is
// relied on in production.
func TestOpenAIAndOpenRouterRequestStreamUsageByDefault(t *testing.T) {
	builders := map[string]func(string, map[string]interface{}) ([]byte, error){
		"openai":     NewOpenAIProvider("key", "gpt-5", nil).PrepareStreamRequest,
		"openrouter": NewOpenRouterProvider("key", "anthropic/claude-3.5-sonnet", nil).PrepareStreamRequest,
	}

	for name, build := range builders {
		t.Run(name, func(t *testing.T) {
			body, err := build("hi", map[string]interface{}{})
			if err != nil {
				t.Fatalf("PrepareStreamRequest: %v", err)
			}
			var req map[string]interface{}
			if err := json.Unmarshal(body, &req); err != nil {
				t.Fatalf("unmarshal request: %v", err)
			}
			// OpenAI uses stream_options.include_usage; OpenRouter uses usage.include.
			_, hasStreamOptions := req["stream_options"]
			_, hasUsage := req["usage"]
			if !hasStreamOptions && !hasUsage {
				t.Errorf("%s stopped requesting streamed usage: %s", name, body)
			}
		})
	}
}
