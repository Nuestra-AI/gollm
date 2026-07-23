package llm

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/teilomillet/gollm/providers"
	"github.com/teilomillet/gollm/types"
)

// usageBody is a billed 200 carrying OpenAI-shaped counts, used where the test needs usage to be
// recoverable from the raw response rather than from a provider's parsed details.
const usageBody = `{"usage":{"prompt_tokens":7,"completion_tokens":13,"total_tokens":20}}`

func okServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func collectUsage(l *LLMImpl) *[]UsageEvent {
	events := &[]UsageEvent{}
	l.SetUsageObserver(func(_ context.Context, e UsageEvent) { *events = append(*events, e) })
	return events
}

// Generate and GenerateWithSchema discard response details, but their round-trips are billed like
// any other. They are the busiest entrypoints in the library, so if they don't report, most spend
// goes unrecorded.
func TestPlainGenerateEntrypointsReportUsage(t *testing.T) {
	srv := okServer(t, usageBody)
	usage := types.TokenUsage{PromptTokens: 7, CompletionTokens: 13, TotalTokens: 20}
	schema := map[string]interface{}{
		"type":       "object",
		"properties": map[string]interface{}{"text": map[string]interface{}{"type": "string"}},
	}

	t.Run("Generate", func(t *testing.T) {
		l := newUsageStubLLM(&stubUsageProvider{
			endpoint: srv.URL, result: "hi",
			details: &types.ResponseDetails{Model: "m", TokenUsage: usage},
		})
		events := collectUsage(l)

		if _, err := l.Generate(context.Background(), NewPrompt("hi")); err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if len(*events) != 1 || (*events)[0].Outcome != UsageOutcomeSuccess {
			t.Fatalf("events = %+v, want one success", *events)
		}
		if (*events)[0].Usage != usage {
			t.Errorf("usage = %+v, want %+v", (*events)[0].Usage, usage)
		}
	})

	t.Run("GenerateWithSchema", func(t *testing.T) {
		l := newUsageStubLLM(&stubUsageProvider{
			endpoint: srv.URL, result: "{}",
			details: &types.ResponseDetails{Model: "m", TokenUsage: usage},
		})
		events := collectUsage(l)

		if _, err := l.GenerateWithSchema(context.Background(), NewPrompt("hi"), schema); err != nil {
			t.Fatalf("GenerateWithSchema: %v", err)
		}
		if len(*events) != 1 || (*events)[0].Usage != usage {
			t.Fatalf("events = %+v, want one carrying %+v", *events, usage)
		}
	})
}

// A 200 whose content won't parse is still billed — the max_tokens-truncated response with no
// content blocks is the everyday case. The provider parser gives nothing back, so usage has to come
// off the raw body or the tokens vanish while the retry loop pays for another attempt.
func TestParseFailureRecoversUsageFromBody(t *testing.T) {
	srv := okServer(t, usageBody)
	l := newUsageStubLLM(&stubUsageProvider{
		endpoint: srv.URL,
		parseErr: io.ErrUnexpectedEOF, // provider rejects the billed response
	})
	events := collectUsage(l)

	if _, _, err := l.attemptGenerateWithUsage(context.Background(), NewPrompt("hi"), 0); err == nil {
		t.Fatal("expected a parse error")
	}

	if len(*events) != 1 {
		t.Fatalf("expected one usage event, got %d", len(*events))
	}
	e := (*events)[0]
	if e.Outcome != UsageOutcomeParseFail {
		t.Errorf("outcome = %q, want %q", e.Outcome, UsageOutcomeParseFail)
	}
	if want := (types.TokenUsage{PromptTokens: 7, CompletionTokens: 13, TotalTokens: 20}); e.Usage != want {
		t.Errorf("usage = %+v, want %+v recovered from the body", e.Usage, want)
	}
}

// Every attempt in a retry loop is billed separately, so each must arrive with its own attempt index
// — that is what distinguishes a first-try success from tokens burned on a third paid attempt.
func TestRetriedAttemptsReportSeparately(t *testing.T) {
	srv := okServer(t, usageBody)
	l := newUsageStubLLM(&stubUsageProvider{
		endpoint: srv.URL, result: "not-json",
		details: &types.ResponseDetails{Model: "m", TokenUsage: types.TokenUsage{PromptTokens: 7}},
	})
	l.MaxRetries = 2
	events := collectUsage(l)

	schema := map[string]interface{}{
		"type":       "object",
		"properties": map[string]interface{}{"text": map[string]interface{}{"type": "string"}},
		"required":   []interface{}{"text"},
	}
	if _, _, err := l.GenerateWithSchemaAndUsage(context.Background(), NewPrompt("hi"), schema); err == nil {
		t.Fatal("expected schema validation to fail on every attempt")
	}

	if len(*events) != 3 {
		t.Fatalf("got %d usage events, want one per billed attempt (3)", len(*events))
	}
	for i, e := range *events {
		if e.Attempt != i {
			t.Errorf("event %d has Attempt = %d, want %d", i, e.Attempt, i)
		}
		if e.Outcome != UsageOutcomeSchemaFail {
			t.Errorf("event %d outcome = %q, want %q", i, e.Outcome, UsageOutcomeSchemaFail)
		}
	}
}

// The observer runs inline on the generation path. A recorder that panics is a bug in the recorder,
// not a reason to fail the generation it was only measuring.
func TestUsageObserverPanicIsContained(t *testing.T) {
	srv := okServer(t, usageBody)
	l := newUsageStubLLM(&stubUsageProvider{
		endpoint: srv.URL, result: "hi",
		details: &types.ResponseDetails{Model: "m"},
	})
	l.SetUsageObserver(func(context.Context, UsageEvent) { panic("recorder blew up") })

	got, err := l.Generate(context.Background(), NewPrompt("hi"))
	if err != nil {
		t.Fatalf("generation failed because the observer panicked: %v", err)
	}
	if got != "hi" {
		t.Errorf("result = %q, want %q", got, "hi")
	}
}

// A stream reports its accumulated usage when it ends, and reports it once. Reading to EOF is the
// completed case.
func TestStreamReportsUsageAtEOF(t *testing.T) {
	sse := "data: {\"usage\":true}\n\ndata: [DONE]\n\n"
	provider := &scriptedRichProvider{fn: func(chunk []byte) (types.StreamChunk, error) {
		if strings.Contains(string(chunk), "DONE") {
			return types.StreamChunk{}, io.EOF
		}
		// The chunk names the model that actually served the stream, which is not the one
		// the caller configured — a gateway or a moving alias resolves to something else.
		return types.StreamChunk{Kind: "usage", Model: "resolved-model",
			Usage: &types.TokenUsage{PromptTokens: 5, CompletionTokens: 9}}, nil
	}}

	var outcomes []UsageOutcome
	var reported types.TokenUsage
	var reportedModel string
	s := newProviderStream(io.NopCloser(strings.NewReader(sse)), provider,
		&StreamConfig{MaxLineSize: DefaultSSEMaxLineSize},
		func(o UsageOutcome, m string, _ string, u types.TokenUsage) {
			outcomes = append(outcomes, o)
			reported, reportedModel = u, m
		})

	for {
		if _, err := s.Next(context.Background()); err != nil {
			break
		}
	}
	// Closing an already-finished stream must not double-count it.
	_ = s.Close()

	if len(outcomes) != 1 || outcomes[0] != UsageOutcomeStream {
		t.Fatalf("outcomes = %v, want exactly [%s]", outcomes, UsageOutcomeStream)
	}
	if want := (types.TokenUsage{PromptTokens: 5, CompletionTokens: 9, TotalTokens: 14}); reported != want {
		t.Errorf("reported usage = %+v, want %+v", reported, want)
	}
	if s.Usage() != reported {
		t.Errorf("Usage() = %+v, want %+v", s.Usage(), reported)
	}
	// Attributing the tokens to the requested model instead of the served one misprices every
	// gateway-routed stream, so the resolved name has to survive to the report.
	if reportedModel != "resolved-model" {
		t.Errorf("reported model = %q, want the provider-resolved %q", reportedModel, "resolved-model")
	}
}

// A consumer that walks away mid-stream still pays for what was generated, so Close reports what
// was accumulated and flags it as aborted rather than staying silent.
func TestAbandonedStreamReportsAborted(t *testing.T) {
	sse := "data: a\n\ndata: b\n\ndata: [DONE]\n\n"
	provider := &scriptedRichProvider{fn: func(chunk []byte) (types.StreamChunk, error) {
		if strings.Contains(string(chunk), "DONE") {
			return types.StreamChunk{}, io.EOF
		}
		return types.StreamChunk{Kind: "text", Text: "tok", Usage: &types.TokenUsage{PromptTokens: 4}}, nil
	}}

	var outcomes []UsageOutcome
	var reported types.TokenUsage
	s := newProviderStream(io.NopCloser(strings.NewReader(sse)), provider,
		&StreamConfig{MaxLineSize: DefaultSSEMaxLineSize},
		func(o UsageOutcome, _ string, _ string, u types.TokenUsage) {
			outcomes = append(outcomes, o)
			reported = u
		})

	if _, err := s.Next(context.Background()); err != nil { // read one token, then give up
		t.Fatalf("first Next: %v", err)
	}
	_ = s.Close()

	if len(outcomes) != 1 || outcomes[0] != UsageOutcomeStreamAborted {
		t.Fatalf("outcomes = %v, want exactly [%s]", outcomes, UsageOutcomeStreamAborted)
	}
	if reported.PromptTokens != 4 {
		t.Errorf("reported prompt tokens = %d, want the 4 accumulated before the abort", reported.PromptTokens)
	}
}

// ExtractUsage is the last line of defence when a provider parser can't help, so it has to know
// every wire shape the supported providers emit.
func TestExtractUsageAcrossWireShapes(t *testing.T) {
	cases := []struct {
		name string
		body string
		want types.TokenUsage
	}{
		{
			"openai with cached and reasoning breakdowns",
			`{"usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,
			  "prompt_tokens_details":{"cached_tokens":80},
			  "completion_tokens_details":{"reasoning_tokens":40}}}`,
			types.TokenUsage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150, CachedPromptTokens: 80, ReasoningTokens: 40},
		},
		{
			// Anthropic reports no total and bills cache reads and writes on top of the uncached
			// input, so all four components make up the total — the same rule the provider parsers
			// follow, so a cached call totals the same whether its content parsed or not.
			"anthropic with cache counts",
			`{"usage":{"input_tokens":10,"output_tokens":5,"cache_creation_input_tokens":3,"cache_read_input_tokens":7}}`,
			types.TokenUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 25, CacheCreationInputTokens: 3, CacheReadInputTokens: 7},
		},
		{
			"openai responses api",
			`{"usage":{"input_tokens":12,"output_tokens":8,"total_tokens":20,"output_tokens_details":{"reasoning_tokens":6}}}`,
			types.TokenUsage{PromptTokens: 12, CompletionTokens: 8, TotalTokens: 20, ReasoningTokens: 6},
		},
		{
			"cohere v2 nested tokens",
			`{"usage":{"billed_units":{"input_tokens":9},"tokens":{"input_tokens":9,"output_tokens":4}}}`,
			types.TokenUsage{PromptTokens: 9, CompletionTokens: 4, TotalTokens: 13},
		},
		{
			"ollama top-level counts",
			`{"model":"llama3","done":true,"prompt_eval_count":11,"eval_count":6}`,
			types.TokenUsage{PromptTokens: 11, CompletionTokens: 6, TotalTokens: 17},
		},
		{
			"bedrock meta counts",
			`{"generation":"hi","prompt_token_count":21,"generation_token_count":3}`,
			types.TokenUsage{PromptTokens: 21, CompletionTokens: 3, TotalTokens: 24},
		},
		{
			// Ollama's non-streamed body is a JSONL run whose final object holds the totals.
			"jsonl with counts on the final object",
			"{\"response\":\"a\",\"done\":false}\n{\"response\":\"\",\"done\":true,\"prompt_eval_count\":2,\"eval_count\":3}",
			types.TokenUsage{PromptTokens: 2, CompletionTokens: 3, TotalTokens: 5},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := ExtractUsage([]byte(tc.body))
			if !ok {
				t.Fatalf("ExtractUsage found no usage in %s", tc.body)
			}
			if got != tc.want {
				t.Errorf("usage = %+v, want %+v", got, tc.want)
			}
		})
	}

	if _, ok := ExtractUsage([]byte(`{"choices":[{"text":"no usage here"}]}`)); ok {
		t.Error("ExtractUsage reported usage for a body that carries none")
	}
}

// nonObservableLLM is an LLM that predates the usage observer — a downstream mock or fake. It is
// what makes the wrapper's honesty testable: wrapping it must not manufacture a capability.
type nonObservableLLM struct{ LLM }

// A wrapper satisfies UsageObservable whether or not the value it wraps does, so AttachUsageObserver
// would report success for a client that will never report a token. Callers use that answer to
// suppress their own accounting, so a false positive here means nothing gets recorded at all —
// exactly the silent under-billing the observer exists to end.
func TestAttachUsageObserverReportsWrappedCapabilityHonestly(t *testing.T) {
	observer := func(context.Context, UsageEvent) {}

	wrapped := &LLMWithMemory{LLM: &nonObservableLLM{}}
	if AttachUsageObserver(wrapped, observer) {
		t.Error("reported success wrapping a client that cannot observe: a caller trusting this records nothing")
	}

	// The same wrapper over a capable client must still report success, or the check is vacuous.
	if !AttachUsageObserver(&LLMWithMemory{LLM: &LLMImpl{}}, observer) {
		t.Error("reported failure wrapping a client that can observe")
	}
}

// A round-trip must total the same however its usage was read. The provider parser and the raw-body
// recovery are separate implementations reached by separate paths — success versus parse failure —
// and a cached Anthropic call is where they would diverge, since cache tokens are billed on top of
// PromptTokens and it is tempting to fold them into the total on one path only.
func TestTotalTokensAgreeBetweenParserAndRecovery(t *testing.T) {
	body := []byte(`{"id":"msg_1","model":"claude-sonnet-4-5","stop_reason":"end_turn",
	  "content":[{"type":"text","text":"hi"}],
	  "usage":{"input_tokens":10,"output_tokens":5,"cache_creation_input_tokens":3,"cache_read_input_tokens":7}}`)

	_, details, err := providers.NewAnthropicProvider("key", "claude-sonnet-4-5", nil).ParseResponseWithUsage(body)
	if err != nil {
		t.Fatalf("ParseResponseWithUsage: %v", err)
	}
	recovered, ok := ExtractUsage(body)
	if !ok {
		t.Fatal("ExtractUsage found no usage in a body that plainly carries it")
	}

	if details.TokenUsage.TotalTokens != recovered.TotalTokens {
		t.Errorf("TotalTokens depends on which path read it: parser %d, recovery %d",
			details.TokenUsage.TotalTokens, recovered.TotalTokens)
	}
	// Pinned rather than merely equal: the total covers every billed token, cache included.
	// Dropping the cache counts here would under-report this call by two thirds.
	if want := 25; recovered.TotalTokens != want {
		t.Errorf("TotalTokens = %d, want %d (uncached input + cache write + cache read + output)", recovered.TotalTokens, want)
	}
	if recovered.CacheCreationInputTokens != 3 || recovered.CacheReadInputTokens != 7 {
		t.Errorf("cache counts lost: %+v", recovered)
	}
}

// Cancellation is how a stream most often ends early in production — a request timeout, a
// disconnected client, a dying parent context. The provider generated and billed whatever arrived
// before it, so Next must report on the way out rather than leaving it to a Close the caller is
// under no obligation to make.
func TestCancelledStreamReportsAborted(t *testing.T) {
	sse := "data: a\n\ndata: b\n\ndata: [DONE]\n\n"
	provider := &scriptedRichProvider{fn: func(chunk []byte) (types.StreamChunk, error) {
		if strings.Contains(string(chunk), "DONE") {
			return types.StreamChunk{}, io.EOF
		}
		return types.StreamChunk{Kind: "text", Text: "tok", Usage: &types.TokenUsage{PromptTokens: 6}}, nil
	}}

	var outcomes []UsageOutcome
	var reported types.TokenUsage
	s := newProviderStream(io.NopCloser(strings.NewReader(sse)), provider,
		&StreamConfig{MaxLineSize: DefaultSSEMaxLineSize},
		func(o UsageOutcome, _ string, _ string, u types.TokenUsage) {
			outcomes = append(outcomes, o)
			reported = u
		})

	ctx, cancel := context.WithCancel(context.Background())
	if _, err := s.Next(ctx); err != nil {
		t.Fatalf("first Next: %v", err)
	}
	cancel()
	if _, err := s.Next(ctx); err != context.Canceled {
		t.Fatalf("Next after cancel: %v, want context.Canceled", err)
	}

	if len(outcomes) != 1 || outcomes[0] != UsageOutcomeStreamAborted {
		t.Fatalf("outcomes = %v, want exactly [%s]", outcomes, UsageOutcomeStreamAborted)
	}
	if reported.PromptTokens != 6 {
		t.Errorf("reported prompt tokens = %d, want the 6 billed before cancellation", reported.PromptTokens)
	}
	// A caller who also Closes, as callers are told to, must not be billed twice for it.
	_ = s.Close()
	if len(outcomes) != 1 {
		t.Errorf("Close after cancellation reported again: outcomes = %v", outcomes)
	}
}

// The raw-body fallback is what covers a billed 200 whose content wouldn't parse. It has to know
// every breakdown the providers emit, or the recovered record silently under-prices the call it
// exists to rescue.
func TestExtractUsageRecoversEveryBilledBreakdown(t *testing.T) {
	t.Run("openai", func(t *testing.T) {
		body := []byte(`{"usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,
		  "prompt_tokens_details":{"cached_tokens":64,"cache_write_tokens":12,"audio_tokens":8},
		  "completion_tokens_details":{"reasoning_tokens":32,"audio_tokens":4,
		                               "accepted_prediction_tokens":18,"rejected_prediction_tokens":10}}}`)
		got, ok := ExtractUsage(body)
		if !ok {
			t.Fatal("no usage recovered")
		}
		want := types.TokenUsage{
			PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150,
			CachedPromptTokens: 64, ReasoningTokens: 32,
			CacheWritePromptTokens: 12, AudioPromptTokens: 8, AudioCompletionTokens: 4,
			AcceptedPredictionTokens: 18, RejectedPredictionTokens: 10,
		}
		if got != want {
			t.Errorf("usage = %+v,\n want %+v", got, want)
		}
	})

	t.Run("anthropic", func(t *testing.T) {
		// thinking_tokens is Anthropic's name for reasoning output, and the cache_creation split
		// is the only way to tell a 1.25x write from a 2x one.
		body := []byte(`{"usage":{"input_tokens":10,"output_tokens":50,"cache_read_input_tokens":7,
		  "cache_creation":{"ephemeral_5m_input_tokens":20,"ephemeral_1h_input_tokens":10},
		  "output_tokens_details":{"thinking_tokens":42}}}`)
		got, ok := ExtractUsage(body)
		if !ok {
			t.Fatal("no usage recovered")
		}
		want := types.TokenUsage{
			PromptTokens: 10, CompletionTokens: 50, TotalTokens: 97,
			CacheCreationInputTokens: 30, CacheReadInputTokens: 7,
			CacheCreation5mInputTokens: 20, CacheCreation1hInputTokens: 10,
			ReasoningTokens: 42,
		}
		if got != want {
			t.Errorf("usage = %+v,\n want %+v", got, want)
		}
	})
}

// The tier is recovered separately from the counts because a parse failure yields no details at
// all, and it scales the price of whatever those counts turn out to be.
func TestExtractServiceTierFromBothShapes(t *testing.T) {
	cases := map[string]string{
		`{"id":"r1","service_tier":"priority","usage":{"prompt_tokens":1}}`:    "priority", // OpenAI: top level
		`{"id":"msg_1","usage":{"input_tokens":1,"service_tier":"batch"}}`:     "batch",    // Anthropic: inside usage
		`{"id":"r1","usage":{"prompt_tokens":1}}`:                              "",         // absent
		"{\"a\":1}\n{\"usage\":{\"service_tier\":\"standard\"},\"done\":true}": "standard", // JSONL terminal object
	}
	for body, want := range cases {
		if got := ExtractServiceTier([]byte(body)); got != want {
			t.Errorf("ExtractServiceTier(%s) = %q, want %q", body, got, want)
		}
	}
}

// A streamed call is billed on the tier the provider served it on just like a blocking one, so the
// tier has to survive the accumulator and reach the observer.
func TestStreamReportsServiceTier(t *testing.T) {
	sse := "data: a\n\ndata: [DONE]\n\n"
	provider := &scriptedRichProvider{fn: func(chunk []byte) (types.StreamChunk, error) {
		if strings.Contains(string(chunk), "DONE") {
			return types.StreamChunk{}, io.EOF
		}
		return types.StreamChunk{Kind: "usage", ServiceTier: "batch",
			Usage: &types.TokenUsage{PromptTokens: 3, CompletionTokens: 4}}, nil
	}}

	var gotTier string
	s := newProviderStream(io.NopCloser(strings.NewReader(sse)), provider,
		&StreamConfig{MaxLineSize: DefaultSSEMaxLineSize},
		func(_ UsageOutcome, _ string, tier string, _ types.TokenUsage) { gotTier = tier })

	for {
		if _, err := s.Next(context.Background()); err != nil {
			break
		}
	}
	if gotTier != "batch" {
		t.Errorf("reported tier = %q, want %q: a batch-tier stream prices at roughly half", gotTier, "batch")
	}
	if s.ServiceTier() != "batch" {
		t.Errorf("ServiceTier() = %q, want %q", s.ServiceTier(), "batch")
	}
}

// The tier reaches the observer on an ordinary successful call, which is the path a recorder
// actually writes its tier column from. It comes from the provider's parsed details here, not from
// the body — see TestSuccessfulCallDoesNotReparseBody for why that distinction matters.
func TestNonStreamEventCarriesServiceTier(t *testing.T) {
	srv := okServer(t, usageBody)
	l := newUsageStubLLM(&stubUsageProvider{
		endpoint: srv.URL, result: "hi",
		details: &types.ResponseDetails{
			Model:       "gpt-5",
			TokenUsage:  types.TokenUsage{PromptTokens: 7, CompletionTokens: 13, TotalTokens: 20},
			ServiceTier: "batch",
		},
	})
	events := collectUsage(l)
	if _, _, err := l.GenerateWithUsage(context.Background(), NewPrompt("hi")); err != nil {
		t.Fatalf("GenerateWithUsage: %v", err)
	}
	if len(*events) != 1 {
		t.Fatalf("got %d events, want 1", len(*events))
	}
	if got := (*events)[0].ServiceTier; got != "batch" {
		t.Errorf("ServiceTier = %q, want %q — batch prices at roughly half rate", got, "batch")
	}
}

// A parse failure has no details to read the tier from, so it is recovered from the raw body
// alongside the counts — in the same pass, not a second one.
func TestParseFailureRecoversServiceTierWithUsage(t *testing.T) {
	// A billed 200 carrying usage and a tier, whose content the provider cannot parse.
	srv := okServer(t, `{"service_tier":"priority","usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13}}`)
	l := newUsageStubLLM(&stubUsageProvider{endpoint: srv.URL, parseErr: errors.New("no content")})
	events := collectUsage(l)
	_, _, _ = l.GenerateWithUsage(context.Background(), NewPrompt("hi"))

	if len(*events) == 0 {
		t.Fatal("no usage event for a billed unparseable response")
	}
	ev := (*events)[len(*events)-1]
	if ev.Usage.PromptTokens != 9 || ev.ServiceTier != "priority" {
		t.Errorf("recovered %+v tier=%q; want 9 prompt tokens at the priority tier", ev.Usage, ev.ServiceTier)
	}
}

// ExtractUsageAndTier reads both in one pass. Reading them separately meant a successful call
// re-decoded its whole response body looking for a tier that most providers never send — a second
// full parse of every response, on the generation path, for nothing.
func TestExtractUsageAndTierIsOnePass(t *testing.T) {
	body := []byte(`{"service_tier":"flex","usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}`)
	usage, tier, ok := ExtractUsageAndTier(body)
	if !ok || usage.PromptTokens != 3 || tier != "flex" {
		t.Errorf("ExtractUsageAndTier = %+v, %q, %v; want the counts and the tier together", usage, tier, ok)
	}

	// A body with a tier but no counts still yields the tier, since the tier is what prices
	// whatever those counts turn out to be.
	if _, tier, ok := ExtractUsageAndTier([]byte(`{"service_tier":"batch"}`)); ok || tier != "batch" {
		t.Errorf("tier-only body gave tier=%q ok=%v; want batch,false", tier, ok)
	}
}
