package llm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

// stubUsageProvider returns a fixed parsed result and usage regardless of the request, so the
// schema-validation branch — and therefore the usage observer — can be exercised without a real API.
// A nil details forces the parse-failure branch, exercising the raw-body usage recovery.
type stubUsageProvider struct {
	endpoint string
	result   string
	details  *types.ResponseDetails
	parseErr error
}

func (p *stubUsageProvider) Name() string                      { return "stub" }
func (p *stubUsageProvider) Endpoint() string                  { return p.endpoint }
func (p *stubUsageProvider) Headers() map[string]string        { return map[string]string{} }
func (p *stubUsageProvider) SupportsStreaming() bool           { return false }
func (p *stubUsageProvider) SupportsJSONSchema() bool          { return true }
func (p *stubUsageProvider) SetExtraHeaders(map[string]string) {}
func (p *stubUsageProvider) SetDefaultOptions(*config.Config)  {}
func (p *stubUsageProvider) SetOption(string, interface{})     {}
func (p *stubUsageProvider) SetLogger(utils.Logger)            {}
func (p *stubUsageProvider) PrepareRequest(string, map[string]interface{}) ([]byte, error) {
	return []byte("{}"), nil
}
func (p *stubUsageProvider) PrepareRequestWithSchema(string, map[string]interface{}, interface{}) ([]byte, error) {
	return []byte("{}"), nil
}
func (p *stubUsageProvider) PrepareRequestWithMessages([]types.MemoryMessage, map[string]interface{}) ([]byte, error) {
	return []byte("{}"), nil
}
func (p *stubUsageProvider) PrepareRequestWithMessagesAndSchema([]types.MemoryMessage, map[string]interface{}, interface{}) ([]byte, error) {
	return []byte("{}"), nil
}
func (p *stubUsageProvider) ParseResponse([]byte) (string, error) { return p.result, nil }
func (p *stubUsageProvider) ParseResponseWithUsage([]byte) (string, *types.ResponseDetails, error) {
	return p.result, p.details, p.parseErr
}
func (p *stubUsageProvider) HandleFunctionCalls([]byte) ([]byte, error) { return nil, nil }
func (p *stubUsageProvider) PrepareStreamRequest(string, map[string]interface{}) ([]byte, error) {
	return []byte("{}"), nil
}
func (p *stubUsageProvider) ParseStreamResponse([]byte) (string, error) { return "", nil }

func newUsageStubLLM(p *stubUsageProvider) *LLMImpl {
	return &LLMImpl{
		Provider:   p,
		client:     &http.Client{},
		logger:     utils.NewLogger(utils.LogLevelOff),
		Options:    make(map[string]interface{}),
		MaxRetries: 0,
		RetryDelay: time.Millisecond,
	}
}

// Every billed round-trip must reach the observer with the outcome that describes it — including the
// attempt that was billed but discarded on a schema or parse failure, which the retry loop would
// otherwise throw away. That discarded-attempt case is the token leak the observer exists to close.
func TestUsageObserverFiresPerOutcome(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		// A usage body so the parse-failure path can still recover counts from the raw response.
		_, _ = w.Write([]byte(`{"usage":{"prompt_tokens":7,"completion_tokens":13,"total_tokens":20}}`))
	}))
	defer srv.Close()

	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"text": map[string]interface{}{"type": "string"},
		},
		"required": []interface{}{"text"},
	}
	usage := types.TokenUsage{PromptTokens: 7, CompletionTokens: 13, TotalTokens: 20}

	cases := []struct {
		name        string
		result      string                 // what the provider "parsed" from the billed 200
		details     *types.ResponseDetails // nil forces the parse-failure branch
		wantOutcome UsageOutcome
	}{
		{"validated response reports success", `{"text":"ok"}`, &types.ResponseDetails{Model: "m", TokenUsage: usage}, UsageOutcomeSuccess},
		{"billed-but-rejected response reports schema_fail", `not-json`, &types.ResponseDetails{Model: "m", TokenUsage: usage}, UsageOutcomeSchemaFail},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			l := newUsageStubLLM(&stubUsageProvider{endpoint: srv.URL, result: tc.result, details: tc.details})

			var events []UsageEvent
			l.SetUsageObserver(func(_ context.Context, e UsageEvent) { events = append(events, e) })

			_, _, _, _ = l.attemptGenerateWithSchemaAndUsage(context.Background(), NewPrompt("hi"), schema, 0)

			if len(events) != 1 {
				t.Fatalf("expected exactly one usage event, got %d", len(events))
			}
			if events[0].Outcome != tc.wantOutcome {
				t.Errorf("outcome = %q, want %q", events[0].Outcome, tc.wantOutcome)
			}
			if events[0].Usage != usage {
				t.Errorf("usage = %+v, want %+v", events[0].Usage, usage)
			}
		})
	}
}
