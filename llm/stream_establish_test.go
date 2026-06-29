package llm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

// stubStreamProvider is a minimal Provider whose Endpoint points at a test
// server, used to exercise LLMImpl.Stream's establishment-retry loop. Only the
// streaming methods carry behavior; the rest are inert stubs.
type stubStreamProvider struct {
	endpoint string
}

func (p *stubStreamProvider) Name() string                      { return "stub" }
func (p *stubStreamProvider) Endpoint() string                  { return p.endpoint }
func (p *stubStreamProvider) Headers() map[string]string        { return map[string]string{} }
func (p *stubStreamProvider) SupportsStreaming() bool           { return true }
func (p *stubStreamProvider) SupportsJSONSchema() bool          { return false }
func (p *stubStreamProvider) SetExtraHeaders(map[string]string) {}
func (p *stubStreamProvider) SetDefaultOptions(*config.Config)  {}
func (p *stubStreamProvider) SetOption(string, interface{})     {}
func (p *stubStreamProvider) SetLogger(utils.Logger)            {}

func (p *stubStreamProvider) PrepareRequest(string, map[string]interface{}) ([]byte, error) {
	return []byte("{}"), nil
}
func (p *stubStreamProvider) PrepareRequestWithSchema(string, map[string]interface{}, interface{}) ([]byte, error) {
	return []byte("{}"), nil
}
func (p *stubStreamProvider) PrepareRequestWithMessages([]types.MemoryMessage, map[string]interface{}) ([]byte, error) {
	return []byte("{}"), nil
}
func (p *stubStreamProvider) PrepareRequestWithMessagesAndSchema([]types.MemoryMessage, map[string]interface{}, interface{}) ([]byte, error) {
	return []byte("{}"), nil
}
func (p *stubStreamProvider) ParseResponse([]byte) (string, error) { return "", nil }
func (p *stubStreamProvider) ParseResponseWithUsage([]byte) (string, *types.ResponseDetails, error) {
	return "", nil, nil
}
func (p *stubStreamProvider) HandleFunctionCalls([]byte) ([]byte, error) { return nil, nil }

func (p *stubStreamProvider) PrepareStreamRequest(string, map[string]interface{}) ([]byte, error) {
	return []byte("{}"), nil
}
func (p *stubStreamProvider) ParseStreamResponse(chunk []byte) (string, error) {
	return string(chunk), nil
}

func newStubLLM(provider *stubStreamProvider) *LLMImpl {
	return &LLMImpl{
		Provider:   provider,
		client:     &http.Client{},
		logger:     utils.NewLogger(utils.LogLevelOff),
		Options:    make(map[string]interface{}),
		MaxRetries: 3,
		RetryDelay: time.Millisecond,
	}
}

// TestStreamEstablishmentRetry verifies that a transient non-200 during stream
// establishment is retried with backoff and ultimately succeeds.
func TestStreamEstablishmentRetry(t *testing.T) {
	var hits int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&hits, 1) < 3 {
			w.WriteHeader(http.StatusServiceUnavailable) // transient failure
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: hello\n\n"))
	}))
	defer srv.Close()

	llm := newStubLLM(&stubStreamProvider{endpoint: srv.URL})

	stream, err := llm.Stream(context.Background(), &Prompt{Input: "hi"})
	if err != nil {
		t.Fatalf("expected establishment to succeed after retries, got %v", err)
	}
	defer stream.Close()

	if got := atomic.LoadInt32(&hits); got != 3 {
		t.Fatalf("expected 3 establishment attempts (2 failures + 1 success), got %d", got)
	}

	tok, err := stream.Next(context.Background())
	if err != nil {
		t.Fatalf("Next: %v", err)
	}
	if strings.TrimSpace(tok.Text) != "hello" {
		t.Fatalf("expected token text %q, got %q", "hello", tok.Text)
	}
}

// TestStreamEstablishmentNoRetryOn401 verifies non-transient statuses (auth /
// validation 4xx) fail fast instead of burning the retry budget.
func TestStreamEstablishmentNoRetryOn401(t *testing.T) {
	var hits int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	llm := newStubLLM(&stubStreamProvider{endpoint: srv.URL})

	if _, err := llm.Stream(context.Background(), &Prompt{Input: "hi"}); err == nil {
		t.Fatal("expected error on 401")
	}
	if got := atomic.LoadInt32(&hits); got != 1 {
		t.Fatalf("expected 1 attempt (no retry on 401), got %d", got)
	}
}

// TestStreamEstablishmentRetryExhausted verifies that persistent failures stop
// after MaxRetries+1 attempts and return an error rather than looping forever.
func TestStreamEstablishmentRetryExhausted(t *testing.T) {
	var hits int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	llm := newStubLLM(&stubStreamProvider{endpoint: srv.URL})

	_, err := llm.Stream(context.Background(), &Prompt{Input: "hi"})
	if err == nil {
		t.Fatal("expected error after exhausting retries")
	}
	// 1 initial attempt + MaxRetries(3) = 4 total.
	if got := atomic.LoadInt32(&hits); got != 4 {
		t.Fatalf("expected 4 attempts (1 + 3 retries), got %d", got)
	}
}
