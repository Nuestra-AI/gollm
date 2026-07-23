package gollm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// recordingTransport counts the provider requests that pass through it.
type recordingTransport struct {
	base     http.RoundTripper
	requests atomic.Int64
	sawAuth  atomic.Bool
}

func (t *recordingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	t.requests.Add(1)
	if req.Header.Get("Authorization") != "" {
		t.sawAuth.Store(true)
	}
	return t.base.RoundTrip(req)
}

// A caller-supplied HTTP client must actually carry provider traffic. This is the seam for
// transport-level concerns the provider layer cannot reach — most concretely, usage that arrives in
// response headers rather than the body, which is how Bedrock reports token counts.
func TestSetHTTPClientInterceptsProviderRequests(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"1","model":"m","choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],
		                        "usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}`))
	}))
	defer srv.Close()

	transport := &recordingTransport{base: http.DefaultTransport}

	var observed atomic.Int64
	client, err := NewLLM(
		SetProvider("vllm"),
		SetModel("qwen"),
		SetVLLMEndpoint(srv.URL),
		SetMaxRetries(0),
		SetHTTPClient(&http.Client{Transport: transport}),
		WithUsageObserver(func(context.Context, UsageEvent) { observed.Add(1) }),
	)
	if err != nil {
		t.Fatalf("NewLLM: %v", err)
	}

	if _, err := client.Generate(context.Background(), NewPrompt("hi")); err != nil {
		t.Fatalf("Generate: %v", err)
	}

	if got := transport.requests.Load(); got != 1 {
		t.Errorf("transport saw %d requests, want 1: a custom RoundTripper is not carrying provider traffic", got)
	}
	// Both layers observe the same call — the transport seam is additive to the observer, not a
	// replacement for it.
	if got := observed.Load(); got != 1 {
		t.Errorf("usage observer fired %d times, want 1", got)
	}
}
