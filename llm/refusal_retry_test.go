package llm

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/providers"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

const refusalBody = `{"id":"1","model":"gpt-4o","choices":[{"message":
	{"content":"","refusal":"I can't help with that."},"finish_reason":"stop"}],
	"usage":{"prompt_tokens":1,"completion_tokens":0}}`

// TestRefusalReachesCallerAndIsNotRetried covers the whole path, not just the
// parser. A refusal must arrive with its reason intact, and must not be retried:
// the model will decline the same prompt again, so the retry budget would turn one
// refusal into several billed calls and then report only the attempt count.
func TestRefusalReachesCallerAndIsNotRetried(t *testing.T) {
	var calls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(refusalBody))
	}))
	defer server.Close()

	registry := providers.NewProviderRegistry()
	registry.Register("openai", func(apiKey, model string, extraHeaders map[string]string) providers.Provider {
		return &endpointOverride{
			Provider: providers.NewOpenAIProvider(apiKey, model, extraHeaders),
			endpoint: server.URL,
		}
	})

	cfg := &config.Config{
		Provider: "openai", Model: "gpt-4o", MaxRetries: 3,
		RetryDelay: time.Millisecond, Timeout: 5 * time.Second,
		APIKeys: map[string]string{"openai": "sk-test-key-that-is-long-enough"},
	}
	client, err := NewLLM(cfg, utils.NewLogger(utils.LogLevelWarn), registry)
	if err != nil {
		t.Fatalf("NewLLM returned error: %v", err)
	}

	_, err = client.Generate(context.Background(), NewPrompt("hi"))
	if err == nil {
		t.Fatal("a refusal must surface as an error")
	}
	if !errors.Is(err, types.ErrRefusal) {
		t.Errorf("error does not identify as a refusal: %v", err)
	}
	if !strings.Contains(err.Error(), "I can't help with that.") {
		t.Errorf("the refusal reason did not reach the caller: %v", err)
	}
	if n := atomic.LoadInt32(&calls); n != 1 {
		t.Errorf("made %d calls; a refusal is deterministic and must not be retried", n)
	}
}

// endpointOverride points a real provider at a test server while leaving its
// request building and parsing intact.
type endpointOverride struct {
	providers.Provider
	endpoint string
}

func (e *endpointOverride) Endpoint() string { return e.endpoint }
