package providers

import (
	"encoding/json"
	"testing"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/types"
)

// TestIsResponsesOnlyModel pins the model families that Chat Completions cannot
// serve. Each "true" case was verified against the supported-endpoints table on
// developers.openai.com (see openai_routing.go for the verification date).
func TestIsResponsesOnlyModel(t *testing.T) {
	tests := []struct {
		name  string
		model string
		want  bool
	}{
		// Codex — Responses-only across every generation shipped so far.
		{"gpt-5-codex", "gpt-5-codex", true},
		{"gpt-5.1-codex", "gpt-5.1-codex", true},
		{"gpt-5.1-codex-mini", "gpt-5.1-codex-mini", true},
		{"gpt-5.2-codex", "gpt-5.2-codex", true},
		{"gpt-5.3-codex", "gpt-5.3-codex", true},
		{"legacy codex-mini-latest", "codex-mini-latest", true},

		// Pro reasoning models, bare and dated snapshots.
		{"gpt-5-pro", "gpt-5-pro", true},
		{"gpt-5-pro snapshot", "gpt-5-pro-2025-10-06", true},
		{"o3-pro", "o3-pro", true},
		{"o3-pro snapshot", "o3-pro-2025-06-10", true},
		{"o1-pro", "o1-pro", true},

		// Deep research.
		{"o3-deep-research", "o3-deep-research", true},
		{"o4-mini-deep-research", "o4-mini-deep-research", true},

		// Computer use.
		{"computer-use-preview", "computer-use-preview", true},
		{"computer-use-preview snapshot", "computer-use-preview-2025-03-11", true},

		// Models served by BOTH endpoints must not route here: rerouting them
		// would change response shape and usage fields for callers that work
		// today. The GPT-5.6 line is served by both but still routes — via
		// rejectsToolsOnChatCompletions, not this predicate.
		{"gpt-5.6-sol", "gpt-5.6-sol", false},
		{"gpt-5.6-terra", "gpt-5.6-terra", false},
		{"gpt-5.6-luna", "gpt-5.6-luna", false},
		{"gpt-5", "gpt-5", false},
		{"gpt-5-mini", "gpt-5-mini", false},
		{"gpt-5-chat-latest", "gpt-5-chat-latest", false},
		{"gpt-4o", "gpt-4o", false},
		{"gpt-4.1", "gpt-4.1", false},
		{"o3", "o3", false},
		{"o4-mini", "o4-mini", false},

		// Unrecognized ids stay on Chat Completions — the fail-safe direction.
		{"empty", "", false},
		{"unknown", "some-future-model", false},

		// Segment matching, not substring: these must not trip the "pro" or
		// "codex" families.
		{"prometheus is not pro", "prometheus-1", false},
		{"probe is not pro", "probe-mini", false},
		{"codexial is not codex", "codexial-2", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isResponsesOnlyModel(tt.model); got != tt.want {
				t.Errorf("isResponsesOnlyModel(%q) = %v, want %v", tt.model, got, tt.want)
			}
		})
	}
}

// TestRejectsToolsOnChatCompletions pins the GPT-5.6 boundary. These models are
// listed as Chat Completions "Supported" in OpenAI's endpoint table, but reject
// any request carrying function tools; see openai_routing.go for the 400 text and
// the reproduction sources.
func TestRejectsToolsOnChatCompletions(t *testing.T) {
	tests := []struct {
		name  string
		model string
		want  bool
	}{
		// The frontier line reasons by default, so tools fail even with no
		// reasoning_effort in the request.
		{"gpt-5.6-sol", "gpt-5.6-sol", true},
		{"gpt-5.6-terra", "gpt-5.6-terra", true},
		{"gpt-5.6-luna", "gpt-5.6-luna", true},
		{"gpt-5.6 bare", "gpt-5.6", true},

		// Expected to inherit the behavior; the rule is a bound, not a list.
		{"future gpt-5.7", "gpt-5.7", true},
		{"future gpt-5.10", "gpt-5.10-something", true},

		// gpt-5.4 is where the restriction was introduced; 5.4 and 5.5 fail on
		// explicit reasoning_effort plus tools and route for the same reason.
		{"gpt-5.5", "gpt-5.5", true},
		{"gpt-5.4", "gpt-5.4", true},
		{"gpt-5.4-mini", "gpt-5.4-mini", true},
		{"gpt-5.4-pro", "gpt-5.4-pro", true},

		// Below the bound: unconfirmed, so left on Chat Completions.
		{"gpt-5.3", "gpt-5.3", false},
		{"gpt-5.2", "gpt-5.2", false},
		{"gpt-5.1", "gpt-5.1", false},
		{"gpt-5", "gpt-5", false},
		{"gpt-5-mini", "gpt-5-mini", false},

		// Non-reasoning chat variants have no reasoning to conflict with tools.
		{"gpt-5.6-chat", "gpt-5-chat-latest", false},

		// Not GPT-5 at all.
		{"o3", "o3", false},
		{"o4-mini", "o4-mini", false},
		{"gpt-4o", "gpt-4o", false},
		{"empty", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := rejectsToolsOnChatCompletions(tt.model); got != tt.want {
				t.Errorf("rejectsToolsOnChatCompletions(%q) = %v, want %v",
					tt.model, got, tt.want)
			}
		})
	}
}

// TestRouteOpenAIProvider covers the provider-name gate, which is what keeps the
// model patterns from reaching providers that merely share OpenAI's model naming.
func TestRouteOpenAIProvider(t *testing.T) {
	tests := []struct {
		name     string
		provider string
		model    string
		want     string
	}{
		{"routes responses-only model", "openai", "o3-pro", "openai-responses"},
		{"routes codex", "openai", "gpt-5.3-codex", "openai-responses"},
		{"routes gpt-5.6 frontier line", "openai", "gpt-5.6-sol", "openai-responses"},
		{"routes gpt-5.6-terra", "openai", "gpt-5.6-terra", "openai-responses"},
		{"routes gpt-5.6-luna", "openai", "gpt-5.6-luna", "openai-responses"},
		{"routes gpt-5.5", "openai", "gpt-5.5", "openai-responses"},
		{"routes gpt-5.4", "openai", "gpt-5.4", "openai-responses"},
		{"leaves gpt-5.3 alone", "openai", "gpt-5.3", "openai"},
		{"leaves gpt-4o alone", "openai", "gpt-4o", "openai"},

		// Explicit choices are never rewritten, in either direction.
		{"explicit responses stays", "openai-responses", "gpt-4o", "openai-responses"},
		{"explicit chat pin stays", "openai-chat", "o3-pro", "openai-chat"},
		{"explicit chat pin stays for frontier", "openai-chat", "gpt-5.6-sol", "openai-chat"},

		// OpenAI-compatible providers have different (or absent) Responses
		// support and must never be rerouted, even on a matching model id.
		{"azure not routed", "azure-openai", "o3-pro", "azure-openai"},
		{"openrouter not routed", "openrouter", "gpt-5-codex", "openrouter"},
		{"groq not routed", "groq", "o3-pro", "groq"},
		{"lambda not routed", "lambda", "gpt-5-pro", "lambda"},
		{"vllm not routed", "vllm", "gpt-5-codex", "vllm"},
		{"lmstudio not routed", "lmstudio", "o3-pro", "lmstudio"},
		{"aliyun not routed", "aliyun", "o3-pro", "aliyun"},

		// google-openai carries Gemini ids through an OpenAI-shaped surface;
		// "gemini-2.5-pro" would match the pro family if the name gate were absent.
		{"google-openai not routed", "google-openai", "gemini-2.5-pro", "google-openai"},

		// Non-OpenAI providers are untouched.
		{"anthropic not routed", "anthropic", "claude-opus-4-5", "anthropic"},
		{"ollama not routed", "ollama", "llama3", "ollama"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := routeOpenAIProvider(tt.provider, tt.model); got != tt.want {
				t.Errorf("routeOpenAIProvider(%q, %q) = %q, want %q",
					tt.provider, tt.model, got, tt.want)
			}
		})
	}
}

// TestRegistryGetRoutes verifies the seam end to end: the substitution happens
// inside ProviderRegistry.Get, the single choke point both gollm.NewLLM and
// llm.NewLLM resolve providers through.
func TestRegistryGetRoutes(t *testing.T) {
	registry := NewProviderRegistry()

	tests := []struct {
		name     string
		provider string
		model    string
		wantName string
	}{
		{"responses-only model gets responses provider", "openai", "gpt-5-pro", "openai-responses"},
		{"ordinary model keeps chat provider", "openai", "gpt-4o", "openai"},
		{"frontier model gets responses provider", "openai", "gpt-5.6-sol", "openai-responses"},
		{"gpt-5.5 gets responses provider", "openai", "gpt-5.5", "openai-responses"},
		{"gpt-5.3 keeps chat provider", "openai", "gpt-5.3", "openai"},
		{"chat pin resists routing", "openai-chat", "gpt-5-pro", "openai"},
		{"explicit responses honored", "openai-responses", "gpt-4o", "openai-responses"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := registry.Get(tt.provider, "sk-test-key-that-is-long-enough", tt.model, nil)
			if err != nil {
				t.Fatalf("registry.Get(%q, ..., %q) returned error: %v", tt.provider, tt.model, err)
			}
			if got := p.Name(); got != tt.wantName {
				t.Errorf("provider.Name() = %q, want %q", got, tt.wantName)
			}
			if got := p.Endpoint(); got == "" {
				t.Errorf("provider.Endpoint() is empty for %q", tt.provider)
			}
		})
	}
}

// TestRegistryGetUnknownProviderStillErrors confirms routing did not swallow the
// unknown-provider error path.
func TestRegistryGetUnknownProviderStillErrors(t *testing.T) {
	registry := NewProviderRegistry()
	if _, err := registry.Get("no-such-provider", "key", "gpt-4o", nil); err == nil {
		t.Fatal("expected an error for an unknown provider, got nil")
	}
}

// TestResponsesRequestOmitsChatOnlyParams guards the pass-through denylist in
// mergeOpenAIResponsesOptions. The Responses API has no seed or stop parameter
// and rejects the whole request with "Unknown parameter", so neither may reach
// the body — whether it arrives as a provider default or a per-request option.
// Routing makes this reachable for callers who never chose Responses themselves.
func TestResponsesRequestOmitsChatOnlyParams(t *testing.T) {
	seed := 42
	cfg := &config.Config{Temperature: 0.7, MaxTokens: 256, Seed: &seed}

	p := NewOpenAIResponsesProvider("sk-test", "gpt-5.6-sol", nil)
	p.SetDefaultOptions(cfg)

	body, err := p.PrepareRequest("hello", map[string]interface{}{
		"stop": []string{"\n\n"},
		"seed": 7,
	})
	if err != nil {
		t.Fatalf("PrepareRequest returned error: %v", err)
	}

	var request map[string]interface{}
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatalf("request body is not valid JSON: %v", err)
	}

	for _, key := range []string{"seed", "stop"} {
		if v, present := request[key]; present {
			t.Errorf("request body carries chat-only parameter %q = %v; "+
				"the Responses API rejects it with 400 Unknown parameter", key, v)
		}
	}

	// The request must still be well formed after the strip.
	if request["model"] != "gpt-5.6-sol" {
		t.Errorf("model = %v, want gpt-5.6-sol", request["model"])
	}
	if _, ok := request["input"]; !ok {
		t.Error("request body is missing the input field")
	}
}

// TestRoutingNeverBreaksResolvableProvider guards the registry seam: routing must
// not turn a provider the caller could resolve into an unknown-provider error.
// A subset registry has no "openai-responses" to route to.
func TestRoutingNeverBreaksResolvableProvider(t *testing.T) {
	registry := NewProviderRegistry("openai")

	for _, model := range []string{"gpt-5.4-mini", "gpt-5.6-sol", "gpt-5-codex", "o3-pro", "gpt-4o"} {
		t.Run(model, func(t *testing.T) {
			p, err := registry.Get("openai", "sk-test-key-that-is-long-enough", model, nil)
			if err != nil {
				t.Fatalf("subset registry could not resolve %q: %v", model, err)
			}
			if got := p.Name(); got != "openai" {
				t.Errorf("provider.Name() = %q, want openai (no responses provider registered)", got)
			}
		})
	}
}

// TestRoutingRespectsRegisterOverride verifies a caller-supplied constructor under
// "openai" is used rather than silently swapped for the built-in Responses provider.
func TestRoutingRespectsRegisterOverride(t *testing.T) {
	registry := NewProviderRegistry()
	registry.Register("openai", func(apiKey, model string, extraHeaders map[string]string) Provider {
		return NewOpenAIProvider(apiKey, model, extraHeaders)
	})

	p, err := registry.Get("openai", "sk-test-key-that-is-long-enough", "gpt-5.6-sol", nil)
	if err != nil {
		t.Fatalf("Get returned error: %v", err)
	}
	if got := p.Name(); got != "openai" {
		t.Errorf("provider.Name() = %q, want openai; a Register override must not be routed away", got)
	}
}

// TestFineTunedModelsRouteOnBaseModel checks that the customer-chosen name in a
// fine-tune id cannot trip the family patterns.
func TestFineTunedModelsRouteOnBaseModel(t *testing.T) {
	tests := []struct {
		name  string
		model string
		want  string
	}{
		{"customer name containing pro", "ft:gpt-4o-mini-2024-07-18:acme:sales-pro-v2:AbC123", "openai"},
		{"customer name containing codex", "ft:gpt-4o-mini-2024-07-18:acme:codex-helper:AbC123", "openai"},
		{"fine-tune of a routed base still routes", "ft:gpt-5.6-sol:acme:tuned:AbC123", "openai-responses"},
		{"malformed ft id falls back to whole string", "ft:", "openai"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := routeOpenAIProvider("openai", tt.model); got != tt.want {
				t.Errorf("routeOpenAIProvider(openai, %q) = %q, want %q", tt.model, got, tt.want)
			}
		})
	}
}

// TestFutureChatVariantIsExcluded covers the non-reasoning chat carve-out above
// the routing bound, which the gpt-5-chat prefix predicate could never reach.
func TestFutureChatVariantIsExcluded(t *testing.T) {
	for _, model := range []string{"gpt-5.6-chat-latest", "gpt-5.4-chat"} {
		if rejectsToolsOnChatCompletions(model) {
			t.Errorf("rejectsToolsOnChatCompletions(%q) = true; non-reasoning chat variants have no reasoning to conflict with tools", model)
		}
	}
	// The reasoning models above the bound must still route.
	if !rejectsToolsOnChatCompletions("gpt-5.6-sol") {
		t.Error("gpt-5.6-sol must still route")
	}
}

// imagePartsFrom pulls the content parts out of a Responses request body's first
// input message, or nil when input is a bare string.
func imagePartsFrom(t *testing.T, body []byte) []map[string]interface{} {
	t.Helper()
	var request map[string]interface{}
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatalf("body is not valid JSON: %v", err)
	}
	items, ok := request["input"].([]interface{})
	if !ok || len(items) == 0 {
		return nil
	}
	first, _ := items[0].(map[string]interface{})
	parts, _ := first["content"].([]interface{})
	out := make([]map[string]interface{}, 0, len(parts))
	for _, p := range parts {
		if m, ok := p.(map[string]interface{}); ok {
			out = append(out, m)
		}
	}
	return out
}

// TestResponsesImagesUseResponsesShape covers every request path that can carry an
// image. The Responses API needs {"type":"input_image","image_url":"<url>"}; the
// Chat Completions shape ({"type":"image_url","image_url":{"url":…}}) is rejected.
// Routing makes these paths reachable for callers who configured plain "openai".
func TestResponsesImagesUseResponsesShape(t *testing.T) {
	images := []types.ContentPart{
		{Type: types.ContentTypeImageURL, ImageURL: &types.ImageURL{URL: "https://example.com/a.png"}},
	}
	opts := map[string]interface{}{"images": images}

	paths := map[string]func(p Provider) ([]byte, error){
		"PrepareRequest": func(p Provider) ([]byte, error) {
			return p.PrepareRequest("describe", opts)
		},
		"PrepareRequestWithSchema": func(p Provider) ([]byte, error) {
			return p.PrepareRequestWithSchema("describe", opts,
				map[string]interface{}{"type": "object", "properties": map[string]interface{}{}})
		},
		"PrepareStreamRequest": func(p Provider) ([]byte, error) {
			return p.PrepareStreamRequest("describe", opts)
		},
	}

	for name, build := range paths {
		t.Run(name, func(t *testing.T) {
			p := NewOpenAIResponsesProvider("sk-test", "gpt-5.6-sol", nil)
			body, err := build(p)
			if err != nil {
				t.Fatalf("%s returned error: %v", name, err)
			}
			parts := imagePartsFrom(t, body)
			if len(parts) == 0 {
				t.Fatalf("%s dropped the image entirely; body=%s", name, body)
			}
			var sawImage bool
			for _, part := range parts {
				switch part["type"] {
				case "input_image":
					sawImage = true
					if _, isString := part["image_url"].(string); !isString {
						t.Errorf("%s: image_url must be a plain string, got %T (%v)",
							name, part["image_url"], part["image_url"])
					}
				case "image_url":
					t.Errorf("%s: emitted the Chat Completions image shape, which /v1/responses rejects", name)
				}
			}
			if !sawImage {
				t.Errorf("%s: no input_image part in body=%s", name, body)
			}
		})
	}
}

// TestResponsesMultiContentMessagesUseResponsesShape covers the multi-turn path,
// where text parts also differ: "input_text" on input roles, "output_text" on
// assistant turns — never Chat Completions' "text".
func TestResponsesMultiContentMessagesUseResponsesShape(t *testing.T) {
	tests := []struct {
		role         string
		wantTextType string
	}{
		{"user", "input_text"},
		{"assistant", "output_text"},
	}

	for _, tt := range tests {
		t.Run(tt.role, func(t *testing.T) {
			p := NewOpenAIResponsesProvider("sk-test", "gpt-5.6-sol", nil)
			body, err := p.PrepareRequestWithMessages([]types.MemoryMessage{{
				Role: tt.role,
				MultiContent: []types.ContentPart{
					{Type: types.ContentTypeText, Text: "what is this"},
					{Type: types.ContentTypeImageURL, ImageURL: &types.ImageURL{URL: "https://example.com/a.png"}},
				},
			}}, map[string]interface{}{})
			if err != nil {
				t.Fatalf("PrepareRequestWithMessages returned error: %v", err)
			}

			var sawText, sawImage bool
			for _, part := range imagePartsFrom(t, body) {
				switch part["type"] {
				case tt.wantTextType:
					sawText = true
				case "text":
					t.Errorf("emitted Chat Completions text shape; want %q", tt.wantTextType)
				case "input_image":
					sawImage = true
					if _, isString := part["image_url"].(string); !isString {
						t.Errorf("image_url must be a plain string, got %T", part["image_url"])
					}
				case "image_url":
					t.Error("emitted the Chat Completions image shape, which /v1/responses rejects")
				}
			}
			if !sawText {
				t.Errorf("no %q part in body=%s", tt.wantTextType, body)
			}
			if !sawImage {
				t.Errorf("no input_image part in body=%s", body)
			}
		})
	}
}
