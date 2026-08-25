package gollm

import (
	"testing"

	"github.com/teilomillet/gollm/config"
)

// TestToolReasoningPolicySelectsTransport covers the policy resolution that runs
// when a client is built. prefer-speed (the default) keeps affected models on Chat
// Completions, where the provider gives up reasoning on tool-carrying requests;
// prefer-quality moves them to the Responses API, the only transport that accepts
// function tools and reasoning together.
func TestToolReasoningPolicySelectsTransport(t *testing.T) {
	tests := []struct {
		name     string
		provider string
		model    string
		policy   config.ToolReasoningPolicy
		want     string
	}{
		// Default: unset behaves as prefer-speed.
		{"unset keeps chat", "openai", "gpt-5.6-sol", "", "openai"},
		{"prefer-speed keeps chat", "openai", "gpt-5.6-sol", config.ToolReasoningPreferSpeed, "openai"},
		{"prefer-speed keeps chat on 5.4", "openai", "gpt-5.4-mini", config.ToolReasoningPreferSpeed, "openai"},

		// prefer-quality moves the affected set, and only the affected set.
		{"prefer-quality moves 5.6", "openai", "gpt-5.6-sol", config.ToolReasoningPreferQuality, "openai-responses"},
		{"prefer-quality moves 5.4", "openai", "gpt-5.4-mini", config.ToolReasoningPreferQuality, "openai-responses"},
		{"prefer-quality moves 5.5", "openai", "gpt-5.5", config.ToolReasoningPreferQuality, "openai-responses"},

		// Unaffected models must not be dragged onto a slower endpoint for nothing.
		{"prefer-quality leaves gpt-4o", "openai", "gpt-4o", config.ToolReasoningPreferQuality, "openai"},
		{"prefer-quality leaves o3", "openai", "o3", config.ToolReasoningPreferQuality, "openai"},
		{"prefer-quality leaves gpt-5.3", "openai", "gpt-5.3", config.ToolReasoningPreferQuality, "openai"},
		{"prefer-quality leaves chat variant", "openai", "gpt-5.6-chat-latest", config.ToolReasoningPreferQuality, "openai"},

		// An explicit transport is a stronger statement of intent than the policy.
		{"explicit chat pin survives", "openai-chat", "gpt-5.6-sol", config.ToolReasoningPreferQuality, "openai-chat"},
		{"explicit responses survives", "openai-responses", "gpt-4o", config.ToolReasoningPreferSpeed, "openai-responses"},

		// Other providers are never touched.
		{"anthropic untouched", "anthropic", "claude-opus-4-5", config.ToolReasoningPreferQuality, "anthropic"},
		{"azure untouched", "azure-openai", "gpt-5.6-sol", config.ToolReasoningPreferQuality, "azure-openai"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.Config{Provider: tt.provider, Model: tt.model, ToolReasoning: tt.policy}
			applyOpenAIToolReasoningPolicy(cfg)
			if cfg.Provider != tt.want {
				t.Errorf("provider = %q, want %q (model %q, policy %q)",
					cfg.Provider, tt.want, tt.model, tt.policy)
			}
		})
	}
}

// TestToolReasoningPolicyResolvesAPIKey verifies the ordering in NewLLM: a client
// the policy switches to the Responses transport must still find the OpenAI key,
// which callers set under "openai".
func TestToolReasoningPolicyResolvesAPIKey(t *testing.T) {
	cfg := &config.Config{
		Provider:      "openai",
		Model:         "gpt-5.6-sol",
		ToolReasoning: config.ToolReasoningPreferQuality,
		APIKeys:       map[string]string{"openai": "sk-test-key-that-is-long-enough"},
	}

	applyOpenAIToolReasoningPolicy(cfg)
	ensureOpenAIAliasKey(cfg)

	if cfg.Provider != "openai-responses" {
		t.Fatalf("provider = %q, want openai-responses", cfg.Provider)
	}
	if got := cfg.APIKeys[cfg.Provider]; got == "" {
		t.Error("API key did not carry over to the routed transport; the client would fail to build")
	}
}

// TestWithOpenAIToolReasoningOption checks the exported option and its root-package
// re-exports actually set the field callers expect.
func TestWithOpenAIToolReasoningOption(t *testing.T) {
	cfg := &config.Config{}
	WithOpenAIToolReasoning(ToolReasoningPreferQuality)(cfg)
	if cfg.ToolReasoning != config.ToolReasoningPreferQuality {
		t.Errorf("ToolReasoning = %q, want %q", cfg.ToolReasoning, config.ToolReasoningPreferQuality)
	}

	WithOpenAIToolReasoning(ToolReasoningPreferSpeed)(cfg)
	if cfg.ToolReasoning != config.ToolReasoningPreferSpeed {
		t.Errorf("ToolReasoning = %q, want %q", cfg.ToolReasoning, config.ToolReasoningPreferSpeed)
	}
}
