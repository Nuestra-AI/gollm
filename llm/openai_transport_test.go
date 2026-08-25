package llm

import (
	"testing"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/providers"
	"github.com/teilomillet/gollm/utils"
)

// TestResolveOpenAITransport covers policy resolution: prefer-speed keeps affected
// models on Chat Completions, prefer-quality moves them to the only transport
// accepting tools and reasoning together.
func TestResolveOpenAITransport(t *testing.T) {
	tests := []struct {
		name     string
		provider string
		model    string
		policy   config.ToolReasoningPolicy
		want     string
	}{
		{"unset keeps chat", "openai", "gpt-5.6-sol", "", "openai"},
		{"prefer-speed keeps chat", "openai", "gpt-5.6-sol", config.ToolReasoningPreferSpeed, "openai"},

		{"prefer-quality moves 5.6", "openai", "gpt-5.6-sol", config.ToolReasoningPreferQuality, "openai-responses"},
		{"prefer-quality moves 5.4", "openai", "gpt-5.4-mini", config.ToolReasoningPreferQuality, "openai-responses"},

		// Unaffected models must not be dragged onto a slower endpoint for nothing.
		{"prefer-quality leaves gpt-4o", "openai", "gpt-4o", config.ToolReasoningPreferQuality, "openai"},
		{"prefer-quality leaves o3", "openai", "o3", config.ToolReasoningPreferQuality, "openai"},
		{"prefer-quality leaves gpt-5.3", "openai", "gpt-5.3", config.ToolReasoningPreferQuality, "openai"},

		// An explicit transport outranks the policy.
		{"explicit chat pin survives", "openai-chat", "gpt-5.6-sol", config.ToolReasoningPreferQuality, "openai-chat"},
		{"explicit responses survives", "openai-responses", "gpt-4o", config.ToolReasoningPreferSpeed, "openai-responses"},

		{"anthropic untouched", "anthropic", "claude-opus-4-5", config.ToolReasoningPreferQuality, "anthropic"},
		{"azure untouched", "azure-openai", "gpt-5.6-sol", config.ToolReasoningPreferQuality, "azure-openai"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.Config{Provider: tt.provider, Model: tt.model, ToolReasoning: tt.policy}
			ResolveOpenAITransport(cfg)
			if cfg.Provider != tt.want {
				t.Errorf("provider = %q, want %q", cfg.Provider, tt.want)
			}
		})
	}
}

// TestResolveOpenAITransportIsIdempotent: it runs from both constructors, so a
// second pass must not change what the first decided.
func TestResolveOpenAITransportIsIdempotent(t *testing.T) {
	cfg := &config.Config{
		Provider: "openai", Model: "gpt-5.6-sol",
		ToolReasoning: config.ToolReasoningPreferQuality,
		APIKeys:       map[string]string{"openai": "sk-test-key-that-is-long-enough"},
	}
	ResolveOpenAITransport(cfg)
	first := cfg.Provider
	ResolveOpenAITransport(cfg)

	if cfg.Provider != first {
		t.Errorf("second pass changed the provider: %q -> %q", first, cfg.Provider)
	}
	if cfg.APIKeys[cfg.Provider] == "" {
		t.Error("API key did not carry over to the resolved transport")
	}
}

// TestNewLLMHonorsToolReasoningPolicy pins the fix for the real defect: llm.NewLLM
// is public and used directly, and resolves providers straight from the registry.
// With resolution only in the gollm wrapper, a prefer-quality client built this way
// silently behaved like prefer-speed.
func TestNewLLMHonorsToolReasoningPolicy(t *testing.T) {
	tests := []struct {
		name   string
		policy config.ToolReasoningPolicy
		want   string
	}{
		{"prefer-quality reaches the Responses transport", config.ToolReasoningPreferQuality, "openai-responses"},
		{"prefer-speed stays on Chat Completions", config.ToolReasoningPreferSpeed, "openai"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.Config{
				Provider: "openai", Model: "gpt-5.6-sol", ToolReasoning: tt.policy,
				APIKeys: map[string]string{"openai": "sk-test-key-that-is-long-enough"},
			}
			client, err := NewLLM(cfg, utils.NewLogger(utils.LogLevelWarn), providers.NewProviderRegistry())
			if err != nil {
				t.Fatalf("NewLLM returned error: %v", err)
			}
			got := client.(*LLMImpl).Provider.Name()
			if got != tt.want {
				t.Errorf("provider = %q, want %q", got, tt.want)
			}
		})
	}
}
