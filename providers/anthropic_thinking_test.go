package providers

import (
	"encoding/json"
	"testing"

	"github.com/teilomillet/gollm/types"
)

// decodeAnthropicRequest unmarshals a prepared request body for inspection.
func decodeAnthropicRequest(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("failed to unmarshal request: %v", err)
	}
	return req
}

// mustObject returns v as a JSON object or fails with a diagnostic message.
func mustObject(t *testing.T, v interface{}, what string) map[string]interface{} {
	t.Helper()
	m, ok := v.(map[string]interface{})
	if !ok {
		t.Fatalf("expected %s to be an object, got %T (%v)", what, v, v)
	}
	return m
}

// mustFloat returns v as a JSON number or fails with a diagnostic message.
func mustFloat(t *testing.T, v interface{}, what string) float64 {
	t.Helper()
	f, ok := v.(float64)
	if !ok {
		t.Fatalf("expected %s to be numeric, got %T (%v)", what, v, v)
	}
	return f
}

// TestAnthropicThinkingAdaptive verifies that adaptive-thinking models emit
// thinking:{type:"adaptive"} plus an output_config.effort — and never the
// budget_tokens field, which those models reject with a 400.
func TestAnthropicThinkingAdaptive(t *testing.T) {
	for _, model := range []string{"claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-5", "claude-fable-5", "claude-sonnet-4-6"} {
		p := NewAnthropicProvider("fake-key", model, nil).(*AnthropicProvider)
		p.SetOption("max_tokens", 1024)

		body, err := p.PrepareRequest("hi", map[string]interface{}{"reasoning_effort": "high"})
		if err != nil {
			t.Fatalf("%s: PrepareRequest failed: %v", model, err)
		}
		req := decodeAnthropicRequest(t, body)

		thinking, ok := req["thinking"].(map[string]interface{})
		if !ok {
			t.Fatalf("%s: expected thinking object, got %v", model, req["thinking"])
		}
		if thinking["type"] != "adaptive" {
			t.Errorf("%s: expected adaptive thinking, got %v", model, thinking["type"])
		}
		if _, hasBudget := thinking["budget_tokens"]; hasBudget {
			t.Errorf("%s: budget_tokens must not be sent for adaptive-thinking models", model)
		}
		oc, ok := req["output_config"].(map[string]interface{})
		if !ok || oc["effort"] != "high" {
			t.Errorf("%s: expected output_config.effort=high, got %v", model, req["output_config"])
		}
	}
}

// TestAnthropicThinkingBudgeted verifies that older models use manual extended
// thinking with an effort-derived budget_tokens strictly below max_tokens.
func TestAnthropicThinkingBudgeted(t *testing.T) {
	p := NewAnthropicProvider("fake-key", "claude-sonnet-4-5", nil).(*AnthropicProvider)
	p.SetOption("max_tokens", 16000)

	body, err := p.PrepareRequest("hi", map[string]interface{}{"reasoning_effort": "medium"})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeAnthropicRequest(t, body)

	thinking, ok := req["thinking"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected thinking object, got %v", req["thinking"])
	}
	if thinking["type"] != "enabled" {
		t.Errorf("expected enabled thinking, got %v", thinking["type"])
	}
	budget, ok := thinking["budget_tokens"].(float64)
	if !ok {
		t.Fatalf("expected numeric budget_tokens, got %v", thinking["budget_tokens"])
	}
	if budget != 8192 {
		t.Errorf("expected medium budget 8192, got %v", budget)
	}
	if _, hasOC := req["output_config"]; hasOC {
		t.Errorf("output_config.effort should not be set for legacy models")
	}
}

// TestAnthropicThinkingBudgetClampedToMaxTokens verifies budget_tokens stays
// below max_tokens when the effort default would exceed it.
func TestAnthropicThinkingBudgetClampedToMaxTokens(t *testing.T) {
	p := NewAnthropicProvider("fake-key", "claude-sonnet-4-5", nil).(*AnthropicProvider)
	p.SetOption("max_tokens", 8000)

	body, err := p.PrepareRequest("hi", map[string]interface{}{"reasoning_effort": "high"})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeAnthropicRequest(t, body)
	thinking := mustObject(t, req["thinking"], "thinking")
	budget := mustFloat(t, thinking["budget_tokens"], "budget_tokens")
	if budget >= 8000 {
		t.Errorf("budget_tokens (%v) must be < max_tokens (8000)", budget)
	}
}

// TestAnthropicThinkingProviderLevelOption verifies that reasoning_effort set as
// a provider-level default (via SetOption) is honored, matching OpenAI's merge of
// provider defaults with per-request options.
func TestAnthropicThinkingProviderLevelOption(t *testing.T) {
	p := NewAnthropicProvider("fake-key", "claude-opus-4-8", nil).(*AnthropicProvider)
	p.SetOption("max_tokens", 1024)
	p.SetOption("reasoning_effort", "low")

	body, err := p.PrepareRequest("hi", map[string]interface{}{})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeAnthropicRequest(t, body)

	thinking, ok := req["thinking"].(map[string]interface{})
	if !ok || thinking["type"] != "adaptive" {
		t.Fatalf("expected adaptive thinking from provider-level effort, got %v", req["thinking"])
	}
	oc, ok := req["output_config"].(map[string]interface{})
	if !ok || oc["effort"] != "low" {
		t.Errorf("expected output_config.effort=low, got %v", req["output_config"])
	}
}

// TestAnthropicThinkingRequestOverridesProvider verifies a per-request
// reasoning_effort takes precedence over the provider-level default.
func TestAnthropicThinkingRequestOverridesProvider(t *testing.T) {
	p := NewAnthropicProvider("fake-key", "claude-opus-4-8", nil).(*AnthropicProvider)
	p.SetOption("max_tokens", 1024)
	p.SetOption("reasoning_effort", "low")

	body, err := p.PrepareRequest("hi", map[string]interface{}{"reasoning_effort": "high"})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeAnthropicRequest(t, body)
	oc := mustObject(t, req["output_config"], "output_config")
	if oc["effort"] != "high" {
		t.Errorf("per-request effort should override provider default, got %v", oc["effort"])
	}
}

// TestAnthropicThinkingSkippedWhenMaxTokensTooSmall verifies that for legacy
// models, thinking is omitted (rather than sent with an invalid budget_tokens)
// when max_tokens is at or below the 1024 budget floor.
func TestAnthropicThinkingSkippedWhenMaxTokensTooSmall(t *testing.T) {
	for _, maxTokens := range []int{100, 1024} {
		p := NewAnthropicProvider("fake-key", "claude-sonnet-4-5", nil).(*AnthropicProvider)
		p.SetOption("max_tokens", maxTokens)

		body, err := p.PrepareRequest("hi", map[string]interface{}{"reasoning_effort": "high"})
		if err != nil {
			t.Fatalf("max_tokens=%d: PrepareRequest failed: %v", maxTokens, err)
		}
		req := decodeAnthropicRequest(t, body)
		if _, ok := req["thinking"]; ok {
			t.Errorf("max_tokens=%d: thinking must be omitted when budget_tokens cannot be < max_tokens", maxTokens)
		}
	}
}

// TestAnthropicThinkingSkippedWhenMaxTokensUnknown verifies that for legacy
// models, thinking is omitted when no max_tokens is available in the request body
// (so budget_tokens < max_tokens cannot be enforced) rather than emitting an
// unvalidatable budget.
func TestAnthropicThinkingSkippedWhenMaxTokensUnknown(t *testing.T) {
	// Provider constructed without SetDefaultOptions / SetOption("max_tokens"),
	// so requestBody["max_tokens"] is nil (toInt -> 0).
	p := NewAnthropicProvider("fake-key", "claude-sonnet-4-5", nil).(*AnthropicProvider)

	body, err := p.PrepareRequest("hi", map[string]interface{}{"reasoning_effort": "high"})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeAnthropicRequest(t, body)
	if _, ok := req["thinking"]; ok {
		t.Errorf("thinking must be omitted for a legacy model when max_tokens is unknown")
	}
}

// TestAnthropicEffortNormalization verifies the reasoning_effort -> output_config
// effort mapping (cross-provider semantics): "minimal" collapses to "low",
// unknown/empty-of-meaning values default to "medium", canonical values pass
// through, and inputs are case-folded and trimmed. Uses an adaptive model so the
// normalized value is emitted verbatim as output_config.effort.
func TestAnthropicEffortNormalization(t *testing.T) {
	cases := map[string]string{
		"minimal": "low",
		"low":     "low",
		"medium":  "medium",
		"high":    "high",
		"max":     "max",
		"MAX":     "max",
		"  High ": "high",
		"bogus":   "medium",
	}
	for input, want := range cases {
		p := NewAnthropicProvider("fake-key", "claude-opus-4-8", nil).(*AnthropicProvider)
		p.SetOption("max_tokens", 4096)

		body, err := p.PrepareRequest("hi", map[string]interface{}{"reasoning_effort": input})
		if err != nil {
			t.Fatalf("effort=%q: PrepareRequest failed: %v", input, err)
		}
		req := decodeAnthropicRequest(t, body)
		oc := mustObject(t, req["output_config"], "output_config")
		if oc["effort"] != want {
			t.Errorf("effort=%q: expected output_config.effort=%q, got %v", input, want, oc["effort"])
		}
	}
}

// TestAnthropicThinkingBudgetBelowMaxTokens verifies the budget is strictly less
// than max_tokens for legacy models even when max_tokens is just above the floor.
func TestAnthropicThinkingBudgetBelowMaxTokens(t *testing.T) {
	for _, maxTokens := range []int{1025, 1500, 5000, 20000} {
		p := NewAnthropicProvider("fake-key", "claude-sonnet-4-5", nil).(*AnthropicProvider)
		p.SetOption("max_tokens", maxTokens)

		body, err := p.PrepareRequest("hi", map[string]interface{}{"reasoning_effort": "high"})
		if err != nil {
			t.Fatalf("max_tokens=%d: PrepareRequest failed: %v", maxTokens, err)
		}
		req := decodeAnthropicRequest(t, body)
		thinking := mustObject(t, req["thinking"], "thinking")
		budget := mustFloat(t, thinking["budget_tokens"], "budget_tokens")
		if budget < 1024 || budget >= float64(maxTokens) {
			t.Errorf("max_tokens=%d: budget %v must satisfy 1024 <= budget < max_tokens", maxTokens, budget)
		}
	}
}

// TestAnthropicThinkingSchemaPathClampsToOptionMaxTokens verifies that the
// schema request path (which sets max_tokens only via the option passthrough)
// still derives a valid budget below max_tokens for legacy models.
func TestAnthropicThinkingSchemaPathClampsToOptionMaxTokens(t *testing.T) {
	p := NewAnthropicProvider("fake-key", "claude-sonnet-4-5", nil).(*AnthropicProvider)

	body, err := p.PrepareRequestWithSchema("hi", map[string]interface{}{
		"reasoning_effort": "high",
		"max_tokens":       4096,
	}, map[string]interface{}{"type": "object"})
	if err != nil {
		t.Fatalf("PrepareRequestWithSchema failed: %v", err)
	}
	req := decodeAnthropicRequest(t, body)
	thinking := mustObject(t, req["thinking"], "thinking")
	budget := mustFloat(t, thinking["budget_tokens"], "budget_tokens")
	if budget >= 4096 {
		t.Errorf("schema-path budget %v must be < max_tokens (4096)", budget)
	}
}

// TestAnthropicThinkingUnknownModelDefaultsAdaptive verifies that an unknown or
// future model id fails open to adaptive thinking rather than sending
// budget_tokens (which newer models reject with a 400).
func TestAnthropicThinkingUnknownModelDefaultsAdaptive(t *testing.T) {
	for _, model := range []string{"claude-opus-4-9", "claude-opus-5", "claude-sonnet-6", "claude-future-x"} {
		p := NewAnthropicProvider("fake-key", model, nil).(*AnthropicProvider)
		p.SetOption("max_tokens", 1024)

		body, err := p.PrepareRequest("hi", map[string]interface{}{"reasoning_effort": "high"})
		if err != nil {
			t.Fatalf("%s: PrepareRequest failed: %v", model, err)
		}
		req := decodeAnthropicRequest(t, body)
		thinking, ok := req["thinking"].(map[string]interface{})
		if !ok || thinking["type"] != "adaptive" {
			t.Errorf("%s: unknown model should default to adaptive, got %v", model, req["thinking"])
		}
		if _, hasBudget := thinking["budget_tokens"]; hasBudget {
			t.Errorf("%s: unknown model must not send budget_tokens", model)
		}
	}
}

// TestAnthropicThinkingXhighDemotedForOlderModels verifies xhigh is demoted to
// high on adaptive models that predate the xhigh effort level (Opus 4.6 /
// Sonnet 4.6), while models that support it keep xhigh.
func TestAnthropicThinkingXhighDemotedForOlderModels(t *testing.T) {
	cases := map[string]string{
		"claude-opus-4-6":   "high",  // xhigh not supported → demoted
		"claude-sonnet-4-6": "high",  // xhigh not supported → demoted
		"claude-opus-4-8":   "xhigh", // supported → preserved
		"claude-sonnet-5":   "xhigh", // supported → preserved
	}
	for model, want := range cases {
		p := NewAnthropicProvider("fake-key", model, nil).(*AnthropicProvider)
		p.SetOption("max_tokens", 4096)

		body, err := p.PrepareRequest("hi", map[string]interface{}{"reasoning_effort": "xhigh"})
		if err != nil {
			t.Fatalf("%s: PrepareRequest failed: %v", model, err)
		}
		req := decodeAnthropicRequest(t, body)
		oc := mustObject(t, req["output_config"], "output_config")
		if oc["effort"] != want {
			t.Errorf("%s: expected effort %q, got %v", model, want, oc["effort"])
		}
	}
}

// TestAnthropicThinkingMergesUserOutputConfig verifies applyThinking merges the
// effort into a caller-supplied output_config rather than clobbering it, and
// wins over a raw thinking option.
func TestAnthropicThinkingMergesUserOutputConfig(t *testing.T) {
	p := NewAnthropicProvider("fake-key", "claude-opus-4-8", nil).(*AnthropicProvider)
	p.SetOption("max_tokens", 4096)

	body, err := p.PrepareRequest("hi", map[string]interface{}{
		"reasoning_effort": "high",
		"output_config":    map[string]interface{}{"other": "keep"},
		"thinking":         map[string]interface{}{"type": "enabled", "budget_tokens": 30000},
	})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeAnthropicRequest(t, body)

	thinking := mustObject(t, req["thinking"], "thinking")
	if thinking["type"] != "adaptive" {
		t.Errorf("applyThinking should win over raw thinking option, got %v", thinking["type"])
	}
	if _, hasBudget := thinking["budget_tokens"]; hasBudget {
		t.Errorf("budget_tokens must not survive on an adaptive model")
	}
	oc := mustObject(t, req["output_config"], "output_config")
	if oc["effort"] != "high" {
		t.Errorf("expected effort=high, got %v", oc["effort"])
	}
	if oc["other"] != "keep" {
		t.Errorf("caller output_config keys must be preserved, got %v", oc)
	}
}

// TestAnthropicReasoningEffortEnumType verifies the published types.ReasoningEffort
// enum is accepted (not just a plain string) on the Anthropic path.
func TestAnthropicReasoningEffortEnumType(t *testing.T) {
	p := NewAnthropicProvider("fake-key", "claude-opus-4-8", nil).(*AnthropicProvider)
	p.SetOption("max_tokens", 4096)

	body, err := p.PrepareRequest("hi", map[string]interface{}{"reasoning_effort": types.ReasoningEffortXHigh})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeAnthropicRequest(t, body)
	oc := mustObject(t, req["output_config"], "output_config")
	if oc["effort"] != "xhigh" {
		t.Errorf("expected ReasoningEffortXHigh honored as xhigh, got %v", oc["effort"])
	}
}

// TestAnthropicThinkingOffByDefault verifies that no thinking field is emitted
// when reasoning_effort is absent, matching Anthropic's default behavior.
func TestAnthropicThinkingOffByDefault(t *testing.T) {
	p := NewAnthropicProvider("fake-key", "claude-opus-4-8", nil).(*AnthropicProvider)
	p.SetOption("max_tokens", 1024)

	body, err := p.PrepareRequest("hi", map[string]interface{}{})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeAnthropicRequest(t, body)
	if _, ok := req["thinking"]; ok {
		t.Errorf("thinking must be absent when reasoning_effort is not set")
	}
}
