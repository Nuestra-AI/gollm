package providers

import (
	"encoding/json"
	"testing"

	"github.com/teilomillet/gollm/types"
)

// decodeGoogleRequest unmarshals a prepared request body for inspection.
func decodeGoogleRequest(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("failed to unmarshal request: %v", err)
	}
	return req
}

// geminiThinkingConfig extracts extra_body.google.thinking_config or fails.
func geminiThinkingConfig(t *testing.T, req map[string]interface{}) map[string]interface{} {
	t.Helper()
	extraBody := mustObject(t, req["extra_body"], "extra_body")
	google := mustObject(t, extraBody["google"], "extra_body.google")
	return mustObject(t, google["thinking_config"], "extra_body.google.thinking_config")
}

func newGoogle(t *testing.T, model string) *GoogleProvider {
	t.Helper()
	p := NewGoogleProvider("fake-key", model, nil).(*GoogleProvider)
	p.SetOption("max_tokens", 4096)
	return p
}

// TestGeminiReasoningEffort verifies reasoning_effort is emitted as the top-level
// Gemini knob (normalized) with no extra_body alongside it, on both families.
func TestGeminiReasoningEffort(t *testing.T) {
	for _, model := range []string{"gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-pro-preview", "gemini-3-flash-preview"} {
		body, err := newGoogle(t, model).PrepareRequest("hi", map[string]interface{}{"reasoning_effort": "high"})
		if err != nil {
			t.Fatalf("%s: PrepareRequest failed: %v", model, err)
		}
		req := decodeGoogleRequest(t, body)
		if req["reasoning_effort"] != "high" {
			t.Errorf("%s: expected top-level reasoning_effort=high, got %v", model, req["reasoning_effort"])
		}
		if _, hasExtra := req["extra_body"]; hasExtra {
			t.Errorf("%s: reasoning_effort must not emit extra_body, got %v", model, req["extra_body"])
		}
	}
}

// TestGeminiEffortNormalization verifies the reasoning_effort mapping onto the
// none/low/medium/high levels Gemini accepts (flash so "none" is not clamped).
func TestGeminiEffortNormalization(t *testing.T) {
	cases := map[string]string{
		"none": "none", "minimal": "low", "low": "low", "medium": "medium",
		"high": "high", "xhigh": "high", "max": "high", "  High ": "high", "bogus": "medium",
	}
	for input, want := range cases {
		body, err := newGoogle(t, "gemini-2.5-flash").PrepareRequest("hi", map[string]interface{}{"reasoning_effort": input})
		if err != nil {
			t.Fatalf("effort=%q: PrepareRequest failed: %v", input, err)
		}
		req := decodeGoogleRequest(t, body)
		if req["reasoning_effort"] != want {
			t.Errorf("effort=%q: expected reasoning_effort=%q, got %v", input, want, req["reasoning_effort"])
		}
	}
}

// TestGeminiEffortIsPrimaryOverBudgetAndLevel verifies reasoning_effort wins over
// an explicit budget/level in the same request (the API rejects both together).
func TestGeminiEffortIsPrimaryOverBudgetAndLevel(t *testing.T) {
	// 2.x model with effort + budget.
	body, err := newGoogle(t, "gemini-2.5-flash").PrepareRequest("hi", map[string]interface{}{
		"reasoning_effort": "high", "thinking_budget": 2048,
	})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeGoogleRequest(t, body)
	if req["reasoning_effort"] != "high" {
		t.Errorf("effort must win over budget, got reasoning_effort=%v", req["reasoning_effort"])
	}
	if _, hasExtra := req["extra_body"]; hasExtra {
		t.Errorf("no budget must be emitted when effort is set, got %v", req["extra_body"])
	}

	// 3.x model with effort + level.
	body, err = newGoogle(t, "gemini-3-flash-preview").PrepareRequest("hi", map[string]interface{}{
		"reasoning_effort": "low", "thinking_level": "high",
	})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req = decodeGoogleRequest(t, body)
	if req["reasoning_effort"] != "low" {
		t.Errorf("effort must win over level, got reasoning_effort=%v", req["reasoning_effort"])
	}
	if _, hasExtra := req["extra_body"]; hasExtra {
		t.Errorf("no level must be emitted when effort is set, got %v", req["extra_body"])
	}
}

// TestGeminiPerRequestEffortOverridesProviderBudget verifies the cross-scope
// precedence fix: a per-request reasoning_effort wins over a provider-level
// thinking_budget default (reasoning_effort is the primary control).
func TestGeminiPerRequestEffortOverridesProviderBudget(t *testing.T) {
	p := newGoogle(t, "gemini-2.5-flash")
	p.SetOption("thinking_budget", 0) // provider-level "disable thinking"

	body, err := p.PrepareRequest("hi", map[string]interface{}{"reasoning_effort": "high"})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeGoogleRequest(t, body)
	if req["reasoning_effort"] != "high" {
		t.Errorf("per-request effort must beat provider budget, got reasoning_effort=%v", req["reasoning_effort"])
	}
	if _, hasExtra := req["extra_body"]; hasExtra {
		t.Errorf("provider budget must not survive a per-request effort, got %v", req["extra_body"])
	}
}

// TestGeminiEmptyEffortFallsBackToProvider verifies an empty-string per-request
// effort does not mask a non-empty provider-level default.
func TestGeminiEmptyEffortFallsBackToProvider(t *testing.T) {
	p := newGoogle(t, "gemini-2.5-flash")
	p.SetOption("reasoning_effort", "high")

	body, err := p.PrepareRequest("hi", map[string]interface{}{"reasoning_effort": ""})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeGoogleRequest(t, body)
	if req["reasoning_effort"] != "high" {
		t.Errorf("empty per-request effort should fall back to provider default, got %v", req["reasoning_effort"])
	}
}

// TestGeminiThinkingBudget verifies an explicit thinking_budget routes through
// extra_body.google.thinking_config on a 2.x model.
func TestGeminiThinkingBudget(t *testing.T) {
	body, err := newGoogle(t, "gemini-2.5-flash").PrepareRequest("hi", map[string]interface{}{"thinking_budget": 2048})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeGoogleRequest(t, body)
	if _, hasEffort := req["reasoning_effort"]; hasEffort {
		t.Errorf("no reasoning_effort expected, got %v", req["reasoning_effort"])
	}
	cfg := geminiThinkingConfig(t, req)
	if mustFloat(t, cfg["thinking_budget"], "thinking_budget") != 2048 {
		t.Errorf("expected thinking_budget=2048, got %v", cfg["thinking_budget"])
	}
}

// TestGeminiThinkingBudgetString verifies a numeric-string thinking_budget is
// parsed (not silently coerced to 0, which would disable thinking).
func TestGeminiThinkingBudgetString(t *testing.T) {
	body, err := newGoogle(t, "gemini-2.5-flash").PrepareRequest("hi", map[string]interface{}{"thinking_budget": "2048"})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	cfg := geminiThinkingConfig(t, decodeGoogleRequest(t, body))
	if mustFloat(t, cfg["thinking_budget"], "thinking_budget") != 2048 {
		t.Errorf("expected string budget parsed to 2048, got %v", cfg["thinking_budget"])
	}
}

// TestGeminiThinkingBudgetUnparseableTreatedAsUnset verifies a malformed budget
// is treated as unset (no thinking fields) rather than emitting a disabling 0.
func TestGeminiThinkingBudgetUnparseableTreatedAsUnset(t *testing.T) {
	body, err := newGoogle(t, "gemini-2.5-flash").PrepareRequest("hi", map[string]interface{}{"thinking_budget": "lots"})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeGoogleRequest(t, body)
	if _, ok := req["extra_body"]; ok {
		t.Errorf("unparseable budget must not emit extra_body, got %v", req["extra_body"])
	}
}

// TestGeminiThinkingBudgetDisableAndDynamic verifies the sentinels (0 disable, -1
// dynamic) survive on flash (which can disable thinking).
func TestGeminiThinkingBudgetDisableAndDynamic(t *testing.T) {
	for _, budget := range []int{0, -1} {
		body, err := newGoogle(t, "gemini-2.5-flash").PrepareRequest("hi", map[string]interface{}{"thinking_budget": budget})
		if err != nil {
			t.Fatalf("budget=%d: PrepareRequest failed: %v", budget, err)
		}
		cfg := geminiThinkingConfig(t, decodeGoogleRequest(t, body))
		if got := mustFloat(t, cfg["thinking_budget"], "thinking_budget"); int(got) != budget {
			t.Errorf("budget=%d: expected thinking_budget=%d, got %v", budget, budget, got)
		}
	}
}

// TestGeminiBudgetClampedForPro verifies a disable/too-small budget is raised to
// the 128-token floor on a 2.5-pro model, which cannot disable thinking.
func TestGeminiBudgetClampedForPro(t *testing.T) {
	for _, budget := range []int{0, 50} {
		body, err := newGoogle(t, "gemini-2.5-pro").PrepareRequest("hi", map[string]interface{}{"thinking_budget": budget})
		if err != nil {
			t.Fatalf("budget=%d: PrepareRequest failed: %v", budget, err)
		}
		cfg := geminiThinkingConfig(t, decodeGoogleRequest(t, body))
		if mustFloat(t, cfg["thinking_budget"], "thinking_budget") != 128 {
			t.Errorf("budget=%d: expected clamp to 128 on pro, got %v", budget, cfg["thinking_budget"])
		}
	}
}

// TestGeminiEffortNoneClampedForPro verifies reasoning_effort "none" is raised to
// "low" on a pro tier, which cannot turn thinking off.
func TestGeminiEffortNoneClampedForPro(t *testing.T) {
	for _, model := range []string{"gemini-2.5-pro", "gemini-3-pro-preview"} {
		body, err := newGoogle(t, model).PrepareRequest("hi", map[string]interface{}{"reasoning_effort": "none"})
		if err != nil {
			t.Fatalf("%s: PrepareRequest failed: %v", model, err)
		}
		req := decodeGoogleRequest(t, body)
		if req["reasoning_effort"] != "low" {
			t.Errorf("%s: expected none clamped to low, got %v", model, req["reasoning_effort"])
		}
	}
}

// TestGeminiThinkingLevel3x verifies thinking_level routes through extra_body as
// the native Gemini 3 control.
func TestGeminiThinkingLevel3x(t *testing.T) {
	body, err := newGoogle(t, "gemini-3-flash-preview").PrepareRequest("hi", map[string]interface{}{"thinking_level": "medium"})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeGoogleRequest(t, body)
	cfg := geminiThinkingConfig(t, req)
	if cfg["thinking_level"] != "medium" {
		t.Errorf("expected thinking_level=medium, got %v", cfg["thinking_level"])
	}
	if _, hasBudget := cfg["thinking_budget"]; hasBudget {
		t.Errorf("Gemini 3 must not receive a thinking_budget, got %v", cfg["thinking_budget"])
	}
}

// TestGemini3ProLevelClamp verifies pro tiers clamp the levels they do not
// support (minimal->low, medium->high) while flash keeps the full set, including
// the gemini-pro-latest alias.
func TestGemini3ProLevelClamp(t *testing.T) {
	cases := []struct{ model, in, want string }{
		{"gemini-3-pro-preview", "minimal", "low"},
		{"gemini-3-pro-preview", "medium", "high"},
		{"gemini-3-pro-preview", "high", "high"},
		{"gemini-pro-latest", "medium", "high"},
		{"gemini-3-flash-preview", "minimal", "minimal"},
		{"gemini-3-flash-preview", "medium", "medium"},
	}
	for _, c := range cases {
		body, err := newGoogle(t, c.model).PrepareRequest("hi", map[string]interface{}{"thinking_level": c.in})
		if err != nil {
			t.Fatalf("%s/%s: PrepareRequest failed: %v", c.model, c.in, err)
		}
		cfg := geminiThinkingConfig(t, decodeGoogleRequest(t, body))
		if cfg["thinking_level"] != c.want {
			t.Errorf("%s: level %q expected %q, got %v", c.model, c.in, c.want, cfg["thinking_level"])
		}
	}
}

// TestGeminiBudgetOn3xTranslatesToLevel verifies a 2.5-style thinking_budget on a
// Gemini 3 model is cross-translated to a thinking_level (3.x rejects budgets).
func TestGeminiBudgetOn3xTranslatesToLevel(t *testing.T) {
	cases := map[int]string{0: "minimal", 2048: "low", 8192: "medium", 30000: "high"}
	for budget, want := range cases {
		body, err := newGoogle(t, "gemini-3-flash-preview").PrepareRequest("hi", map[string]interface{}{"thinking_budget": budget})
		if err != nil {
			t.Fatalf("budget=%d: PrepareRequest failed: %v", budget, err)
		}
		cfg := geminiThinkingConfig(t, decodeGoogleRequest(t, body))
		if cfg["thinking_level"] != want {
			t.Errorf("budget=%d: expected thinking_level=%q, got %v", budget, want, cfg["thinking_level"])
		}
		if _, hasBudget := cfg["thinking_budget"]; hasBudget {
			t.Errorf("budget=%d: raw thinking_budget must not be sent to Gemini 3", budget)
		}
	}
}

// TestGeminiDynamicBudgetOn3xLeavesDefault verifies a dynamic (-1) budget on a
// Gemini 3 model emits no explicit control, leaving the model's default in place.
func TestGeminiDynamicBudgetOn3xLeavesDefault(t *testing.T) {
	body, err := newGoogle(t, "gemini-3-flash-preview").PrepareRequest("hi", map[string]interface{}{"thinking_budget": -1})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeGoogleRequest(t, body)
	if _, ok := req["extra_body"]; ok {
		t.Errorf("dynamic budget on Gemini 3 must not emit extra_body, got %v", req["extra_body"])
	}
}

// TestGeminiLevelOn2xTranslatesToBudget verifies a Gemini 3-style thinking_level
// on a 2.5 model is cross-translated to a thinking_budget, and that "none" maps
// to 0 (disable) on flash rather than a nonzero budget.
func TestGeminiLevelOn2xTranslatesToBudget(t *testing.T) {
	cases := map[string]float64{"none": 0, "minimal": 512, "low": 4096, "medium": 8192, "high": 16384}
	for level, want := range cases {
		body, err := newGoogle(t, "gemini-2.5-flash").PrepareRequest("hi", map[string]interface{}{"thinking_level": level})
		if err != nil {
			t.Fatalf("level=%s: PrepareRequest failed: %v", level, err)
		}
		cfg := geminiThinkingConfig(t, decodeGoogleRequest(t, body))
		if mustFloat(t, cfg["thinking_budget"], "thinking_budget") != want {
			t.Errorf("level=%s: expected thinking_budget=%v, got %v", level, want, cfg["thinking_budget"])
		}
		if _, hasLevel := cfg["thinking_level"]; hasLevel {
			t.Errorf("level=%s: raw thinking_level must not be sent to Gemini 2.5", level)
		}
	}
}

// TestGeminiIncludeThoughts verifies include_thoughts routes through
// thinking_config and coexists with a top-level reasoning_effort.
func TestGeminiIncludeThoughts(t *testing.T) {
	body, err := newGoogle(t, "gemini-2.5-flash").PrepareRequest("hi", map[string]interface{}{
		"reasoning_effort": "low", "include_thoughts": true,
	})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeGoogleRequest(t, body)
	if req["reasoning_effort"] != "low" {
		t.Errorf("expected reasoning_effort=low, got %v", req["reasoning_effort"])
	}
	cfg := geminiThinkingConfig(t, req)
	if cfg["include_thoughts"] != true {
		t.Errorf("expected include_thoughts=true, got %v", cfg["include_thoughts"])
	}
	if _, hasBudget := cfg["thinking_budget"]; hasBudget {
		t.Errorf("include_thoughts alone must not add a thinking_budget, got %v", cfg["thinking_budget"])
	}
}

// TestGeminiIncludeThoughtsNonBoolIgnored verifies a non-bool include_thoughts is
// ignored (treated as unset) rather than perturbing the request.
func TestGeminiIncludeThoughtsNonBoolIgnored(t *testing.T) {
	body, err := newGoogle(t, "gemini-2.5-flash").PrepareRequest("hi", map[string]interface{}{"include_thoughts": "true"})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeGoogleRequest(t, body)
	if _, ok := req["extra_body"]; ok {
		t.Errorf("non-bool include_thoughts must be ignored, got %v", req["extra_body"])
	}
}

// TestGeminiIncludeThoughtsFalseIsNoop verifies include_thoughts:false emits
// nothing (not a bare thinking_config).
func TestGeminiIncludeThoughtsFalseIsNoop(t *testing.T) {
	body, err := newGoogle(t, "gemini-2.5-flash").PrepareRequest("hi", map[string]interface{}{"include_thoughts": false})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeGoogleRequest(t, body)
	if _, ok := req["extra_body"]; ok {
		t.Errorf("include_thoughts:false must not emit extra_body, got %v", req["extra_body"])
	}
}

// TestGeminiThinkingProviderLevelBudgetNotLeakedTopLevel verifies a provider-level
// thinking_budget does not leak as an invalid top-level field.
func TestGeminiThinkingProviderLevelBudgetNotLeakedTopLevel(t *testing.T) {
	p := newGoogle(t, "gemini-2.5-flash")
	p.SetOption("thinking_budget", 1500)

	body, err := p.PrepareRequest("hi", map[string]interface{}{})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeGoogleRequest(t, body)
	if _, leaked := req["thinking_budget"]; leaked {
		t.Errorf("thinking_budget must not appear as a top-level field, got %v", req["thinking_budget"])
	}
	cfg := geminiThinkingConfig(t, req)
	if mustFloat(t, cfg["thinking_budget"], "thinking_budget") != 1500 {
		t.Errorf("expected provider-level thinking_budget=1500 in extra_body, got %v", cfg["thinking_budget"])
	}
}

// TestGeminiThinkingOffByDefault verifies no thinking fields are emitted when no
// thinking option is set.
func TestGeminiThinkingOffByDefault(t *testing.T) {
	body, err := newGoogle(t, "gemini-2.5-flash").PrepareRequest("hi", map[string]interface{}{})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeGoogleRequest(t, body)
	if _, ok := req["reasoning_effort"]; ok {
		t.Errorf("reasoning_effort must be absent by default")
	}
	if _, ok := req["extra_body"]; ok {
		t.Errorf("extra_body must be absent by default")
	}
}

// TestGeminiThinkingSkippedForNonThinkingModel verifies thinking controls are not
// sent to the non-thinking base models (1.x / 2.0), and leaked keys are stripped.
func TestGeminiThinkingSkippedForNonThinkingModel(t *testing.T) {
	for _, model := range []string{"gemini-2.0-flash", "gemini-1.5-pro"} {
		body, err := newGoogle(t, model).PrepareRequest("hi", map[string]interface{}{
			"reasoning_effort": "high", "thinking_budget": 2048,
		})
		if err != nil {
			t.Fatalf("%s: PrepareRequest failed: %v", model, err)
		}
		req := decodeGoogleRequest(t, body)
		if _, ok := req["reasoning_effort"]; ok {
			t.Errorf("%s: reasoning_effort must not be sent to a non-thinking model", model)
		}
		if _, ok := req["extra_body"]; ok {
			t.Errorf("%s: extra_body must not be sent to a non-thinking model", model)
		}
		if _, ok := req["thinking_budget"]; ok {
			t.Errorf("%s: leaked thinking_budget must be stripped", model)
		}
	}
}

// TestGeminiLatestAliasesRouteTo3x verifies unversioned "-latest" aliases are
// treated as the newest (3.x) generation: an explicit budget cross-translates to
// a thinking_level rather than being sent as a raw budget.
func TestGeminiLatestAliasesRouteTo3x(t *testing.T) {
	for _, model := range []string{"gemini-flash-latest", "gemini-pro-latest", "gemini-latest"} {
		body, err := newGoogle(t, model).PrepareRequest("hi", map[string]interface{}{"thinking_budget": 8192})
		if err != nil {
			t.Fatalf("%s: PrepareRequest failed: %v", model, err)
		}
		cfg := geminiThinkingConfig(t, decodeGoogleRequest(t, body))
		if _, hasLevel := cfg["thinking_level"]; !hasLevel {
			t.Errorf("%s: latest alias should route to a 3.x thinking_level, got %v", model, cfg)
		}
		if _, hasBudget := cfg["thinking_budget"]; hasBudget {
			t.Errorf("%s: latest alias must not receive a raw thinking_budget", model)
		}
	}
}

// TestGeminiUnversionedNonLatestDefaultsTo2x verifies an unversioned id that is
// not a "-latest" alias routes to the 2.x form (thinking_budget), not 3.x
// thinking_level.
func TestGeminiUnversionedNonLatestDefaultsTo2x(t *testing.T) {
	body, err := newGoogle(t, "gemini-exp").PrepareRequest("hi", map[string]interface{}{"thinking_budget": 4096})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	cfg := geminiThinkingConfig(t, decodeGoogleRequest(t, body))
	if _, hasLevel := cfg["thinking_level"]; hasLevel {
		t.Errorf("unversioned non-latest id must not route to 3.x thinking_level, got %v", cfg)
	}
	if mustFloat(t, cfg["thinking_budget"], "thinking_budget") != 4096 {
		t.Errorf("expected 2.x thinking_budget=4096, got %v", cfg["thinking_budget"])
	}
	if isGemini3Model("gemini-exp") {
		t.Errorf("gemini-exp (unversioned, non-latest) must not be treated as 3.x")
	}
}

// TestGeminiFractionalBudgetTreatedAsUnset verifies a fractional float budget is
// rejected (treated as unset) rather than truncated, while a whole float works.
func TestGeminiFractionalBudgetTreatedAsUnset(t *testing.T) {
	body, err := newGoogle(t, "gemini-2.5-flash").PrepareRequest("hi", map[string]interface{}{"thinking_budget": 2048.9})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	if _, ok := decodeGoogleRequest(t, body)["extra_body"]; ok {
		t.Errorf("fractional budget must be treated as unset, got extra_body")
	}

	body, err = newGoogle(t, "gemini-2.5-flash").PrepareRequest("hi", map[string]interface{}{"thinking_budget": 4096.0})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	cfg := geminiThinkingConfig(t, decodeGoogleRequest(t, body))
	if mustFloat(t, cfg["thinking_budget"], "thinking_budget") != 4096 {
		t.Errorf("whole float budget should convert cleanly, got %v", cfg["thinking_budget"])
	}
}

// TestGeminiPreservesCallerExtraBody verifies caller-supplied extra_body content
// (sibling google keys and existing thinking_config entries) survives thinking
// normalization, with the thinking field merged in.
func TestGeminiPreservesCallerExtraBody(t *testing.T) {
	body, err := newGoogle(t, "gemini-2.5-flash").PrepareRequest("hi", map[string]interface{}{
		"thinking_budget": 2048,
		"extra_body": map[string]interface{}{
			"google": map[string]interface{}{
				"custom_key":      "keep",
				"thinking_config": map[string]interface{}{"existing": "stay"},
			},
		},
	})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeGoogleRequest(t, body)
	google := mustObject(t, mustObject(t, req["extra_body"], "extra_body")["google"], "google")
	if google["custom_key"] != "keep" {
		t.Errorf("caller extra_body.google sibling key must survive, got %v", google["custom_key"])
	}
	cfg := mustObject(t, google["thinking_config"], "thinking_config")
	if cfg["existing"] != "stay" {
		t.Errorf("caller thinking_config entry must survive, got %v", cfg["existing"])
	}
	if mustFloat(t, cfg["thinking_budget"], "thinking_budget") != 2048 {
		t.Errorf("thinking_budget should be merged in, got %v", cfg["thinking_budget"])
	}
}

// TestGeminiMajorVersionParsing guards the version parser that routes families.
func TestGeminiMajorVersionParsing(t *testing.T) {
	cases := map[string]int{
		"gemini-2.5-flash": 2, "gemini-2.0-flash": 2, "gemini-3-pro-preview": 3,
		"gemini-3.5-flash": 3, "gemini-4-ultra": 4, "gemini-flash-latest": 0,
		"models/gemini-2.5-pro": 2, "not-a-gemini": 0,
	}
	for model, want := range cases {
		if got := geminiMajorVersion(model); got != want {
			t.Errorf("geminiMajorVersion(%q)=%d, want %d", model, got, want)
		}
	}
	if !isGemini3Model("gemini-flash-latest") {
		t.Errorf("unversioned latest alias should route to the 3.x form")
	}
	if isGemini3Model("gemini-2.5-flash") {
		t.Errorf("gemini-2.5 must not be treated as 3.x")
	}
}

// TestGeminiReasoningEffortEnumType verifies the published types.ReasoningEffort
// enum is accepted (not just a plain string) and mapped to Gemini's values.
func TestGeminiReasoningEffortEnumType(t *testing.T) {
	body, err := newGoogle(t, "gemini-2.5-flash").PrepareRequest("hi", map[string]interface{}{
		"reasoning_effort": types.ReasoningEffortMax, // xhigh/max clamp to high
	})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	req := decodeGoogleRequest(t, body)
	if req["reasoning_effort"] != "high" {
		t.Errorf("expected ReasoningEffortMax mapped to high, got %v", req["reasoning_effort"])
	}
}

// TestGeminiThinkingStreamAndMessagePaths verifies the thinking normalization is
// applied across every prepare entrypoint, since Go embedding does not dispatch
// the inherited methods to these overrides.
func TestGeminiThinkingStreamAndMessagePaths(t *testing.T) {
	msgs := []types.MemoryMessage{{Role: "user", Content: "hi"}}
	opts := map[string]interface{}{"thinking_budget": 800}

	check := func(name string, body []byte, err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("%s failed: %v", name, err)
		}
		cfg := geminiThinkingConfig(t, decodeGoogleRequest(t, body))
		if mustFloat(t, cfg["thinking_budget"], "thinking_budget") != 800 {
			t.Errorf("%s: expected thinking_budget=800, got %v", name, cfg["thinking_budget"])
		}
	}

	b, err := newGoogle(t, "gemini-2.5-flash").PrepareStreamRequest("hi", opts)
	check("PrepareStreamRequest", b, err)

	b, err = newGoogle(t, "gemini-2.5-flash").PrepareRequestWithMessages(msgs, opts)
	check("PrepareRequestWithMessages", b, err)

	b, err = newGoogle(t, "gemini-2.5-flash").PrepareStreamRequestWithMessages(msgs, opts)
	check("PrepareStreamRequestWithMessages", b, err)

	b, err = newGoogle(t, "gemini-2.5-flash").PrepareRequestWithSchema("hi", opts, map[string]interface{}{"type": "object"})
	check("PrepareRequestWithSchema", b, err)

	b, err = newGoogle(t, "gemini-2.5-flash").PrepareRequestWithMessagesAndSchema(msgs, opts, map[string]interface{}{"type": "object"})
	check("PrepareRequestWithMessagesAndSchema", b, err)
}
