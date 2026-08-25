package config

import (
	"os"
	"testing"
)

// TestOptionalSamplingParamsDefaultToUnset guards the envDefault removal. These are
// pointers so that "unset" is distinguishable from a chosen value; an envDefault
// made every LoadConfig look like an explicit request for min_p 0.05 and
// repeat_penalty 1.1, which providers would then put on every request.
func TestOptionalSamplingParamsDefaultToUnset(t *testing.T) {
	unsetEnv(t,
		"LLM_TOP_K", "LLM_MIN_P", "LLM_REPEAT_PENALTY", "LLM_REPEAT_LAST_N",
		"LLM_MIROSTAT", "LLM_MIROSTAT_ETA", "LLM_MIROSTAT_TAU", "LLM_TFS_Z",
		"LLM_SEED", "LLM_STOP_SEQUENCES",
	)

	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig failed: %v", err)
	}

	if cfg.MinP != nil {
		t.Errorf("MinP = %v, want nil", *cfg.MinP)
	}
	if cfg.RepeatPenalty != nil {
		t.Errorf("RepeatPenalty = %v, want nil", *cfg.RepeatPenalty)
	}
	if cfg.RepeatLastN != nil {
		t.Errorf("RepeatLastN = %v, want nil", *cfg.RepeatLastN)
	}
	if cfg.Mirostat != nil {
		t.Errorf("Mirostat = %v, want nil", *cfg.Mirostat)
	}
	if cfg.MirostatEta != nil {
		t.Errorf("MirostatEta = %v, want nil", *cfg.MirostatEta)
	}
	if cfg.MirostatTau != nil {
		t.Errorf("MirostatTau = %v, want nil", *cfg.MirostatTau)
	}
	if cfg.TfsZ != nil {
		t.Errorf("TfsZ = %v, want nil", *cfg.TfsZ)
	}
	if cfg.TopK != nil {
		t.Errorf("TopK = %v, want nil", *cfg.TopK)
	}
	if cfg.StopSequences != nil {
		t.Errorf("StopSequences = %v, want nil", cfg.StopSequences)
	}
}

// TestSettersMarkParamsAsSet verifies the setters are what turns a parameter on.
func TestSettersMarkParamsAsSet(t *testing.T) {
	cfg := &Config{}
	for _, apply := range []ConfigOption{
		SetTopK(40), SetMinP(0.05), SetRepeatPenalty(1.1), SetRepeatLastN(64),
		SetMirostat(2), SetMirostatEta(0.1), SetMirostatTau(5.0), SetTfsZ(1.0),
		SetStopSequences("END"),
	} {
		apply(cfg)
	}

	if cfg.TopK == nil || *cfg.TopK != 40 {
		t.Errorf("SetTopK did not take: %v", cfg.TopK)
	}
	if cfg.MinP == nil || *cfg.MinP != 0.05 {
		t.Errorf("SetMinP did not take: %v", cfg.MinP)
	}
	if cfg.Mirostat == nil || *cfg.Mirostat != 2 {
		t.Errorf("SetMirostat did not take: %v", cfg.Mirostat)
	}
	if cfg.TfsZ == nil || *cfg.TfsZ != 1.0 {
		t.Errorf("SetTfsZ did not take: %v", cfg.TfsZ)
	}
	if len(cfg.StopSequences) != 1 || cfg.StopSequences[0] != "END" {
		t.Errorf("SetStopSequences did not take: %v", cfg.StopSequences)
	}
	// Calling it with nothing clears the setting rather than storing an empty list.
	SetStopSequences()(cfg)
	if cfg.StopSequences != nil {
		t.Errorf("SetStopSequences() should clear, got %v", cfg.StopSequences)
	}
}

// TestDefaultModelIsNotRetired guards the shipped default. It was
// claude-3-5-haiku-latest, which Anthropic retired on 2026-02-19 — every request
// from a caller who never set a model failed outright.
func TestDefaultModelIsNotRetired(t *testing.T) {
	unsetEnv(t, "LLM_MODEL")

	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig failed: %v", err)
	}
	if cfg.Model == "" {
		t.Fatal("default model is empty")
	}
	if retiredAnthropicModels[cfg.Model] {
		t.Errorf("default model %q is retired; requests to it fail", cfg.Model)
	}
}

// retiredAnthropicModels are the ids and aliases Anthropic has retired, from
// platform.claude.com/docs/en/about-claude/model-deprecations (2026-08-25).
// Matched exactly rather than by substring: "claude-2" is a substring of a
// plausible future id, and a false positive here would fail a correct default.
var retiredAnthropicModels = map[string]bool{
	"claude-3-5-haiku-latest": true, "claude-3-5-haiku-20241022": true,
	"claude-3-5-sonnet-latest": true, "claude-3-5-sonnet-20241022": true,
	"claude-3-5-sonnet-20240620": true,
	"claude-3-opus-latest":       true, "claude-3-opus-20240229": true,
	"claude-3-sonnet-20240229": true, "claude-3-haiku-20240307": true,
	"claude-3-7-sonnet-latest": true, "claude-3-7-sonnet-20250219": true,
	"claude-2.0": true, "claude-2.1": true, "claude-instant-1.2": true,
	"claude-opus-4-20250514": true, "claude-sonnet-4-20250514": true,
	"claude-opus-4-1-20250805": true,
}

// unsetEnv removes variables for the duration of the test and restores whatever
// was there afterwards. LoadConfig reads the process environment, so a developer
// with LLM_MODEL or LLM_MIN_P exported would otherwise see these fail on correct
// code. Unsetting rather than blanking matters: env.Parse treats an empty string
// as a value to parse, not as absence.
func unsetEnv(t *testing.T, keys ...string) {
	t.Helper()
	for _, key := range keys {
		if original, ok := os.LookupEnv(key); ok {
			t.Cleanup(func() { os.Setenv(key, original) })
		}
		os.Unsetenv(key)
	}
}
