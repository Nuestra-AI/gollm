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
	// LoadConfig reads the process environment, so a developer with LLM_MIN_P or
	// LLM_TOP_K exported would see this fail on correct code. t.Setenv also restores
	// the previous values when the test ends.
	for _, key := range []string{
		"LLM_TOP_K", "LLM_MIN_P", "LLM_REPEAT_PENALTY", "LLM_REPEAT_LAST_N",
		"LLM_MIROSTAT", "LLM_MIROSTAT_ETA", "LLM_MIROSTAT_TAU", "LLM_TFS_Z",
		"LLM_SEED", "LLM_STOP_SEQUENCES",
	} {
		t.Setenv(key, "")
		os.Unsetenv(key)
	}

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
