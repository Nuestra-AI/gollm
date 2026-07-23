package types

// ReasoningEffort is the canonical, cross-provider reasoning/"thinking" effort
// level. Pass one to SetOption("reasoning_effort", ...) (a plain string works
// too). Each provider maps these onto the values its API actually accepts:
//
//   - OpenAI:    low/medium/high everywhere; "none" on gpt-5.1+, "minimal" on the original
//     GPT-5 models, "xhigh" on gpt-5.4+ and gpt-5.1-codex-max, "max" on gpt-5.6+. Levels the
//     target model doesn't accept are clamped inward to the nearest one it does.
//   - Anthropic: low/medium/high/xhigh/max (xhigh demoted to high on models that
//     predate it).
//   - Gemini:    none/low/medium/high; xhigh/max clamp to high. "none" disables
//     thinking where the model allows it.
//
// The values form an ascending scale; not every provider honors every level, so
// the ends (none, xhigh, max) are clamped rather than rejected.
type ReasoningEffort string

const (
	ReasoningEffortNone ReasoningEffort = "none"
	// ReasoningEffortMinimal sits between none and low: the model reasons as little as it
	// can rather than not at all. OpenAI-only, and only on the original GPT-5 models.
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
	ReasoningEffortMax     ReasoningEffort = "max"
)

// String returns the underlying string value.
func (e ReasoningEffort) String() string { return string(e) }
