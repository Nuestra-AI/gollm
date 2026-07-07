package types

// ReasoningEffort is the canonical, cross-provider reasoning/"thinking" effort
// level. Pass one to SetOption("reasoning_effort", ...) (a plain string works
// too). Each provider maps these onto the values its API actually accepts:
//
//   - OpenAI:    low/medium/high (GPT-5 also "minimal"); xhigh/max clamp to high.
//   - Anthropic: low/medium/high/xhigh/max (xhigh demoted to high on models that
//     predate it).
//   - Gemini:    none/low/medium/high; xhigh/max clamp to high. "none" disables
//     thinking where the model allows it.
//
// The values form an ascending scale; not every provider honors every level, so
// the ends (none, xhigh, max) are clamped rather than rejected.
type ReasoningEffort string

const (
	ReasoningEffortNone   ReasoningEffort = "none"
	ReasoningEffortLow    ReasoningEffort = "low"
	ReasoningEffortMedium ReasoningEffort = "medium"
	ReasoningEffortHigh   ReasoningEffort = "high"
	ReasoningEffortXHigh  ReasoningEffort = "xhigh"
	ReasoningEffortMax    ReasoningEffort = "max"
)

// String returns the underlying string value.
func (e ReasoningEffort) String() string { return string(e) }
