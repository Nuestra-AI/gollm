package gollm

import "github.com/teilomillet/gollm/types"

// ReasoningEffort is the canonical, cross-provider reasoning/"thinking" effort
// level passed via SetOption("reasoning_effort", ...). It is an alias for
// types.ReasoningEffort; see that type for the per-provider value mapping.
type ReasoningEffort = types.ReasoningEffort

// Reasoning effort levels, ascending. Not every provider honors every level; the
// ends (none, xhigh, max) are clamped to the nearest supported value rather than
// rejected. See types.ReasoningEffort for the per-provider mapping.
const (
	ReasoningEffortNone   = types.ReasoningEffortNone
	ReasoningEffortLow    = types.ReasoningEffortLow
	ReasoningEffortMedium = types.ReasoningEffortMedium
	ReasoningEffortHigh   = types.ReasoningEffortHigh
	ReasoningEffortXHigh  = types.ReasoningEffortXHigh
	ReasoningEffortMax    = types.ReasoningEffortMax
)
