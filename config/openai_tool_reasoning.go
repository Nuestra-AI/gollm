package config

// ToolReasoningPolicy selects how a request pairing function tools with reasoning
// is served on OpenAI, which /v1/chat/completions rejects from gpt-5.4 onward.
// Applies only to models carrying that restriction; ones Chat Completions cannot
// serve at all are routed regardless.
type ToolReasoningPolicy string

const (
	// ToolReasoningPreferSpeed keeps Chat Completions, sending reasoning_effort
	// "none" when tools are present. The default: /v1/responses is several times
	// slower. The provider logs when reasoning is dropped.
	ToolReasoningPreferSpeed ToolReasoningPolicy = "prefer-speed"

	// ToolReasoningPreferQuality serves affected models on /v1/responses, the only
	// transport accepting tools and reasoning together, at that latency cost.
	ToolReasoningPreferQuality ToolReasoningPolicy = "prefer-quality"
)

// WithOpenAIToolReasoning sets how requests combining function tools with reasoning
// are served; see ToolReasoningPolicy. Empty or unrecognized means
// ToolReasoningPreferSpeed. Resolved when the client is built, since it selects the
// transport — callers needing both behaviors should build one client of each.
func WithOpenAIToolReasoning(policy ToolReasoningPolicy) ConfigOption {
	return func(c *Config) {
		c.ToolReasoning = policy
	}
}
