package config

// ToolReasoningPolicy selects how a request that pairs function tools with
// reasoning is served on OpenAI.
//
// From gpt-5.4 onward, /v1/chat/completions rejects that combination outright:
//
//	Function tools with reasoning_effort are not supported for gpt-5.6-sol in
//	/v1/chat/completions. To use function tools, use /v1/responses or set
//	reasoning_effort to 'none'
//
// Only two things satisfy the API, and they trade against each other — keep the
// fast transport and give up reasoning, or keep reasoning and pay for the slower
// one. Which is right depends on what the caller is building, so it is a policy
// rather than a default the library can pick on their behalf.
//
// The policy applies only to the models that actually carry the restriction.
// Anything else — gpt-4o, the o-series, gpt-5.3 and earlier — is unaffected by
// either setting, and models that /v1/chat/completions cannot serve at all are
// routed to /v1/responses regardless.
type ToolReasoningPolicy string

const (
	// ToolReasoningPreferSpeed keeps requests on /v1/chat/completions and sends
	// reasoning_effort "none" when function tools are present. This is the
	// default: /v1/responses measures several times slower in independent
	// benchmarks, and most callers would rather have the latency than reasoning
	// on a tool-dispatch turn. The provider logs when reasoning is given up.
	ToolReasoningPreferSpeed ToolReasoningPolicy = "prefer-speed"

	// ToolReasoningPreferQuality serves affected models on /v1/responses, which
	// is the only transport that accepts function tools and reasoning together.
	// Choose it when the model's reasoning is doing real work on tool-calling
	// turns — deciding which tool to call, or interpreting a result — and the
	// added latency is worth it.
	ToolReasoningPreferQuality ToolReasoningPolicy = "prefer-quality"
)

// WithOpenAIToolReasoning sets how requests combining function tools with
// reasoning are served. See ToolReasoningPolicy. An empty or unrecognized value
// is treated as ToolReasoningPreferSpeed.
//
// This is a per-client policy resolved when the client is built, not a per-request
// switch: it selects the transport, and a client speaks to one endpoint for its
// lifetime. Callers who need both behaviors should build a client for each.
func WithOpenAIToolReasoning(policy ToolReasoningPolicy) ConfigOption {
	return func(c *Config) {
		c.ToolReasoning = policy
	}
}
