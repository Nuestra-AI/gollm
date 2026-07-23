package llm

import (
	"bytes"
	"encoding/json"

	"github.com/teilomillet/gollm/types"
)

// The usage observer types live in the types package so that config can carry an observer without
// importing llm (llm imports config, so the reverse would be a cycle). They are aliased here because
// this is where they are consumed, and so callers already holding llm types need no new import.
type (
	// UsageOutcome describes what became of the response a billed round-trip paid for.
	UsageOutcome = types.UsageOutcome
	// UsageEvent is one billed provider round-trip.
	UsageEvent = types.UsageEvent
	// UsageObserver is fired once per billed provider round-trip.
	UsageObserver = types.UsageObserver
)

// Usage outcomes; see types.UsageOutcome for what each one means.
const (
	UsageOutcomeSuccess       = types.UsageOutcomeSuccess
	UsageOutcomeSchemaFail    = types.UsageOutcomeSchemaFail
	UsageOutcomeParseFail     = types.UsageOutcomeParseFail
	UsageOutcomeStream        = types.UsageOutcomeStream
	UsageOutcomeStreamAborted = types.UsageOutcomeStreamAborted
)

// UsageObservable is the optional capability of accepting a usage observer after construction.
//
// It is deliberately NOT part of the LLM interface: requiring the method there would break every
// existing implementation of LLM downstream (mocks and fakes especially) for a capability most
// callers get from config.WithUsageObserver instead. Optional interfaces are how this package
// already handles provider capabilities — see richStreamParser and streamMessagesPreparer.
//
// SetUsageObserver reports whether the observer was actually installed. A leaf implementation
// always returns true; a wrapper that delegates to an inner LLM must forward its inner result,
// because the inner value is an interface that need not have the capability at all.
type UsageObservable interface {
	SetUsageObserver(observer UsageObserver) bool
}

// AttachUsageObserver installs a usage observer on an already-built client, reporting whether it
// was installed. Prefer config.WithUsageObserver, which needs no type assertion and reaches clients
// constructed internally by MOA and the assess harness; use this when the client already exists and
// its concrete type isn't known.
//
// A false result means nothing will be observed, and a caller that suppresses its own accounting
// when this returns true must keep that accounting when it returns false. The answer is exact
// through wrapper types, which forward the result from whatever they wrap rather than reporting
// their own ability to accept the call.
func AttachUsageObserver(client interface{}, observer UsageObserver) bool {
	observable, ok := client.(UsageObservable)
	if !ok {
		return false
	}
	return observable.SetUsageObserver(observer)
}

// UsageReporter is the optional capability of reporting a stream's accumulated token usage. Like
// UsageObservable it is kept off the TokenStream interface so external implementations keep
// compiling; the streams this package returns satisfy it.
type UsageReporter interface {
	Usage() types.TokenUsage
}

// StreamUsage returns a stream's accumulated token usage when the stream can report it. The count is
// final once Next has returned io.EOF and partial before that.
func StreamUsage(stream TokenStream) (types.TokenUsage, bool) {
	reporter, ok := stream.(UsageReporter)
	if !ok {
		return types.TokenUsage{}, false
	}
	return reporter.Usage(), true
}

// wireUsage covers every usage shape the supported providers emit, so usage can
// be recovered straight from a response body when the provider's own parser
// failed or reports none. Unknown keys unmarshal to zero, so one struct can
// carry all shapes without knowing which provider sent the body.
type wireUsage struct {
	Usage struct {
		// OpenAI-compatible (OpenAI, Groq, Mistral, vLLM, Lambda, OpenRouter, …).
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		TotalTokens         int `json:"total_tokens"`
		PromptTokensDetails struct {
			CachedTokens     int `json:"cached_tokens"`
			CacheWriteTokens int `json:"cache_write_tokens"`
			AudioTokens      int `json:"audio_tokens"`
		} `json:"prompt_tokens_details"`
		CompletionTokensDetails struct {
			ReasoningTokens          int `json:"reasoning_tokens"`
			AudioTokens              int `json:"audio_tokens"`
			AcceptedPredictionTokens int `json:"accepted_prediction_tokens"`
			RejectedPredictionTokens int `json:"rejected_prediction_tokens"`
		} `json:"completion_tokens_details"`

		// Anthropic / Bedrock-Anthropic / OpenAI Responses.
		InputTokens              int `json:"input_tokens"`
		OutputTokens             int `json:"output_tokens"`
		CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
		CacheReadInputTokens     int `json:"cache_read_input_tokens"`
		InputTokensDetails       struct {
			CachedTokens     int `json:"cached_tokens"`
			CacheWriteTokens int `json:"cache_write_tokens"`
			AudioTokens      int `json:"audio_tokens"`
		} `json:"input_tokens_details"`
		OutputTokensDetails struct {
			ReasoningTokens int `json:"reasoning_tokens"`
			AudioTokens     int `json:"audio_tokens"`
			// Anthropic's name for the same thing.
			ThinkingTokens int `json:"thinking_tokens"`
		} `json:"output_tokens_details"`
		// Anthropic cache writes split by lifetime: 1.25x for 5m, 2x for 1h.
		CacheCreation struct {
			Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens"`
			Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens"`
		} `json:"cache_creation"`
		// Scales the price of everything above; not itself a token count.
		ServiceTier string `json:"service_tier"`

		// Cohere v2 nests the real counts one level down.
		Tokens struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"tokens"`
	} `json:"usage"`

	// Ollama reports counts at the top level of its final object.
	PromptEvalCount int `json:"prompt_eval_count"`
	EvalCount       int `json:"eval_count"`

	// Bedrock non-Anthropic families.
	PromptTokenCount     int `json:"prompt_token_count"`
	GenerationTokenCount int `json:"generation_token_count"`

	// OpenAI reports the served tier at the top level; Anthropic nests it in usage (above).
	ServiceTier string `json:"service_tier"`
}

// ExtractUsage recovers token usage directly from a raw provider response body,
// independent of that provider's content parser.
//
// This is the safety net for the paths where the parser cannot help: a billed
// 200 whose content failed to parse, and providers that return no usage detail
// of their own. It accepts JSON and JSONL (using the last line, which carries
// the terminal object), and reports whether anything was found.
func ExtractUsage(body []byte) (types.TokenUsage, bool) {
	usage, _, ok := ExtractUsageAndTier(body)
	return usage, ok
}

// ExtractUsageAndTier recovers token usage and the served tier from a raw response body in a single
// pass. Both come from the same JSON, so a caller that wants both should ask for both rather than
// parse the body twice — decoding scans the whole document however few fields are wanted, and this
// runs on the generation path.
func ExtractUsageAndTier(body []byte) (types.TokenUsage, string, bool) {
	if usage, tier, ok := extractUsageJSON(body); ok {
		return usage, tier, true
	}
	// JSONL (notably Ollama's streamed-to-completion form): the final object is
	// the one carrying totals.
	lines := bytes.Split(bytes.TrimSpace(body), []byte("\n"))
	for i := len(lines) - 1; i >= 0; i-- {
		if usage, tier, ok := extractUsageJSON(lines[i]); ok {
			return usage, tier, true
		}
	}
	// No counts anywhere; the tier may still be present on a body that reported no usage.
	return types.TokenUsage{}, ExtractServiceTier(body), false
}

// ExtractServiceTier recovers the served tier from a raw response body, for the paths where the
// provider's own parser produced no details. The tier multiplies the price of every token in the
// response — batch is roughly half rate, priority a premium — so a usage record recovered without
// it can be exactly right about counts and still wrong about cost.
//
// Prefer ExtractUsageAndTier when the counts are wanted too: this parses the body on its own.
func ExtractServiceTier(body []byte) string {
	if tier := serviceTierJSON(body); tier != "" {
		return tier
	}
	lines := bytes.Split(bytes.TrimSpace(body), []byte("\n"))
	for i := len(lines) - 1; i >= 0; i-- {
		if tier := serviceTierJSON(lines[i]); tier != "" {
			return tier
		}
	}
	return ""
}

func serviceTierJSON(body []byte) string {
	var w struct {
		ServiceTier string `json:"service_tier"`
		Usage       struct {
			ServiceTier string `json:"service_tier"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(bytes.TrimSpace(body), &w); err != nil {
		return ""
	}
	// OpenAI reports it at the top level, Anthropic inside usage.
	if w.ServiceTier != "" {
		return w.ServiceTier
	}
	return w.Usage.ServiceTier
}

func extractUsageJSON(body []byte) (types.TokenUsage, string, bool) {
	var w wireUsage
	if err := json.Unmarshal(bytes.TrimSpace(body), &w); err != nil {
		return types.TokenUsage{}, "", false
	}
	// OpenAI reports the tier at the top level, Anthropic inside usage.
	tier := w.ServiceTier
	if tier == "" {
		tier = w.Usage.ServiceTier
	}

	u := types.TokenUsage{
		PromptTokens:             w.Usage.PromptTokens,
		CompletionTokens:         w.Usage.CompletionTokens,
		TotalTokens:              w.Usage.TotalTokens,
		CacheCreationInputTokens: w.Usage.CacheCreationInputTokens,
		CacheReadInputTokens:     w.Usage.CacheReadInputTokens,
		CachedPromptTokens:       w.Usage.PromptTokensDetails.CachedTokens + w.Usage.InputTokensDetails.CachedTokens,
		ReasoningTokens: w.Usage.CompletionTokensDetails.ReasoningTokens +
			w.Usage.OutputTokensDetails.ReasoningTokens +
			w.Usage.OutputTokensDetails.ThinkingTokens,
		CacheCreation5mInputTokens: w.Usage.CacheCreation.Ephemeral5mInputTokens,
		CacheCreation1hInputTokens: w.Usage.CacheCreation.Ephemeral1hInputTokens,
		CacheWritePromptTokens:     w.Usage.PromptTokensDetails.CacheWriteTokens + w.Usage.InputTokensDetails.CacheWriteTokens,
		AudioPromptTokens:          w.Usage.PromptTokensDetails.AudioTokens + w.Usage.InputTokensDetails.AudioTokens,
		AudioCompletionTokens:      w.Usage.CompletionTokensDetails.AudioTokens + w.Usage.OutputTokensDetails.AudioTokens,
		AcceptedPredictionTokens:   w.Usage.CompletionTokensDetails.AcceptedPredictionTokens,
		RejectedPredictionTokens:   w.Usage.CompletionTokensDetails.RejectedPredictionTokens,
	}
	// Anthropic sends either the aggregate or the per-lifetime split depending on
	// version; derive the aggregate so cache writes are never invisible.
	if u.CacheCreationInputTokens == 0 {
		u.CacheCreationInputTokens = u.CacheCreation5mInputTokens + u.CacheCreation1hInputTokens
	}

	// Fill the normalized input/output counts from whichever naming the body used.
	if u.PromptTokens == 0 {
		switch {
		case w.Usage.InputTokens > 0:
			u.PromptTokens = w.Usage.InputTokens
		case w.Usage.Tokens.InputTokens > 0:
			u.PromptTokens = w.Usage.Tokens.InputTokens
		case w.PromptEvalCount > 0:
			u.PromptTokens = w.PromptEvalCount
		case w.PromptTokenCount > 0:
			u.PromptTokens = w.PromptTokenCount
		}
	}
	if u.CompletionTokens == 0 {
		switch {
		case w.Usage.OutputTokens > 0:
			u.CompletionTokens = w.Usage.OutputTokens
		case w.Usage.Tokens.OutputTokens > 0:
			u.CompletionTokens = w.Usage.Tokens.OutputTokens
		case w.EvalCount > 0:
			u.CompletionTokens = w.EvalCount
		case w.GenerationTokenCount > 0:
			u.CompletionTokens = w.GenerationTokenCount
		}
	}
	// Same definition the provider parsers use, so a cached call's total does not depend on
	// whether its content happened to parse.
	u.TotalTokens = u.ComputedTotal()

	return u, tier, !u.IsZero()
}
