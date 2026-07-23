package types

import "context"

// UsageOutcome describes what happened to the response a billed round-trip paid
// for. Tokens are charged the moment the provider produces a response, so every
// outcome below — not just Success — represents money spent.
type UsageOutcome string

const (
	// UsageOutcomeSuccess: the response was parsed, validated, and returned.
	UsageOutcomeSuccess UsageOutcome = "success"
	// UsageOutcomeSchemaFail: the response was billed and parsed, then discarded
	// because it did not satisfy the caller's JSON schema. The retry loop will
	// pay for another attempt.
	UsageOutcomeSchemaFail UsageOutcome = "schema_fail"
	// UsageOutcomeParseFail: the provider returned a billed 200 that could not be
	// parsed into content — most commonly a max_tokens-truncated response with no
	// content blocks, which is charged for the full input plus whatever output was
	// generated. Usage here comes from the raw body, not from the provider parser.
	UsageOutcomeParseFail UsageOutcome = "parse_fail"
	// UsageOutcomeStream: a stream ran to completion; usage is the accumulated
	// total across its chunks.
	UsageOutcomeStream UsageOutcome = "stream"
	// UsageOutcomeStreamAborted: a stream ended before reaching its terminal event —
	// closed early by the consumer, cancelled via its context, or broken by a
	// mid-stream error. Whatever was generated before the abort is still billed,
	// though the provider may never have sent a usage chunk, so Usage can be partial
	// or zero.
	//
	// This outcome depends on the stream being ended rather than abandoned. A stream
	// that is neither read to completion, nor cancelled, nor Closed has no moment at
	// which to report, and its tokens are billed but never observed — which is why
	// Close is required of every caller, not merely recommended.
	UsageOutcomeStreamAborted UsageOutcome = "stream_aborted"
)

// UsageEvent is one billed provider round-trip.
//
// Usage may be zero when the provider reports no usage at all (some local
// backends) or when a stream was abandoned before its usage chunk arrived. That
// is deliberately still delivered: a call that cannot be accounted for is itself
// worth recording, and an event count is a lower bound on requests billed.
//
// It lives in types rather than llm so that config — which llm imports — can
// carry an observer without an import cycle. That is what lets a single observer
// be attached at configuration time and travel into every client built from it,
// including the ones MOA and the assess harness construct internally.
type UsageEvent struct {
	// Provider is the provider name (e.g. "anthropic", "openai").
	Provider string
	// Model is the model reported by the provider when available, falling back to
	// the configured model. Prefer this over the caller's own configuration —
	// gateways and aliases (e.g. OpenRouter, "-latest" tags) resolve to a
	// different, differently-priced model than the one requested.
	Model string
	// Outcome says what became of the response; see the UsageOutcome constants.
	Outcome UsageOutcome
	// Attempt is the zero-based index of this attempt within the retry loop, so a
	// recorder can tell a first-try success from the third paid attempt.
	Attempt int
	// Usage is the token accounting for this round-trip alone, never a running
	// total across attempts. Sum across events to cost a whole operation.
	Usage TokenUsage
	// ServiceTier is the tier the provider served this request on ("standard",
	// "priority", "batch", …). It is not a token count but it scales the price of
	// every token in Usage, so a recorder that ignores it can be wrong by 2x on
	// counts that are exactly right. Empty when the provider reports none.
	ServiceTier string
	// Details is the parsed provider response detail when one was available (nil
	// on parse failures, on aborted streams, and for providers that report no
	// usage). Usage is populated regardless — read Usage, not Details.TokenUsage.
	Details *ResponseDetails
}

// UsageObserver is fired once per billed provider round-trip.
//
// It is the source of truth for cost: the *WithUsage generators return only the
// details of the attempt that ultimately succeeded, so after retries the
// returned details under-report what was actually billed. The observer sees
// every attempt, including the discarded ones.
//
// It runs synchronously on the generation path. Keep it cheap — buffer to a
// channel or counter rather than doing I/O inline. A panic is recovered and
// logged so a faulty recorder cannot take down generation.
//
// The same observer value may be shared by many clients running concurrently, so
// an implementation must be safe for concurrent use.
type UsageObserver func(ctx context.Context, event UsageEvent)
