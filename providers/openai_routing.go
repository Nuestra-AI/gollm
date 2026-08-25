package providers

import "strings"

// This file is fork-local: upstream (teilomillet/gollm) has no equivalent, so it
// will never conflict on a merge. The only upstream-owned line this feature adds
// is the single routeOpenAIProvider call at the top of ProviderRegistry.Get.

// routeOpenAIProvider picks the OpenAI transport for a given model.
//
// Some OpenAI models cannot be served by /v1/chat/completions at all. Handing one
// of those to the default "openai" provider is a guaranteed API error, so the
// registry transparently substitutes "openai-responses".
//
// Routing covers only that hard constraint. Models that /v1/chat/completions can
// serve stay there, because /v1/responses measures several times slower and the
// transport should not change under a caller who did not ask for it. The one
// conditional restriction — gpt-5.4+ rejecting function tools combined with
// reasoning — is handled on the Chat Completions path instead, by pinning
// reasoning_effort to "none" for those requests; see
// applyOpenAIToolReasoningCarveOut. Callers who would rather keep reasoning and
// pay the latency select the Responses transport explicitly via
// gollm.WithOpenAIResponsesAPI.
//
// Scope rules, in order:
//
//  1. Only the bare "openai" provider is routed. azure-openai, google-openai,
//     openrouter, groq, lambda, aliyun, vllm and lmstudio all carry OpenAI-shaped
//     model ids but have entirely different (or absent) Responses support, so they
//     are left alone. Gating on the provider name first is what makes the model
//     patterns below safe — "gemini-2.5-pro" never reaches them.
//  2. An explicit "openai-responses" is already correct and is never rewritten.
//  3. "openai-chat" pins Chat Completions and is never rewritten, for callers
//     behind a proxy or test double that only speaks /v1/chat/completions.
//  4. Anything unrecognized stays on Chat Completions — the fail-safe direction,
//     since that is the pre-routing status quo.
//
// Routing stays narrow: it fires only where Chat Completions genuinely cannot
// serve the model, never where Responses is merely preferred. Models that work
// on both keep their existing transport so response shapes and usage fields stay
// stable for downstream callers.
//

// Note for callers: LLM.GetProvider reports the transport actually in use, so a
// routed model reads back as "openai-responses" even though "openai" was configured.
func routeOpenAIProvider(name, model string) string {
	if name != "openai" {
		return name
	}
	model = baseModelID(model)
	if isResponsesOnlyModel(model) {
		return "openai-responses"
	}
	return name
}

// baseModelID unwraps a fine-tuned model id to the base model it was trained
// from. Fine-tune ids are colon-delimited — "ft:<base>:<org>:<name>:<id>" — and
// the trailing free-text name is chosen by the customer, so matching the family
// patterns against the whole id would let a name like "sales-pro-v2" trip the
// "pro" rule. Routing follows the base model, which is what the endpoint
// restrictions actually attach to.
func baseModelID(model string) string {
	if !strings.HasPrefix(model, "ft:") {
		return model
	}
	parts := strings.Split(model, ":")
	if len(parts) < 2 || parts[1] == "" {
		return model
	}
	return parts[1]
}

// isResponsesOnlyModel reports whether an OpenAI model id is served exclusively by
// /v1/responses. Each family below was verified against the "supported endpoints"
// table on the model's page at developers.openai.com on 2026-08-25, where the row
// reads "Chat Completions | v1/chat/completions | Not supported".
//
// These are family patterns rather than an exhaustive id list on purpose: OpenAI
// ships new Codex and deep-research models often, and every one so far has been
// Responses-only. Matching the family means a new id routes correctly on release
// instead of waiting for a table update here.
//
// Callers must gate on the "openai" provider name before calling this — see
// routeOpenAIProvider.
func isResponsesOnlyModel(model string) bool {
	switch {
	// Codex: gpt-5-codex, gpt-5.1-codex, gpt-5.1-codex-mini, gpt-5.2-codex,
	// gpt-5.3-codex. Every Codex-optimized model has shipped Responses-only.
	// The legacy codex-mini-latest matches too; it was shut down 2026-02-12
	// (replaced by the gpt-5.x-codex line) and is kept here only so a stale
	// caller gets a coherent error rather than a 404 from the wrong endpoint.
	case hasModelSegment(model, "codex"):
		return true

	// Pro reasoning: gpt-5-pro, o3-pro, o1-pro, plus dated snapshots such as
	// gpt-5-pro-2025-10-06. Note that gpt-5-pro's successor is not a -pro model
	// at all — it is gpt-5.6-sol with reasoning.mode "pro", which is routed by
	// rejectsToolsOnChatCompletions below rather than by this rule.
	case hasModelSegment(model, "pro"):
		return true

	// Deep research: o3-deep-research, o4-mini-deep-research. Contains rather
	// than a segment match because the marker spans two hyphenated segments.
	case strings.Contains(model, "deep-research"):
		return true

	// Computer use: computer-use-preview and its dated snapshots. Shut down
	// 2026-07-23 (replaced by gpt-5.6-terra); retained for the same reason as
	// codex-mini-latest above.
	case strings.HasPrefix(model, "computer-use"):
		return true
	}
	return false
}

// rejectsToolsOnChatCompletions reports whether a model fails on
// /v1/chat/completions when function tools are combined with reasoning.
//
// This does not drive routing. It is the shared definition of the affected set,
// consumed by applyOpenAIToolReasoningCarveOut on the Chat Completions path,
// which keeps those requests working by sending reasoning_effort "none" rather
// than by moving them to a slower transport.
//
// This deliberately contradicts the "supported endpoints" table on
// developers.openai.com, which lists Chat Completions as "Supported" for these
// models — and it is, for plain text. The moment a request pairs function tools
// with reasoning, the API returns 400:
//
//	Function tools with reasoning_effort are not supported for gpt-5.6-sol in
//	/v1/chat/completions. To use function tools, use /v1/responses or set
//	reasoning_effort to 'none'
//
// The bound is GPT-5.4, where the restriction was introduced. Within it the
// failure comes in two shapes:
//
//   - gpt-5.4 and gpt-5.5 fail when the caller sets reasoning_effort explicitly
//     alongside tools.
//   - The GPT-5.6 frontier line (sol, terra, luna) reasons by default, so the
//     rejection fires even with reasoning_effort omitted entirely.
//
// Both shapes are covered by the same carve-out, since it pins the parameter
// rather than relying on its absence.
//
// Expressed as a bound rather than an id list so later releases inherit it.
//
// Verified 2026-08-25 from the 400 reproduced across litellm (#33221), LibreChat
// (#14231, #14355), pipecat (#4043, on gpt-5.4) and ruby_llm (#785). OpenAI has
// not documented the restriction in the API reference or changelog, so this rule
// rests on reproduced behavior rather than published spec — re-check before
// removing it.
func rejectsToolsOnChatCompletions(model string) bool {
	minor, isGPT5 := gpt5MinorVersion(model)
	if !isGPT5 || minor < 4 {
		return false
	}
	// The non-reasoning chat variants have no reasoning to conflict with tools.
	// Matched on a "chat" segment rather than isGPT5ChatModel, which is anchored
	// to the "gpt-5-chat" prefix and so can only ever match minor 0 — below this
	// bound already. A future "gpt-5.6-chat-latest" needs this to exclude it.
	return !hasModelSegment(model, "chat")
}

// hasModelSegment reports whether a hyphen-delimited model id contains the exact
// segment seg. Segment matching avoids the false positives a substring test would
// produce on ids that merely embed the word, while still matching dated snapshots
// like "o3-pro-2025-06-10".
func hasModelSegment(model, seg string) bool {
	for _, part := range strings.Split(model, "-") {
		if part == seg {
			return true
		}
	}
	return false
}
