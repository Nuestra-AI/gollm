package providers

import "strings"

// This file is fork-local: upstream (teilomillet/gollm) has no equivalent, so it
// will never conflict on a merge. The only upstream-owned line this feature adds
// is the single routeOpenAIProvider call at the top of ProviderRegistry.Get.

// routeOpenAIProvider substitutes "openai-responses" for models /v1/chat/completions
// cannot serve at all. Models it can serve stay there — /v1/responses measures
// several times slower — including gpt-5.4+, whose conditional tool restriction is
// handled by applyOpenAIToolReasoningCarveOut instead.
//
// Only the bare "openai" name routes; gating on it first is what keeps Gemini and
// OpenRouter ids away from the patterns below. "openai-chat" and unrecognized
// models stay on Chat Completions. A routed model reads back from LLM.GetProvider
// as "openai-responses".
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

// baseModelID unwraps "ft:<base>:<org>:<name>:<id>" to <base>. The name is
// customer-chosen, so matching patterns against the whole id lets something like
// "sales-pro-v2" trip the "pro" rule.
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

// isResponsesOnlyModel reports whether a model is served exclusively by
// /v1/responses. Each family was verified against the supported-endpoints table on
// its model page at developers.openai.com (2026-08-25), which reads "Chat
// Completions | v1/chat/completions | Not supported".
//
// Family patterns, not an id list: every Codex and deep-research model shipped so
// far has been Responses-only, so a new id routes correctly on release. Gate on
// the "openai" provider name before calling this.
func isResponsesOnlyModel(model string) bool {
	switch {
	// gpt-5-codex through gpt-5.3-codex. Legacy codex-mini-latest matches too;
	// it shut down 2026-02-12 but is kept so a stale caller gets a coherent
	// error rather than a 404 from the wrong endpoint.
	case hasModelSegment(model, "codex"):
		return true

	// gpt-5-pro, gpt-5.2-pro, gpt-5.4-pro, o3-pro, o1-pro and dated snapshots.
	// gpt-5-pro's successor is not a -pro model — it is gpt-5.6-sol with
	// reasoning.mode "pro", which Chat Completions serves.
	case hasModelSegment(model, "pro"):
		return true

	// o3-deep-research, o4-mini-deep-research. Contains, not a segment match:
	// the marker spans two hyphenated segments.
	case strings.Contains(model, "deep-research"):
		return true

	// computer-use-preview and snapshots. Shut down 2026-07-23; retained for the
	// same reason as codex-mini-latest.
	case strings.HasPrefix(model, "computer-use"):
		return true
	}
	return false
}

// hasModelSegment reports whether a hyphen-delimited id contains seg exactly.
// Avoids the false positives a substring test gives on ids that merely embed the
// word, while still matching snapshots like "o3-pro-2025-06-10".
func hasModelSegment(model, seg string) bool {
	for _, part := range strings.Split(model, "-") {
		if part == seg {
			return true
		}
	}
	return false
}
