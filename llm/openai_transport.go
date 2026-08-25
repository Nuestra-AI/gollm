package llm

import (
	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/providers"
)

// openAIAliasProviders select a specific OpenAI transport: "openai-responses" for
// /v1/responses, "openai-chat" to pin /v1/chat/completions against automatic
// routing. All authenticate with the same key as plain "openai".
var openAIAliasProviders = []string{"openai-responses", "openai-chat"}

// ResolveOpenAITransport applies the caller's tool+reasoning policy and makes the
// OpenAI key reachable under whichever transport name results.
//
// It lives here, not in gollm.NewLLM, because both constructors must agree:
// llm.NewLLM is public and used directly, and it resolves providers straight from
// the registry, which never sees config. Running only in the wrapper left a
// prefer-quality client silently behaving like prefer-speed.
//
// Idempotent, so calling it from both paths is safe — gollm.NewLLM needs it before
// its own validation, which runs earlier than this constructor.
func ResolveOpenAITransport(cfg *config.Config) {
	if cfg == nil {
		return
	}
	applyToolReasoningPolicy(cfg)
	ensureOpenAIAliasKey(cfg)
}

// applyToolReasoningPolicy moves a client to the Responses transport when the caller
// asked to keep reasoning on tool-carrying requests, which Chat Completions rejects
// from gpt-5.4 onward. See config.ToolReasoningPolicy.
//
// Applies only to the bare "openai" provider — an explicit transport outranks a
// policy — and only to affected models, so prefer-quality does not drag gpt-4o onto
// a slower endpoint.
func applyToolReasoningPolicy(cfg *config.Config) {
	if cfg.Provider != "openai" || cfg.ToolReasoning != config.ToolReasoningPreferQuality {
		return
	}
	if providers.ModelRejectsToolsOnChatCompletions(cfg.Model) {
		cfg.Provider = "openai-responses"
	}
}

// ensureOpenAIAliasKey copies the "openai" key into the configured alias slot when
// empty, so selecting a transport does not mean duplicating the key. Automatic
// routing never reaches here: it happens in the registry and leaves cfg.Provider
// as "openai".
func ensureOpenAIAliasKey(cfg *config.Config) {
	isAlias := false
	for _, alias := range openAIAliasProviders {
		if cfg.Provider == alias {
			isAlias = true
			break
		}
	}
	if !isAlias {
		return
	}
	if cfg.APIKeys == nil {
		cfg.APIKeys = make(map[string]string)
	}
	if cfg.APIKeys[cfg.Provider] == "" && cfg.APIKeys["openai"] != "" {
		cfg.APIKeys[cfg.Provider] = cfg.APIKeys["openai"]
	}
}
