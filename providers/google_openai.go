// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"encoding/json"
	"math"
	"strconv"
	"strings"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/types"
)

// GoogleProvider implements the Provider interface for Google's Gemini API through the
// OpenAI-compatible endpoint. Accordingly, it inherits from OpenAIProvider
type GoogleProvider struct {
	OpenAIProvider
}

// NewGoogleProvider creates a new Google provider instance.
// It initializes the provider with the given API key, model, and optional headers.
//
// Parameters:
//   - apiKey: Gemini API key for authentication
//   - model: The model to use (e.g., "gemini-2.0-flash")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured Google Provider instance
func NewGoogleProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	provider := &GoogleProvider{
		OpenAIProvider: *NewOpenAIProvider(apiKey, model, extraHeaders).(*OpenAIProvider),
	}
	// Gemini maps a "system"-role message to system_instruction and doesn't
	// recognize OpenAI's "developer" role.
	provider.systemRole = "system"

	return provider
}

// Name returns "google-openai" as the provider identifier.
func (p *GoogleProvider) Name() string {
	return "google-openai"
}

// Endpoint returns the Gemini API endpoint URL for generating content.
func (p *GoogleProvider) Endpoint() string {
	return "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
}

// SetDefaultOptions configures standard options from the global configuration.
func (p *GoogleProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}
	p.logger.Debug("Default options set", "temperature", config.Temperature, "max_tokens", config.MaxTokens)
}

// Gemini thinking/budget support (Gemini 2.x and 3.x).
//
// The OpenAI-compatible endpoint exposes Gemini's thinking controls three ways
// (https://ai.google.dev/gemini-api/docs/openai, .../docs/thinking):
//
//   - reasoning_effort — a top-level "none"/"low"/"medium"/"high" knob, the
//     cross-provider lever gollm already uses for OpenAI and Anthropic. Accepted
//     by both the 2.x and 3.x families, so it is the portable choice and the
//     PRIMARY control: when set it wins over an explicit budget/level.
//   - thinking_budget — an explicit thinking token budget, the 2.5-family native
//     control (0 disables thinking where allowed, -1 requests a model-chosen
//     dynamic budget). Gemini 3 dropped token budgets.
//   - thinking_level — "minimal"/"low"/"medium"/"high", the Gemini 3 native
//     control (pro tiers accept only low/high). Not accepted by 2.5.
//
// The explicit controls live under extra_body.google.thinking_config and overlap
// with reasoning_effort — the API rejects sending an effort together with a
// budget/level. reasoning_effort is primary: if a caller sets it, it is emitted
// and the budget/level are ignored. A budget/level applies only when no effort
// is set. Callers may pass either explicit field regardless of the target model:
// the provider emits whichever form that model's family accepts, cross-translating
// budget<->level so code keeps working when a model is swapped across families.
// include_thoughts (thought summaries) is orthogonal and rides alongside any of
// the above.
//
// The embedded OpenAIProvider strips reasoning_effort for non-OpenAI models and
// otherwise passes unknown option keys straight through to the request body, so
// each prepare entrypoint is wrapped to normalize the thinking fields on the
// marshaled body. Go embedding is non-virtual — the inherited methods call each
// other on *OpenAIProvider, never these overrides — so every public entrypoint
// needs its own wrapper to guarantee the normalization runs exactly once.

// PrepareRequest wraps OpenAIProvider.PrepareRequest to apply Gemini thinking
// controls (see the package-level note above).
func (p *GoogleProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	body, err := p.OpenAIProvider.PrepareRequest(prompt, options)
	return p.withThinking(body, err, options)
}

// PrepareRequestWithSchema wraps OpenAIProvider.PrepareRequestWithSchema to apply
// Gemini thinking controls.
func (p *GoogleProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	body, err := p.OpenAIProvider.PrepareRequestWithSchema(prompt, options, schema)
	return p.withThinking(body, err, options)
}

// PrepareRequestWithMessages wraps OpenAIProvider.PrepareRequestWithMessages to
// apply Gemini thinking controls.
func (p *GoogleProvider) PrepareRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error) {
	body, err := p.OpenAIProvider.PrepareRequestWithMessages(messages, options)
	return p.withThinking(body, err, options)
}

// PrepareRequestWithMessagesAndSchema wraps
// OpenAIProvider.PrepareRequestWithMessagesAndSchema to apply Gemini thinking
// controls.
func (p *GoogleProvider) PrepareRequestWithMessagesAndSchema(messages []types.MemoryMessage, options map[string]interface{}, schema interface{}) ([]byte, error) {
	body, err := p.OpenAIProvider.PrepareRequestWithMessagesAndSchema(messages, options, schema)
	return p.withThinking(body, err, options)
}

// PrepareStreamRequest wraps OpenAIProvider.PrepareStreamRequest to apply Gemini
// thinking controls.
func (p *GoogleProvider) PrepareStreamRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	body, err := p.OpenAIProvider.PrepareStreamRequest(prompt, options)
	return p.withThinking(body, err, options)
}

// PrepareStreamRequestWithMessages wraps
// OpenAIProvider.PrepareStreamRequestWithMessages to apply Gemini thinking
// controls.
func (p *GoogleProvider) PrepareStreamRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error) {
	body, err := p.OpenAIProvider.PrepareStreamRequestWithMessages(messages, options)
	return p.withThinking(body, err, options)
}

// withThinking applies Gemini thinking normalization to a prepared body,
// short-circuiting on a preparation error. The six prepare overrides funnel
// through it so the normalization runs in exactly one place.
func (p *GoogleProvider) withThinking(body []byte, err error, options map[string]interface{}) ([]byte, error) {
	if err != nil {
		return nil, err
	}
	return p.applyThinking(body, options)
}

// geminiMajorVersion returns the integer major version parsed from a Gemini
// model id ("gemini-2.5-flash" -> 2, "gemini-3-pro-preview" -> 3), or 0 when no
// numeric version is present ("gemini-flash-latest", "gemini-future-x").
func geminiMajorVersion(model string) int {
	m := strings.ToLower(model)
	i := strings.Index(m, "gemini-")
	if i < 0 {
		return 0
	}
	rest := m[i+len("gemini-"):]
	j := 0
	for j < len(rest) && rest[j] >= '0' && rest[j] <= '9' {
		j++
	}
	if j == 0 {
		return 0
	}
	n, _ := strconv.Atoi(rest[:j])
	return n
}

// isGeminiThinkingModel reports whether the model supports Gemini's thinking
// controls. The 1.x and 2.0 base models are non-thinking (only their explicit
// "-thinking" variants reason); everything from 2.5 onward — including unknown
// future ids and unversioned aliases (gemini-flash-latest) — is thinking-capable,
// so this fails open forward, matching the forward-compatible posture of the
// Anthropic legacy-thinking gate.
func isGeminiThinkingModel(model string) bool {
	m := strings.ToLower(model)
	if !strings.Contains(m, "gemini") {
		return false
	}
	if strings.Contains(m, "gemini-1.") || strings.Contains(m, "gemini-2.0") {
		return strings.Contains(m, "thinking")
	}
	return true
}

// isGemini3Model reports whether the model belongs to the Gemini 3+ family,
// whose native explicit control is thinking_level (not thinking_budget). Known
// 1.x/2.x ids are not 3.x. Among unversioned ids, only the "-latest" aliases
// (gemini-flash-latest, gemini-pro-latest) fail open to 3.x, since "latest"
// tracks the newest generation; other unrecognized ids default to the 2.x form
// rather than risk sending a 3.x-only thinking_level to a 2.x endpoint.
func isGemini3Model(model string) bool {
	switch v := geminiMajorVersion(model); {
	case v >= 3:
		return true
	case v == 1 || v == 2:
		return false
	default:
		return strings.Contains(strings.ToLower(model), "latest")
	}
}

// geminiIsProModel reports whether the model is a Gemini "pro" tier. Pro tiers
// cannot fully disable thinking (2.5 Pro has a 128-token floor; 3 Pro's lowest
// level is "low"), so a disable request is clamped up to the model's floor.
func geminiIsProModel(model string) bool {
	return strings.Contains(strings.ToLower(model), "pro")
}

// geminiMinBudget returns the smallest thinking_budget a 2.x model accepts: pro
// tiers have a 128-token floor and cannot disable thinking; flash/flash-lite
// accept 0 (thinking off).
func geminiMinBudget(model string) int {
	if geminiIsProModel(model) {
		return 128
	}
	return 0
}

// normalizeGeminiEffort maps a reasoning_effort hint onto the levels Gemini's
// OpenAI-compatible reasoning_effort field accepts (none/low/medium/high).
// OpenAI's "minimal" collapses to "low"; Anthropic's "xhigh"/"max" collapse to
// "high" (Gemini's ceiling); anything unrecognized defaults to "medium".
func normalizeGeminiEffort(effort string) string {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "none":
		return "none"
	case "minimal", "low":
		return "low"
	case "medium":
		return "medium"
	case "high", "xhigh", "max":
		return "high"
	default:
		return "medium"
	}
}

// effortForModel normalizes an effort and clamps a disable request ("none") up to
// "low" on pro tiers, which cannot turn thinking off.
func (p *GoogleProvider) effortForModel(effort string) string {
	e := normalizeGeminiEffort(effort)
	if e == "none" && geminiIsProModel(p.model) {
		return "low"
	}
	return e
}

// clampGemini3Level constrains a canonical thinking_level (minimal/low/medium/
// high) to the set a model accepts. Pro tiers accept only low/high, so minimal
// collapses to low and medium to high; other 3.x models (flash and unknown
// future ids) accept the full set.
func clampGemini3Level(level, model string) string {
	if geminiIsProModel(model) {
		switch level {
		case "minimal":
			return "low"
		case "medium":
			return "high"
		}
	}
	return level
}

// normalizeGemini3Level maps an arbitrary level/effort hint onto a canonical
// thinking_level clamped to what the model accepts.
func normalizeGemini3Level(level, model string) string {
	var canon string
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "none", "minimal", "min":
		canon = "minimal"
	case "low":
		canon = "low"
	case "medium", "med":
		canon = "medium"
	case "high", "xhigh", "max":
		canon = "high"
	default:
		canon = "medium"
	}
	return clampGemini3Level(canon, model)
}

// geminiBudgetToLevel translates a 2.5-style token budget onto a Gemini 3
// thinking_level (used when a thinking_budget is supplied for a 3.x model, which
// rejects token budgets). A negative (dynamic) budget yields "" so the model's
// own dynamic default is left in place.
func geminiBudgetToLevel(budget int, model string) string {
	var lvl string
	switch {
	case budget < 0:
		return ""
	case budget == 0:
		lvl = "minimal"
	case budget <= 4096:
		lvl = "low"
	case budget <= 12288:
		lvl = "medium"
	default:
		lvl = "high"
	}
	return clampGemini3Level(lvl, model)
}

// geminiLevelToBudget translates a Gemini 3 thinking_level onto a 2.5-style
// token budget (used when a thinking_level is supplied for a 2.x model, which
// rejects levels). "none" maps to 0 (disable) to match the level semantics;
// clampBudget then raises 0 to the model's floor on pro tiers. Values sit
// comfortably inside the 2.5 Flash/Pro ranges.
func geminiLevelToBudget(level string) int {
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "none":
		return 0
	case "minimal", "min":
		return 512
	case "low":
		return 4096
	case "medium", "med":
		return 8192
	case "high", "xhigh", "max":
		return 16384
	default:
		return 8192
	}
}

// clampBudget constrains a 2.x thinking_budget to what the model accepts: a
// negative value is preserved as the -1 dynamic sentinel; anything below the
// model floor (0 for flash, 128 for pro) is raised to the floor so a disable
// request is not rejected by a pro tier.
func (p *GoogleProvider) clampBudget(budget int) int {
	if budget < 0 {
		return -1
	}
	if min := geminiMinBudget(p.model); budget < min {
		return min
	}
	return budget
}

// geminiParseBudget converts a thinking_budget option value to an int, accepting
// int/int64/float64 and numeric strings. ok is false for an unparseable value
// (a malformed string, or an unexpected type), so the caller treats it as unset
// rather than silently emitting a 0 that would disable thinking.
func geminiParseBudget(v interface{}) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int64:
		return int(n), true
	case float64:
		// Reject a fractional budget (e.g. 2048.9) rather than truncate it into a
		// different, unintended value. Whole floats (JSON numbers unmarshal as
		// float64) still convert cleanly.
		if math.Trunc(n) != n {
			return 0, false
		}
		return int(n), true
	case string:
		i, err := strconv.Atoi(strings.TrimSpace(n))
		if err != nil {
			return 0, false
		}
		return i, true
	default:
		return 0, false
	}
}

// geminiThinkingIntent holds the resolved thinking controls for one request,
// with per-request options already preferred over provider-level defaults.
type geminiThinkingIntent struct {
	effort         string // "" when unset
	level          string // "" when unset
	budget         int
	hasBudget      bool
	includeThought bool
}

// resolveThinkingIntent reads the thinking options, preferring a per-request
// value over the provider-level default for each key. Empty-string efforts/levels
// and unparseable/non-matching-typed values are treated as unset, so they fall
// through to the provider default instead of masking it.
func (p *GoogleProvider) resolveThinkingIntent(options map[string]interface{}) geminiThinkingIntent {
	str := func(key string) string {
		for _, m := range []map[string]interface{}{options, p.options} {
			if s, ok := optionString(m[key]); ok && s != "" {
				return s
			}
		}
		return ""
	}
	intent := geminiThinkingIntent{
		effort: str("reasoning_effort"),
		level:  str("thinking_level"),
	}
	for _, m := range []map[string]interface{}{options, p.options} {
		if v, ok := m["thinking_budget"]; ok {
			if b, parsed := geminiParseBudget(v); parsed {
				intent.budget, intent.hasBudget = b, true
				break
			}
		}
	}
	for _, m := range []map[string]interface{}{options, p.options} {
		if b, ok := m["include_thoughts"].(bool); ok {
			intent.includeThought = b
			break
		}
	}
	return intent
}

// applyThinking normalizes Gemini thinking controls on an already-marshaled
// request body. reasoning_effort is the primary control (it wins over an explicit
// budget/level); a budget/level applies only when no effort is set;
// include_thoughts is orthogonal. When nothing is requested — or the model does
// not support thinking — the model's own defaults are left untouched.
func (p *GoogleProvider) applyThinking(body []byte, options map[string]interface{}) ([]byte, error) {
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}

	// reasoning_effort/thinking_budget/thinking_level/include_thoughts are gollm
	// option keys, not valid Gemini top-level fields; strip any that leaked
	// through the OpenAI option passthrough (e.g. a provider-level SetOption)
	// before re-emitting them in the shapes the Gemini API expects. reasoning_effort
	// is re-derived below.
	delete(req, "reasoning_effort")
	delete(req, "thinking_budget")
	delete(req, "thinking_level")
	delete(req, "include_thoughts")

	intent := p.resolveThinkingIntent(options)
	requested := intent.effort != "" || intent.level != "" || intent.hasBudget || intent.includeThought

	// Nothing requested, or a non-thinking model: leave the model's own defaults.
	if !requested || !isGeminiThinkingModel(p.model) {
		return json.Marshal(req)
	}

	switch {
	case intent.effort != "":
		// Primary control: reasoning_effort wins over any budget/level.
		req["reasoning_effort"] = p.effortForModel(intent.effort)
	case intent.level != "" || intent.hasBudget:
		if cfg := p.explicitThinkingConfig(intent); cfg != nil {
			p.setThinkingConfig(req, cfg, false)
		}
	}

	// include_thoughts is orthogonal to the effort/budget/level overlap, so it
	// rides alongside whatever control (or none) was chosen.
	if intent.includeThought {
		p.setThinkingConfig(req, nil, true)
	}

	// Log only the thinking fields we set (effort + thinking_config), never the
	// whole extra_body, which may carry unrelated caller-supplied data.
	p.logger.Debug("Gemini thinking configured",
		"model", p.model, "reasoning_effort", req["reasoning_effort"], "thinking_config", geminiThinkingConfigOf(req))
	return json.Marshal(req)
}

// explicitThinkingConfig resolves an explicit budget/level into the thinking_config
// field the target model's family accepts: thinking_level for Gemini 3+,
// thinking_budget for Gemini 2.x. When only the other family's field is supplied
// it is cross-translated. Returns nil when there is nothing to emit (e.g. a
// dynamic budget on a 3.x model, where the model's own default stands). Only
// called when intent has a level or budget (never an effort).
func (p *GoogleProvider) explicitThinkingConfig(intent geminiThinkingIntent) map[string]interface{} {
	if isGemini3Model(p.model) {
		// Native control: thinking_level. Prefer an explicit level, else derive
		// one from a supplied budget.
		if intent.level != "" {
			return map[string]interface{}{"thinking_level": normalizeGemini3Level(intent.level, p.model)}
		}
		if lvl := geminiBudgetToLevel(intent.budget, p.model); lvl != "" {
			return map[string]interface{}{"thinking_level": lvl}
		}
		return nil
	}

	// Gemini 2.x native control: thinking_budget. Prefer an explicit budget, else
	// derive one from a supplied level; clamp to the model's accepted range.
	budget := intent.budget
	if !intent.hasBudget {
		budget = geminiLevelToBudget(intent.level)
	}
	return map[string]interface{}{"thinking_budget": p.clampBudget(budget)}
}

// geminiThinkingConfigOf returns the extra_body.google.thinking_config map for
// logging, or nil when absent. Used to log only the thinking fields rather than
// the whole extra_body.
func geminiThinkingConfigOf(req map[string]interface{}) interface{} {
	extraBody, _ := req["extra_body"].(map[string]interface{})
	google, _ := extraBody["google"].(map[string]interface{})
	return google["thinking_config"]
}

// setThinkingConfig merges cfg (and include_thoughts) into
// extra_body.google.thinking_config, preserving any caller-supplied extra_body
// keys rather than clobbering them.
func (p *GoogleProvider) setThinkingConfig(req, cfg map[string]interface{}, includeThoughts bool) {
	extraBody, _ := req["extra_body"].(map[string]interface{})
	if extraBody == nil {
		extraBody = map[string]interface{}{}
	}
	google, _ := extraBody["google"].(map[string]interface{})
	if google == nil {
		google = map[string]interface{}{}
	}
	thinkingConfig, _ := google["thinking_config"].(map[string]interface{})
	if thinkingConfig == nil {
		thinkingConfig = map[string]interface{}{}
	}
	for k, v := range cfg {
		thinkingConfig[k] = v
	}
	if includeThoughts {
		thinkingConfig["include_thoughts"] = true
	}
	google["thinking_config"] = thinkingConfig
	extraBody["google"] = google
	req["extra_body"] = extraBody
}
