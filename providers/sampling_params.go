// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/utils"
)

// Per-provider sampling vocabularies.
//
// No two providers accept the same subset of gollm's sampling setters, and several
// spell the same idea differently. Each table cites the doc it came from; a
// parameter missing from one is missing because that API does not have it.

// paramSource reads one sampling value out of a Config, reporting false when the
// caller never set it. The optional parameters are pointers with no envDefault, so
// nil means unset.
type paramSource func(*config.Config) (interface{}, bool)

// paramSpec binds one gollm sampling value to the request key a provider uses for it.
type paramSpec struct {
	wire string
	read paramSource
}

func srcTemperature(c *config.Config) (interface{}, bool) { return c.Temperature, true }
func srcMaxTokens(c *config.Config) (interface{}, bool)   { return c.MaxTokens, true }

// srcTopP treats a zero as unset: top_p 0 is degenerate, and NewConfig leaves Go's
// zero where LoadConfig applies an envDefault.
func srcTopP(c *config.Config) (interface{}, bool) {
	if c.TopP > 0 {
		return c.TopP, true
	}
	return nil, false
}

func srcFrequencyPenalty(c *config.Config) (interface{}, bool) { return c.FrequencyPenalty, true }
func srcPresencePenalty(c *config.Config) (interface{}, bool)  { return c.PresencePenalty, true }

func srcSeed(c *config.Config) (interface{}, bool)        { return derefInt(c.Seed) }
func srcTopK(c *config.Config) (interface{}, bool)        { return derefInt(c.TopK) }
func srcRepeatLastN(c *config.Config) (interface{}, bool) { return derefInt(c.RepeatLastN) }
func srcMirostat(c *config.Config) (interface{}, bool)    { return derefInt(c.Mirostat) }

// srcStopSequences reports false for an empty list, which some APIs reject.
func srcStopSequences(c *config.Config) (interface{}, bool) {
	if len(c.StopSequences) == 0 {
		return nil, false
	}
	return c.StopSequences, true
}

func srcMinP(c *config.Config) (interface{}, bool)          { return derefFloat(c.MinP) }
func srcRepeatPenalty(c *config.Config) (interface{}, bool) { return derefFloat(c.RepeatPenalty) }
func srcMirostatEta(c *config.Config) (interface{}, bool)   { return derefFloat(c.MirostatEta) }
func srcMirostatTau(c *config.Config) (interface{}, bool)   { return derefFloat(c.MirostatTau) }
func srcTfsZ(c *config.Config) (interface{}, bool)          { return derefFloat(c.TfsZ) }

func derefInt(v *int) (interface{}, bool) {
	if v == nil {
		return nil, false
	}
	return *v, true
}

func derefFloat(v *float64) (interface{}, bool) {
	if v == nil {
		return nil, false
	}
	return *v, true
}

// openAIChatSamplingParams is the sampling subset of POST /v1/chat/completions, and
// the baseline for every OpenAI-compatible gateway. No top_k or min_p.
// CreateChatCompletionRequest in github.com/openai/openai-openapi (2026-08-25).
var openAIChatSamplingParams = []paramSpec{
	{"temperature", srcTemperature},
	{"max_tokens", srcMaxTokens},
	{"top_p", srcTopP},
	{"frequency_penalty", srcFrequencyPenalty},
	{"presence_penalty", srcPresencePenalty},
	{"seed", srcSeed},
	{"stop", srcStopSequences},
}

// openAIResponsesSamplingParams is the sampling subset of POST /v1/responses: no
// seed, no penalties, and the one endpoint here with no stop parameter at all.
// CreateResponse in github.com/openai/openai-openapi (2026-08-25).
var openAIResponsesSamplingParams = []paramSpec{
	{"temperature", srcTemperature},
	{"max_tokens", srcMaxTokens},
	{"top_p", srcTopP},
}

// anthropicSamplingParams is the Messages API sampling set. temperature, top_p and
// top_k reach the wire only on models through Claude Opus 4.6; see
// anthropicAcceptsSamplingParams. No seed, no penalties.
var anthropicSamplingParams = []paramSpec{
	{"max_tokens", srcMaxTokens},
	{"temperature", srcTemperature},
	{"top_p", srcTopP},
	{"top_k", srcTopK},
	{"stop_sequences", srcStopSequences},
}

// anthropicRequiredParams is what survives on models that dropped the sampling
// controls: max_tokens is required, and stop_sequences was not deprecated.
var anthropicRequiredParams = []paramSpec{
	{"max_tokens", srcMaxTokens},
	{"stop_sequences", srcStopSequences},
}

// mistralSamplingParams follows https://docs.mistral.ai/api/ (2026-08-25). The seed
// is spelled random_seed; sending "seed" does nothing. No top_k.
var mistralSamplingParams = []paramSpec{
	{"temperature", srcTemperature},
	{"max_tokens", srcMaxTokens},
	{"top_p", srcTopP},
	{"frequency_penalty", srcFrequencyPenalty},
	{"presence_penalty", srcPresencePenalty},
	{"random_seed", srcSeed},
	{"stop", srcStopSequences},
}

// openRouterSamplingParams follows https://openrouter.ai/docs/api-reference/parameters
// (2026-08-25). repeat_last_n, mirostat and tfs_z are not in its schema.
var openRouterSamplingParams = []paramSpec{
	{"temperature", srcTemperature},
	{"max_tokens", srcMaxTokens},
	{"top_p", srcTopP},
	{"top_k", srcTopK},
	{"min_p", srcMinP},
	{"frequency_penalty", srcFrequencyPenalty},
	{"presence_penalty", srcPresencePenalty},
	{"repetition_penalty", srcRepeatPenalty},
	{"seed", srcSeed},
	{"stop", srcStopSequences},
}

// groqSamplingParams is OpenAI's set: what https://console.groq.com/docs/openai
// (2026-08-25) rejects is not sampling.
var groqSamplingParams = openAIChatSamplingParams

// deepSeekSamplingParams follows https://api-docs.deepseek.com/api/create-chat-completion
// (2026-08-25). No seed; the penalties are deprecated there and "will not take
// effect if you pass it to the API", so they are not sent.
var deepSeekSamplingParams = []paramSpec{
	{"temperature", srcTemperature},
	{"max_tokens", srcMaxTokens},
	{"top_p", srcTopP},
	{"stop", srcStopSequences},
}

// googleSamplingParams covers Gemini's OpenAI-compatibility endpoint, which publishes
// no parameter matrix (https://ai.google.dev/gemini-api/docs/openai, 2026-08-25), so
// it is held to the OpenAI shape it emulates.
var googleSamplingParams = []paramSpec{
	{"temperature", srcTemperature},
	{"max_tokens", srcMaxTokens},
	{"top_p", srcTopP},
	{"frequency_penalty", srcFrequencyPenalty},
	{"presence_penalty", srcPresencePenalty},
	{"stop", srcStopSequences},
}

// cohereSamplingParams follows Cohere's v2 chat API, which names top-k and top-p k
// and p.
var cohereSamplingParams = []paramSpec{
	{"temperature", srcTemperature},
	{"max_tokens", srcMaxTokens},
	{"p", srcTopP},
	{"k", srcTopK},
	{"frequency_penalty", srcFrequencyPenalty},
	{"presence_penalty", srcPresencePenalty},
	{"seed", srcSeed},
	{"stop_sequences", srcStopSequences},
}

// vllmSamplingParams is OpenAI's set plus vLLM's body extras: top_k, min_p,
// repetition_penalty.
// https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/ (2026-08-25)
var vllmSamplingParams = []paramSpec{
	{"temperature", srcTemperature},
	{"max_tokens", srcMaxTokens},
	{"top_p", srcTopP},
	{"top_k", srcTopK},
	{"min_p", srcMinP},
	{"frequency_penalty", srcFrequencyPenalty},
	{"presence_penalty", srcPresencePenalty},
	{"repetition_penalty", srcRepeatPenalty},
	{"seed", srcSeed},
	{"stop", srcStopSequences},
}

// lmStudioSamplingParams follows the payload list at
// https://lmstudio.ai/docs/developer/openai-compat/chat-completions (2026-08-25):
// OpenAI's set plus top_k and repeat_penalty. No min_p.
var lmStudioSamplingParams = []paramSpec{
	{"temperature", srcTemperature},
	{"max_tokens", srcMaxTokens},
	{"top_p", srcTopP},
	{"top_k", srcTopK},
	{"frequency_penalty", srcFrequencyPenalty},
	{"presence_penalty", srcPresencePenalty},
	{"repeat_penalty", srcRepeatPenalty},
	{"seed", srcSeed},
	{"stop", srcStopSequences},
}

// aliyunSamplingParams covers DashScope's compatible mode, which adds top_k to the
// OpenAI shape.
// https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope (2026-08-25)
var aliyunSamplingParams = []paramSpec{
	{"temperature", srcTemperature},
	{"max_tokens", srcMaxTokens},
	{"top_p", srcTopP},
	{"top_k", srcTopK},
	{"frequency_penalty", srcFrequencyPenalty},
	{"presence_penalty", srcPresencePenalty},
	{"seed", srcSeed},
	{"stop", srcStopSequences},
}

// ollamaSamplingParams is the runner option set for POST /api/generate. These do NOT
// go at the top level — Ollama reads them from a nested "options" object and ignores
// a top-level temperature. See nestOllamaOptions.
// https://github.com/ollama/ollama/blob/main/docs/api.md (2026-08-25)
//
// tfs_z is kept although current builds no longer document it: it is the only API
// SetTfsZ was ever aimed at, and Ollama ignores options it does not recognize.
var ollamaSamplingParams = []paramSpec{
	{"temperature", srcTemperature},
	{"num_predict", srcMaxTokens},
	{"top_p", srcTopP},
	{"top_k", srcTopK},
	{"min_p", srcMinP},
	{"frequency_penalty", srcFrequencyPenalty},
	{"presence_penalty", srcPresencePenalty},
	{"repeat_penalty", srcRepeatPenalty},
	{"repeat_last_n", srcRepeatLastN},
	{"mirostat", srcMirostat},
	{"mirostat_eta", srcMirostatEta},
	{"mirostat_tau", srcMirostatTau},
	{"tfs_z", srcTfsZ},
	{"seed", srcSeed},
	{"stop", srcStopSequences},
}

// lambdaSamplingParams covers Lambda Labs' Inference API, an OpenAI-compatible
// gateway with no documented parameters beyond that shape.
var lambdaSamplingParams = openAIChatSamplingParams

// bedrockSamplingParams is what the Bedrock provider stores under gollm's own names;
// bedrockSamplingFields translates them per model family. No family accepts a seed.
var bedrockSamplingParams = []paramSpec{
	{"temperature", srcTemperature},
	{"max_tokens", srcMaxTokens},
	{"top_p", srcTopP},
	{"top_k", srcTopK},
	{"stop", srcStopSequences},
}

// samplingWireNames is every request key the tables above can emit. It bounds what
// stripUnsupportedSampling may delete, so a provider-specific extra a caller passes
// through — OpenRouter's "provider", Google's "extra_body" — is never touched.
var samplingWireNames = func() map[string]bool {
	names := map[string]bool{}
	for _, table := range [][]paramSpec{
		openAIChatSamplingParams, anthropicSamplingParams, mistralSamplingParams,
		openRouterSamplingParams, deepSeekSamplingParams, googleSamplingParams,
		cohereSamplingParams, vllmSamplingParams, lmStudioSamplingParams,
		aliyunSamplingParams, ollamaSamplingParams,
	} {
		for _, spec := range table {
			names[spec.wire] = true
		}
	}
	// Titan's stopSequences comes from bedrockSamplingFields, not a table.
	names["stopSequences"] = true
	return names
}()

// Precomputed lookups for stripUnsupportedSampling, one per table above.
var (
	anthropicSupportedParams  = supportedSampling(anthropicSamplingParams)
	anthropicRequiredOnly     = supportedSampling(anthropicRequiredParams)
	mistralSupportedParams    = supportedSampling(mistralSamplingParams)
	openRouterSupportedParams = supportedSampling(openRouterSamplingParams)
	groqSupportedParams       = supportedSampling(groqSamplingParams)
	deepSeekSupportedParams   = supportedSampling(deepSeekSamplingParams)
	googleSupportedParams     = supportedSampling(googleSamplingParams)
	vllmSupportedParams       = supportedSampling(vllmSamplingParams)
	lambdaSupportedParams     = supportedSampling(lambdaSamplingParams)
	ollamaSupportedParams     = supportedSampling(ollamaSamplingParams)
)

// stopWireNames are the names the same stop-sequence list travels under.
var stopWireNames = []string{"stop", "stop_sequences", "stopSequences"}

// normalizeStopSequences rewrites whichever stop key the caller used to the one this
// API accepts, and widens a bare string into a list.
//
// The three names mean the same thing, so moving between providers should not mean
// renaming the key; without this the strip pass would delete the foreign spelling and
// generation would run past the caller's stop sequence. An API with no stop parameter
// has nothing to rename to, so the key is left for the strip pass.
func normalizeStopSequences(request map[string]interface{}, supported map[string]bool) {
	target := ""
	for _, name := range stopWireNames {
		if supported[name] {
			target = name
			break
		}
	}
	if target == "" {
		return
	}

	// A foreign spelling outranks the API's own: the tables only write defaults under
	// the API's own name, so a value under a different one was set for this call and
	// must not lose to a provider-level default.
	for _, name := range stopWireNames {
		value, present := request[name]
		if !present {
			continue
		}
		// A lone string is legal on OpenAI but not on the array-only APIs.
		if single, isString := value.(string); isString {
			value = []string{single}
		}
		if name != target {
			delete(request, name)
		}
		request[target] = value
	}
}

// optionSetter is the part of Provider that applySamplingDefaults needs.
type optionSetter interface {
	SetOption(key string, value interface{})
}

// applySamplingDefaults forwards every configured value the API accepts, under that
// API's own name. Values the caller never set are skipped.
func applySamplingDefaults(target optionSetter, cfg *config.Config, specs []paramSpec) {
	for _, spec := range specs {
		if value, ok := spec.read(cfg); ok {
			target.SetOption(spec.wire, value)
		}
	}
}

// supportedSampling turns a table into the lookup stripUnsupportedSampling wants.
func supportedSampling(specs []paramSpec) map[string]bool {
	supported := make(map[string]bool, len(specs))
	for _, spec := range specs {
		supported[spec.wire] = true
	}
	return supported
}

// stripUnsupportedSampling removes sampling parameters the provider does not accept
// from an assembled body, in place. SetDefaultOptions already forwards only supported
// ones; this covers per-request options, which providers copy in verbatim.
func stripUnsupportedSampling(request map[string]interface{}, supported map[string]bool, provider string, logger utils.Logger) {
	normalizeStopSequences(request, supported)

	for key := range request {
		if !samplingWireNames[key] || supported[key] {
			continue
		}
		delete(request, key)
		if logger != nil {
			logger.Debug("Dropped sampling parameter not accepted by this provider",
				"parameter", key, "provider", provider)
		}
	}
}
