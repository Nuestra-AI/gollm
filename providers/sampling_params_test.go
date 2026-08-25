package providers

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/utils"
)

// fullConfig is a Config with every sampling setter exercised, so a test can assert
// what each provider does and does not forward. The values are distinct so a
// mis-wired table shows up as the wrong number rather than a coincidental match.
func fullConfig() *config.Config {
	seed, topK, repeatLastN, mirostat := 7, 40, 64, 2
	minP, repeatPenalty, mirostatEta, mirostatTau, tfsZ := 0.05, 1.1, 0.1, 5.0, 1.0
	return &config.Config{
		Temperature:      0.3,
		MaxTokens:        512,
		TopP:             0.8,
		FrequencyPenalty: 0.25,
		PresencePenalty:  0.5,
		Seed:             &seed,
		TopK:             &topK,
		MinP:             &minP,
		RepeatPenalty:    &repeatPenalty,
		RepeatLastN:      &repeatLastN,
		Mirostat:         &mirostat,
		MirostatEta:      &mirostatEta,
		MirostatTau:      &mirostatTau,
		TfsZ:             &tfsZ,
	}
}

// decodeBody unmarshals a prepared request body for field-level assertions.
func decodeBody(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()
	var decoded map[string]interface{}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("request body is not valid JSON: %v\n%s", err, body)
	}
	return decoded
}

func assertNum(t *testing.T, body map[string]interface{}, key string, want float64) {
	t.Helper()
	got, ok := body[key]
	if !ok {
		t.Errorf("%q missing from request body", key)
		return
	}
	num, ok := got.(float64)
	if !ok {
		t.Errorf("%q = %v (%T), want a number", key, got, got)
		return
	}
	if num != want {
		t.Errorf("%q = %v, want %v", key, num, want)
	}
}

func assertAbsent(t *testing.T, body map[string]interface{}, keys ...string) {
	t.Helper()
	for _, key := range keys {
		if value, ok := body[key]; ok {
			t.Errorf("%q should not be in the request body, got %v", key, value)
		}
	}
}

// TestSamplingDefaultsPerProvider checks that each provider forwards exactly the
// parameters its API accepts, under that API's names.
func TestSamplingDefaultsPerProvider(t *testing.T) {
	cases := []struct {
		name     string
		provider Provider
		want     map[string]float64
		absent   []string
	}{
		{
			name:     "mistral spells the seed random_seed",
			provider: NewMistralProvider("k", "mistral-large-latest", nil),
			want: map[string]float64{
				"temperature": 0.3, "max_tokens": 512, "top_p": 0.8,
				"frequency_penalty": 0.25, "presence_penalty": 0.5, "random_seed": 7,
			},
			absent: []string{"seed", "top_k", "min_p", "repetition_penalty"},
		},
		{
			name:     "openrouter takes min_p and top_k, and names the repeat penalty repetition_penalty",
			provider: NewOpenRouterProvider("k", "anthropic/claude-sonnet-4-5", nil),
			want: map[string]float64{
				"temperature": 0.3, "max_tokens": 512, "top_p": 0.8, "top_k": 40,
				"min_p": 0.05, "repetition_penalty": 1.1, "seed": 7,
				"frequency_penalty": 0.25, "presence_penalty": 0.5,
			},
			absent: []string{"repeat_penalty", "mirostat", "tfs_z", "repeat_last_n"},
		},
		{
			name:     "groq takes OpenAI's set",
			provider: NewGroqProvider("k", "llama-3.3-70b-versatile", nil),
			want: map[string]float64{
				"temperature": 0.3, "max_tokens": 512, "top_p": 0.8, "seed": 7,
				"frequency_penalty": 0.25, "presence_penalty": 0.5,
			},
			absent: []string{"top_k", "min_p", "repetition_penalty"},
		},
		{
			name:     "vllm adds top_k, min_p and repetition_penalty to OpenAI's set",
			provider: NewVLLMProvider("k", "meta-llama/Llama-3-8B", nil),
			want: map[string]float64{
				"temperature": 0.3, "max_tokens": 512, "top_p": 0.8, "top_k": 40,
				"min_p": 0.05, "repetition_penalty": 1.1, "seed": 7,
			},
			absent: []string{"repeat_penalty", "mirostat", "tfs_z"},
		},
		{
			name:     "lambda is held to OpenAI's set",
			provider: NewLambdaProvider("k", "llama3.1-8b-instruct", nil),
			want: map[string]float64{
				"temperature": 0.3, "max_tokens": 512, "top_p": 0.8, "seed": 7,
			},
			absent: []string{"top_k", "min_p"},
		},
		{
			name:     "cohere names top-k and top-p k and p",
			provider: NewCohereProvider("k", "command-r-plus-08-2024", nil),
			want: map[string]float64{
				"temperature": 0.3, "max_tokens": 512, "p": 0.8, "k": 40, "seed": 7,
				"frequency_penalty": 0.25, "presence_penalty": 0.5,
			},
			absent: []string{"top_p", "top_k", "min_p"},
		},
		{
			name:     "lmstudio adds top_k and repeat_penalty, but not min_p",
			provider: NewLMStudioProvider("k", "lfm2.5-1.2b-instruct-mlx", nil),
			want: map[string]float64{
				"temperature": 0.3, "max_tokens": 512, "top_p": 0.8, "top_k": 40,
				"repeat_penalty": 1.1, "seed": 7,
			},
			absent: []string{"min_p", "repetition_penalty", "mirostat"},
		},
		{
			name:     "aliyun adds top_k to the OpenAI shape",
			provider: NewAliyunProvider("k", "qwen-turbo", nil),
			want: map[string]float64{
				"temperature": 0.3, "max_tokens": 512, "top_p": 0.8, "top_k": 40, "seed": 7,
			},
			absent: []string{"min_p", "repeat_penalty"},
		},
		{
			name:     "azure-openai gets no non-OpenAI extras",
			provider: NewAzureOpenAIProvider("k", "gpt-4o", map[string]string{"azure_endpoint": "https://x/y"}),
			want: map[string]float64{
				"temperature": 0.3, "max_tokens": 512, "top_p": 0.8, "seed": 7,
			},
			absent: []string{"top_k", "min_p", "repeat_penalty"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.provider.SetDefaultOptions(fullConfig())
			body, err := tc.provider.PrepareRequest("hi", map[string]interface{}{})
			if err != nil {
				t.Fatalf("PrepareRequest failed: %v", err)
			}
			decoded := decodeBody(t, body)
			for key, want := range tc.want {
				assertNum(t, decoded, key, want)
			}
			assertAbsent(t, decoded, tc.absent...)
		})
	}
}

// TestDeepSeekOmitsIgnoredParams verifies gollm does not send DeepSeek a seed it has
// no parameter for, nor the penalties its docs mark deprecated and inert.
func TestDeepSeekOmitsIgnoredParams(t *testing.T) {
	p := NewDeepSeekProvider("k", "deepseek-chat", nil)
	p.SetDefaultOptions(fullConfig())

	body, err := p.PrepareRequest("hi", map[string]interface{}{})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	decoded := decodeBody(t, body)
	assertNum(t, decoded, "temperature", 0.3)
	assertNum(t, decoded, "max_tokens", 512)
	assertNum(t, decoded, "top_p", 0.8)
	assertAbsent(t, decoded, "seed", "frequency_penalty", "presence_penalty", "top_k")
}

// TestUnsetParamsAreNotForwarded is the guard on the envDefault removal: a config
// the caller never touched must not put a sampling choice on the wire. The optional
// pointers are nil here, as LoadConfig now leaves them.
func TestUnsetParamsAreNotForwarded(t *testing.T) {
	cfg := &config.Config{Temperature: 0.7, MaxTokens: 100, TopP: 0.9}

	for _, p := range []Provider{
		NewOpenRouterProvider("k", "anthropic/claude-sonnet-4-5", nil),
		NewVLLMProvider("k", "meta-llama/Llama-3-8B", nil),
		NewMistralProvider("k", "mistral-large-latest", nil),
	} {
		p.SetDefaultOptions(cfg)
		body, err := p.PrepareRequest("hi", map[string]interface{}{})
		if err != nil {
			t.Fatalf("%s: PrepareRequest failed: %v", p.Name(), err)
		}
		decoded := decodeBody(t, body)
		for _, key := range []string{
			"top_k", "min_p", "repetition_penalty", "repeat_penalty",
			"repeat_last_n", "mirostat", "mirostat_eta", "mirostat_tau", "tfs_z",
			"seed", "random_seed",
		} {
			if value, ok := decoded[key]; ok {
				t.Errorf("%s: unset %q reached the request as %v", p.Name(), key, value)
			}
		}
	}
}

// TestAnthropicSamplingIsModelGated covers the deprecation on the Messages API:
// models through Claude Opus 4.6 still take temperature, top_p and top_k, and later
// ones reject them. max_tokens is required and survives either way.
func TestAnthropicSamplingIsModelGated(t *testing.T) {
	t.Run("legacy model keeps the sampling controls", func(t *testing.T) {
		p := NewAnthropicProvider("k", "claude-sonnet-4-5", nil)
		p.SetDefaultOptions(fullConfig())

		body, err := p.PrepareRequest("hi", map[string]interface{}{})
		if err != nil {
			t.Fatalf("PrepareRequest failed: %v", err)
		}
		decoded := decodeBody(t, body)
		assertNum(t, decoded, "temperature", 0.3)
		assertNum(t, decoded, "top_p", 0.8)
		assertNum(t, decoded, "top_k", 40)
		assertNum(t, decoded, "max_tokens", 512)
		assertAbsent(t, decoded, "seed", "frequency_penalty", "presence_penalty", "min_p")
	})

	t.Run("current model drops them but keeps max_tokens", func(t *testing.T) {
		p := NewAnthropicProvider("k", "claude-opus-5", nil)
		p.SetDefaultOptions(fullConfig())

		body, err := p.PrepareRequest("hi", map[string]interface{}{})
		if err != nil {
			t.Fatalf("PrepareRequest failed: %v", err)
		}
		decoded := decodeBody(t, body)
		assertNum(t, decoded, "max_tokens", 512)
		assertAbsent(t, decoded, "temperature", "top_p", "top_k", "seed")
	})

	t.Run("a per-request top_k is dropped too", func(t *testing.T) {
		p := NewAnthropicProvider("k", "claude-opus-5", nil)
		p.SetDefaultOptions(fullConfig())

		body, err := p.PrepareRequest("hi", map[string]interface{}{
			"top_k": 10, "temperature": 0.9,
		})
		if err != nil {
			t.Fatalf("PrepareRequest failed: %v", err)
		}
		assertAbsent(t, decodeBody(t, body), "top_k", "temperature")
	})

	t.Run("unknown future models fail closed", func(t *testing.T) {
		if anthropicAcceptsSamplingParams("claude-something-9") {
			t.Error("an unknown model should not be assumed to accept the deprecated controls")
		}
		if !anthropicAcceptsSamplingParams("claude-opus-4-6-20260101") {
			t.Error("claude-opus-4-6 predates the removal and should accept them")
		}
	})
}

// TestOllamaNestsRunnerOptions covers the bug that made every advanced setter inert:
// Ollama reads sampling from a nested "options" object, and nothing merged the
// provider defaults into the request at all.
func TestOllamaNestsRunnerOptions(t *testing.T) {
	p := NewOllamaProvider("", "llama3.2", nil)
	p.SetDefaultOptions(fullConfig())

	body, err := p.PrepareRequest("hi", map[string]interface{}{})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	decoded := decodeBody(t, body)

	runner, ok := decoded["options"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected a nested \"options\" object, got %v", decoded["options"])
	}
	for key, want := range map[string]float64{
		"temperature": 0.3, "num_predict": 512, "top_p": 0.8, "top_k": 40,
		"min_p": 0.05, "repeat_penalty": 1.1, "repeat_last_n": 64,
		"mirostat": 2, "mirostat_eta": 0.1, "mirostat_tau": 5.0, "tfs_z": 1.0,
		"seed": 7,
	} {
		assertNum(t, runner, key, want)
	}

	// Sampling must not also sit at the top level, where Ollama ignores it; the
	// request-shape fields must not be swept into the nested object.
	assertAbsent(t, decoded, "temperature", "top_k", "min_p", "seed", "num_predict")
	if decoded["model"] != "llama3.2" || decoded["prompt"] != "hi" {
		t.Errorf("model/prompt should stay at the top level, got %v", decoded)
	}
}

// TestOllamaTranslatesMaxTokens verifies the cross-provider name reaches Ollama's
// runner under the name Ollama uses.
func TestOllamaTranslatesMaxTokens(t *testing.T) {
	p := NewOllamaProvider("", "llama3.2", nil)

	body, err := p.PrepareRequest("hi", map[string]interface{}{"max_tokens": 256})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	decoded := decodeBody(t, body)
	runner, ok := decoded["options"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected a nested \"options\" object, got %v", decoded["options"])
	}
	assertNum(t, runner, "num_predict", 256)
	assertAbsent(t, decoded, "max_tokens")
}

// TestOllamaCallerOptionsWin verifies an options map the caller assembled directly
// is preserved and takes precedence over the individually-set defaults.
func TestOllamaCallerOptionsWin(t *testing.T) {
	p := NewOllamaProvider("", "llama3.2", nil)
	p.SetDefaultOptions(fullConfig())

	body, err := p.PrepareRequest("hi", map[string]interface{}{
		"options": map[string]interface{}{"temperature": 0.9, "num_ctx": 4096},
	})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	runner, ok := decodeBody(t, body)["options"].(map[string]interface{})
	if !ok {
		t.Fatal("expected a nested \"options\" object")
	}
	assertNum(t, runner, "temperature", 0.9) // caller's value, not the config's 0.3
	assertNum(t, runner, "num_ctx", 4096)    // preserved, though gollm has no setter
	assertNum(t, runner, "top_k", 40)        // still merged in from the defaults
}

// TestBedrockSamplingPerFamily verifies each model family gets its own field names.
func TestBedrockSamplingPerFamily(t *testing.T) {
	cases := []struct {
		model  string
		want   map[string]float64
		absent []string
	}{
		{
			model: "anthropic.claude-3-5-sonnet-20241022-v2:0",
			want:  map[string]float64{"temperature": 0.3, "top_p": 0.8, "top_k": 40},
		},
		{
			// Meta Llama's body has no top_k field at all.
			model:  "meta.llama3-70b-instruct-v1:0",
			want:   map[string]float64{"temperature": 0.3, "top_p": 0.8},
			absent: []string{"top_k"},
		},
		{
			model: "mistral.mistral-7b-instruct-v0:2",
			want:  map[string]float64{"temperature": 0.3, "top_p": 0.8, "top_k": 40},
		},
		{
			model:  "cohere.command-r-plus-v1:0",
			want:   map[string]float64{"temperature": 0.3, "p": 0.8, "k": 40},
			absent: []string{"top_p", "top_k"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			p := NewBedrockProvider("k", tc.model, nil)
			p.SetDefaultOptions(fullConfig())

			body, err := p.PrepareRequest("hi", map[string]interface{}{})
			if err != nil {
				t.Fatalf("PrepareRequest failed: %v", err)
			}
			decoded := decodeBody(t, body)
			for key, want := range tc.want {
				assertNum(t, decoded, key, want)
			}
			assertAbsent(t, decoded, tc.absent...)
			assertAbsent(t, decoded, "seed", "min_p")
		})
	}
}

// TestBedrockTitanNestsCamelCaseConfig covers Titan's separate body shape.
func TestBedrockTitanNestsCamelCaseConfig(t *testing.T) {
	p := NewBedrockProvider("k", "amazon.titan-text-express-v1", nil)
	p.SetDefaultOptions(fullConfig())

	body, err := p.PrepareRequest("hi", map[string]interface{}{})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	decoded := decodeBody(t, body)
	genConfig, ok := decoded["textGenerationConfig"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected textGenerationConfig, got %v", decoded["textGenerationConfig"])
	}
	assertNum(t, genConfig, "temperature", 0.3)
	assertNum(t, genConfig, "topP", 0.8)
	assertNum(t, genConfig, "maxTokenCount", 512)
	assertAbsent(t, genConfig, "top_p", "topK", "top_k")
}

// TestStripLeavesNonSamplingKeysAlone is the guard on the filter's blast radius:
// it may only delete keys that are sampling parameters somewhere.
func TestStripLeavesNonSamplingKeysAlone(t *testing.T) {
	request := map[string]interface{}{
		"model":       "x",
		"messages":    []string{},
		"stream":      true,
		"transforms":  []string{"reasoning"},
		"provider":    map[string]interface{}{"order": []string{"anthropic"}},
		"extra_body":  map[string]interface{}{"google": "thinking"},
		"temperature": 0.5,
		"top_k":       40,
	}
	stripUnsupportedSampling(request, supportedSampling(openAIChatSamplingParams), "openai", nil)

	for _, key := range []string{"model", "messages", "stream", "transforms", "provider", "extra_body"} {
		if _, ok := request[key]; !ok {
			t.Errorf("%q is not a sampling parameter and should have been left alone", key)
		}
	}
	if _, ok := request["temperature"]; !ok {
		t.Error("temperature is supported here and should have been kept")
	}
	if _, ok := request["top_k"]; ok {
		t.Error("top_k is not an OpenAI parameter and should have been dropped")
	}
}

// TestSamplingWireNamesCoversEveryTable guards the filter against a new table
// introducing a name it would then refuse to strip.
func TestSamplingWireNamesCoversEveryTable(t *testing.T) {
	for _, table := range [][]paramSpec{
		openAIChatSamplingParams, anthropicSamplingParams, mistralSamplingParams,
		openRouterSamplingParams, deepSeekSamplingParams, googleSamplingParams,
		cohereSamplingParams, vllmSamplingParams, lmStudioSamplingParams,
		aliyunSamplingParams, ollamaSamplingParams, bedrockSamplingParams,
	} {
		for _, spec := range table {
			if !samplingWireNames[spec.wire] {
				t.Errorf("%q is emitted by a table but missing from samplingWireNames", spec.wire)
			}
		}
	}
}

// wantStop is the stop list every stop-sequence test sends.
var wantStop = []string{"END", "\n\nUser:"}

func stopConfig() *config.Config {
	cfg := fullConfig()
	cfg.StopSequences = wantStop
	return cfg
}

// assertStop checks that key holds exactly the sequences wantStop carries.
func assertStop(t *testing.T, body map[string]interface{}, key string) {
	t.Helper()
	raw, ok := body[key]
	if !ok {
		t.Errorf("%q missing from request body", key)
		return
	}
	list, ok := raw.([]interface{})
	if !ok {
		t.Errorf("%q = %v (%T), want an array", key, raw, raw)
		return
	}
	if len(list) != len(wantStop) {
		t.Errorf("%q has %d entries, want %d", key, len(list), len(wantStop))
		return
	}
	for i, want := range wantStop {
		if list[i] != want {
			t.Errorf("%q[%d] = %v, want %q", key, i, list[i], want)
		}
	}
}

// TestStopSequencesReachEachProviderName covers the three spellings the same list
// travels under.
func TestStopSequencesReachEachProviderName(t *testing.T) {
	cases := []struct {
		name     string
		provider Provider
		wire     string
		absent   []string
	}{
		{"openai uses stop", NewOpenAIProvider("k", "gpt-4o", nil), "stop", []string{"stop_sequences"}},
		{"anthropic uses stop_sequences", NewAnthropicProvider("k", "claude-sonnet-4-5", nil), "stop_sequences", []string{"stop"}},
		{"mistral uses stop", NewMistralProvider("k", "mistral-large-latest", nil), "stop", []string{"stop_sequences"}},
		{"openrouter uses stop", NewOpenRouterProvider("k", "x/y", nil), "stop", []string{"stop_sequences"}},
		{"groq uses stop", NewGroqProvider("k", "llama-3.3-70b-versatile", nil), "stop", []string{"stop_sequences"}},
		{"deepseek uses stop", NewDeepSeekProvider("k", "deepseek-chat", nil), "stop", []string{"stop_sequences"}},
		{"google uses stop", NewGoogleProvider("k", "gemini-2.0-flash", nil), "stop", []string{"stop_sequences"}},
		{"cohere uses stop_sequences", NewCohereProvider("k", "command-r-plus-08-2024", nil), "stop_sequences", []string{"stop"}},
		{"vllm uses stop", NewVLLMProvider("k", "m", nil), "stop", []string{"stop_sequences"}},
		{"lambda uses stop", NewLambdaProvider("k", "m", nil), "stop", []string{"stop_sequences"}},
		{"aliyun uses stop", NewAliyunProvider("k", "qwen-turbo", nil), "stop", []string{"stop_sequences"}},
		{"lmstudio uses stop", NewLMStudioProvider("k", "m", nil), "stop", []string{"stop_sequences"}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.provider.SetDefaultOptions(stopConfig())
			body, err := tc.provider.PrepareRequest("hi", map[string]interface{}{})
			if err != nil {
				t.Fatalf("PrepareRequest failed: %v", err)
			}
			decoded := decodeBody(t, body)
			assertStop(t, decoded, tc.wire)
			assertAbsent(t, decoded, tc.absent...)
		})
	}
}

// TestOllamaStopSequencesNestUnderOptions verifies stop follows the other runner
// parameters into the nested object rather than sitting at the top level.
func TestOllamaStopSequencesNestUnderOptions(t *testing.T) {
	p := NewOllamaProvider("", "llama3.2", nil)
	p.SetDefaultOptions(stopConfig())

	body, err := p.PrepareRequest("hi", map[string]interface{}{})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	decoded := decodeBody(t, body)
	runner, ok := decoded["options"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected a nested \"options\" object, got %v", decoded["options"])
	}
	assertStop(t, runner, "stop")
	assertAbsent(t, decoded, "stop")
}

// TestBedrockStopSequencesPerFamily covers the per-family names, including the
// family that has no stop field at all.
func TestBedrockStopSequencesPerFamily(t *testing.T) {
	for _, tc := range []struct{ model, wire string }{
		{"anthropic.claude-3-5-sonnet-20241022-v2:0", "stop_sequences"},
		{"mistral.mistral-7b-instruct-v0:2", "stop"},
		{"cohere.command-r-plus-v1:0", "stop_sequences"},
	} {
		t.Run(tc.model, func(t *testing.T) {
			p := NewBedrockProvider("k", tc.model, nil)
			p.SetDefaultOptions(stopConfig())
			body, err := p.PrepareRequest("hi", map[string]interface{}{})
			if err != nil {
				t.Fatalf("PrepareRequest failed: %v", err)
			}
			assertStop(t, decodeBody(t, body), tc.wire)
		})
	}

	t.Run("titan nests camelCase stopSequences", func(t *testing.T) {
		p := NewBedrockProvider("k", "amazon.titan-text-express-v1", nil)
		p.SetDefaultOptions(stopConfig())
		body, err := p.PrepareRequest("hi", map[string]interface{}{})
		if err != nil {
			t.Fatalf("PrepareRequest failed: %v", err)
		}
		genConfig, ok := decodeBody(t, body)["textGenerationConfig"].(map[string]interface{})
		if !ok {
			t.Fatal("expected textGenerationConfig")
		}
		assertStop(t, genConfig, "stopSequences")
	})

	t.Run("meta llama has no stop field", func(t *testing.T) {
		p := NewBedrockProvider("k", "meta.llama3-70b-instruct-v1:0", nil)
		p.SetDefaultOptions(stopConfig())
		body, err := p.PrepareRequest("hi", map[string]interface{}{})
		if err != nil {
			t.Fatalf("PrepareRequest failed: %v", err)
		}
		assertAbsent(t, decodeBody(t, body), "stop", "stop_sequences", "stopSequences")
	})
}

// TestStopSequencesAreRenamedNotDropped covers a caller who sets the key by hand
// under another provider's spelling: the list must survive under the name this API
// uses, rather than being stripped as a foreign parameter.
func TestStopSequencesAreRenamedNotDropped(t *testing.T) {
	t.Run("stop becomes stop_sequences on anthropic", func(t *testing.T) {
		p := NewAnthropicProvider("k", "claude-sonnet-4-5", nil)
		body, err := p.PrepareRequest("hi", map[string]interface{}{
			"max_tokens": 100, "stop": wantStop,
		})
		if err != nil {
			t.Fatalf("PrepareRequest failed: %v", err)
		}
		decoded := decodeBody(t, body)
		assertStop(t, decoded, "stop_sequences")
		assertAbsent(t, decoded, "stop")
	})

	t.Run("stop_sequences becomes stop on mistral", func(t *testing.T) {
		p := NewMistralProvider("k", "mistral-large-latest", nil)
		body, err := p.PrepareRequest("hi", map[string]interface{}{"stop_sequences": wantStop})
		if err != nil {
			t.Fatalf("PrepareRequest failed: %v", err)
		}
		decoded := decodeBody(t, body)
		assertStop(t, decoded, "stop")
		assertAbsent(t, decoded, "stop_sequences")
	})

	t.Run("a bare string widens to a list", func(t *testing.T) {
		p := NewAnthropicProvider("k", "claude-sonnet-4-5", nil)
		body, err := p.PrepareRequest("hi", map[string]interface{}{
			"max_tokens": 100, "stop": "END",
		})
		if err != nil {
			t.Fatalf("PrepareRequest failed: %v", err)
		}
		raw := decodeBody(t, body)["stop_sequences"]
		list, ok := raw.([]interface{})
		if !ok || len(list) != 1 || list[0] != "END" {
			t.Errorf("stop_sequences = %v (%T), want the one-element list [END]", raw, raw)
		}
	})
}

// TestUnsetStopSequencesAreNotForwarded keeps an empty list off the wire, which
// some APIs reject.
func TestUnsetStopSequencesAreNotForwarded(t *testing.T) {
	for _, cfg := range []*config.Config{
		{Temperature: 0.7, MaxTokens: 100},
		{Temperature: 0.7, MaxTokens: 100, StopSequences: []string{}},
	} {
		p := NewOpenRouterProvider("k", "x/y", nil)
		p.SetDefaultOptions(cfg)
		body, err := p.PrepareRequest("hi", map[string]interface{}{})
		if err != nil {
			t.Fatalf("PrepareRequest failed: %v", err)
		}
		assertAbsent(t, decodeBody(t, body), "stop", "stop_sequences")
	}
}

// TestSystemPromptReachesEveryProvider covers the bug that made a system prompt
// inert: gollm's system_prompt key travelled to the wire verbatim, where each API
// either ignored it or rejected it as an unknown parameter.
func TestSystemPromptReachesEveryProvider(t *testing.T) {
	const systemPrompt = "You are a helpful assistant."

	providers := map[string]Provider{
		"ollama":     NewOllamaProvider("", "llama3.2", nil),
		"mistral":    NewMistralProvider("k", "mistral-large-latest", nil),
		"groq":       NewGroqProvider("k", "llama-3.3-70b-versatile", nil),
		"openrouter": NewOpenRouterProvider("k", "x/y", nil),
		"lambda":     NewLambdaProvider("k", "m", nil),
		"vllm":       NewVLLMProvider("k", "m", nil),
		"cohere":     NewCohereProvider("k", "command-r-plus-08-2024", nil),
		"anthropic":  NewAnthropicProvider("k", "claude-sonnet-4-5", nil),
		"openai":     NewOpenAIProvider("k", "gpt-4o", nil),
		"deepseek":   NewDeepSeekProvider("k", "deepseek-chat", nil),
		"google":     NewGoogleProvider("k", "gemini-2.0-flash", nil),
		"aliyun":     NewAliyunProvider("k", "qwen-turbo", nil),
		"lmstudio":   NewLMStudioProvider("k", "m", nil),
		"bedrock":    NewBedrockProvider("k", "anthropic.claude-3-5-sonnet-20241022-v2:0", nil),
	}

	for name, p := range providers {
		t.Run(name, func(t *testing.T) {
			body, err := p.PrepareRequest("hi", map[string]interface{}{
				"system_prompt": systemPrompt,
			})
			if err != nil {
				t.Fatalf("PrepareRequest failed: %v", err)
			}
			if _, leaked := decodeBody(t, body)["system_prompt"]; leaked {
				t.Errorf("system_prompt is a gollm control key and must not reach the wire:\n%s", body)
			}
			if !strings.Contains(string(body), systemPrompt) {
				t.Errorf("the system prompt was dropped entirely:\n%s", body)
			}
		})
	}
}

// TestOllamaSystemPromptUsesNativeField pins the field Ollama actually reads.
func TestOllamaSystemPromptUsesNativeField(t *testing.T) {
	p := NewOllamaProvider("", "llama3.2", nil)

	body, err := p.PrepareRequest("What is 2+2?", map[string]interface{}{
		"system_prompt": "Answer in pirate speak.",
	})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	decoded := decodeBody(t, body)
	if decoded["system"] != "Answer in pirate speak." {
		t.Errorf("system = %v, want the system prompt on Ollama's native field", decoded["system"])
	}
	assertAbsent(t, decoded, "system_prompt")
}

// TestSystemPromptLeadsTheMessages verifies ordering on the OpenAI-shaped APIs: a
// system message that follows the user turn is not a system prompt.
func TestSystemPromptLeadsTheMessages(t *testing.T) {
	for name, p := range map[string]Provider{
		"mistral": NewMistralProvider("k", "mistral-large-latest", nil),
		"groq":    NewGroqProvider("k", "llama-3.3-70b-versatile", nil),
		"cohere":  NewCohereProvider("k", "command-r-plus-08-2024", nil),
		"aliyun":  NewAliyunProvider("k", "qwen-turbo", nil),
	} {
		t.Run(name, func(t *testing.T) {
			body, err := p.PrepareRequest("hi", map[string]interface{}{"system_prompt": "SYS"})
			if err != nil {
				t.Fatalf("PrepareRequest failed: %v", err)
			}
			messages, ok := decodeBody(t, body)["messages"].([]interface{})
			if !ok || len(messages) != 2 {
				t.Fatalf("expected two messages, got %v", decodeBody(t, body)["messages"])
			}
			first, _ := messages[0].(map[string]interface{})
			if first["role"] != "system" || first["content"] != "SYS" {
				t.Errorf("first message = %v, want the system prompt", first)
			}
			second, _ := messages[1].(map[string]interface{})
			if second["role"] != "user" || second["content"] != "hi" {
				t.Errorf("second message = %v, want the user turn", second)
			}
		})
	}
}

// TestAnthropicClaude3AcceptsSampling pins the model-id shapes apart. Claude 3
// numbers before the model name and Claude 4 after it, so one substring list cannot
// match both — and getting it wrong silently stripped temperature from every Claude 3
// model, this library's default among them.
func TestAnthropicClaude3AcceptsSampling(t *testing.T) {
	for _, model := range []string{
		"claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
		"claude-3-5-sonnet-20241022", "claude-3-5-haiku-latest", "claude-3-7-sonnet-latest",
		"claude-sonnet-4-0", "claude-opus-4-1", "claude-haiku-4-5",
		"claude-sonnet-4-6", "claude-opus-4-6", "claude-2.1",
	} {
		if !anthropicAcceptsSamplingParams(model) {
			t.Errorf("%s predates the deprecation and should accept temperature/top_p/top_k", model)
		}
	}

	for _, model := range []string{
		"claude-opus-4-7", "claude-opus-4-8", "claude-opus-5",
		"claude-sonnet-5", "claude-fable-5", "claude-mythos-5",
	} {
		if anthropicAcceptsSamplingParams(model) {
			t.Errorf("%s is after Claude Opus 4.6 and rejects those parameters", model)
		}
	}
}

// TestAnthropicDefaultModelKeepsTemperature is the end-to-end version of the above,
// on the model gollm defaults to.
func TestAnthropicDefaultModelKeepsTemperature(t *testing.T) {
	p := NewAnthropicProvider("k", "claude-3-5-haiku-latest", nil)
	p.SetDefaultOptions(fullConfig())

	body, err := p.PrepareRequest("hi", map[string]interface{}{})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	decoded := decodeBody(t, body)
	assertNum(t, decoded, "temperature", 0.3)
	assertNum(t, decoded, "top_p", 0.8)
	assertNum(t, decoded, "top_k", 40)
}

// TestAnthropicStreamRequestUsesTheSameMerge covers the one builder that hand-rolled
// its own option handling: it read only max_tokens and temperature from the
// per-request options and nothing from the provider defaults.
func TestAnthropicStreamRequestUsesTheSameMerge(t *testing.T) {
	t.Run("configured defaults reach a streamed call", func(t *testing.T) {
		p := NewAnthropicProvider("k", "claude-sonnet-4-5", nil)
		p.SetDefaultOptions(stopConfig())

		body, err := p.PrepareStreamRequest("hi", map[string]interface{}{})
		if err != nil {
			t.Fatalf("PrepareStreamRequest failed: %v", err)
		}
		decoded := decodeBody(t, body)
		assertNum(t, decoded, "max_tokens", 512) // not the hard-coded 1024
		assertNum(t, decoded, "temperature", 0.3)
		assertNum(t, decoded, "top_k", 40)
		assertStop(t, decoded, "stop_sequences")
		if decoded["stream"] != true {
			t.Errorf("stream = %v, want true", decoded["stream"])
		}
	})

	t.Run("a streamed call strips what the model rejects", func(t *testing.T) {
		p := NewAnthropicProvider("k", "claude-opus-5", nil)
		p.SetDefaultOptions(fullConfig())

		body, err := p.PrepareStreamRequest("hi", map[string]interface{}{
			"temperature": 0.9, "top_k": 7,
		})
		if err != nil {
			t.Fatalf("PrepareStreamRequest failed: %v", err)
		}
		decoded := decodeBody(t, body)
		assertAbsent(t, decoded, "temperature", "top_k", "top_p")
		assertNum(t, decoded, "max_tokens", 512)
		if decoded["stream"] != true {
			t.Errorf("stream = %v, want true", decoded["stream"])
		}
	})

	t.Run("the caller's options map is not mutated", func(t *testing.T) {
		p := NewAnthropicProvider("k", "claude-sonnet-4-5", nil)
		options := map[string]interface{}{"system_prompt": "SYS", "max_tokens": 256}

		if _, err := p.PrepareStreamRequest("hi", options); err != nil {
			t.Fatalf("PrepareStreamRequest failed: %v", err)
		}
		if _, ok := options["system_prompt"]; !ok {
			t.Error("PrepareStreamRequest deleted system_prompt from the caller's map")
		}
		if _, ok := options["max_tokens"]; !ok {
			t.Error("PrepareStreamRequest deleted max_tokens from the caller's map")
		}
	})
}

// TestPerRequestStopSequencesBeatProviderDefaults covers the precedence inversion:
// a default written under the API's own name must not outrank a stop list the caller
// passed for this one call under another spelling.
func TestPerRequestStopSequencesBeatProviderDefaults(t *testing.T) {
	p := NewAnthropicProvider("k", "claude-sonnet-4-5", nil)
	p.SetOption("stop_sequences", []string{"DEFAULT"})
	p.SetOption("max_tokens", 100)

	body, err := p.PrepareRequest("hi", map[string]interface{}{"stop": []string{"PERCALL"}})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	raw := decodeBody(t, body)["stop_sequences"]
	list, ok := raw.([]interface{})
	if !ok || len(list) != 1 || list[0] != "PERCALL" {
		t.Errorf("stop_sequences = %v, want the per-request [PERCALL]", raw)
	}
}

// TestCallerSuppliedMessagesSurvive verifies that assembling a messages array by hand
// still works: these providers copied options into the body verbatim, so a caller
// could always drive a multi-turn exchange through the single-prompt path.
func TestCallerSuppliedMessagesSurvive(t *testing.T) {
	caller := []map[string]interface{}{
		{"role": "user", "content": "FIRST"},
		{"role": "assistant", "content": "SECOND"},
	}

	for name, p := range map[string]Provider{
		"mistral": NewMistralProvider("k", "mistral-large-latest", nil),
		"groq":    NewGroqProvider("k", "llama-3.3-70b-versatile", nil),
		"aliyun":  NewAliyunProvider("k", "qwen-turbo", nil),
	} {
		t.Run(name, func(t *testing.T) {
			body, err := p.PrepareRequest("ignored", map[string]interface{}{"messages": caller})
			if err != nil {
				t.Fatalf("PrepareRequest failed: %v", err)
			}
			messages, ok := decodeBody(t, body)["messages"].([]interface{})
			if !ok || len(messages) != 2 {
				t.Fatalf("expected the caller's two messages, got %v", decodeBody(t, body)["messages"])
			}
			first, _ := messages[0].(map[string]interface{})
			if first["content"] != "FIRST" {
				t.Errorf("first message = %v, want the caller's own", first)
			}
		})
	}
}

// TestProviderLevelSystemPromptIsPromoted covers a system prompt set on the provider
// rather than passed per request: it must still become a system message, not vanish.
func TestProviderLevelSystemPromptIsPromoted(t *testing.T) {
	for name, p := range map[string]Provider{
		"mistral": NewMistralProvider("k", "mistral-large-latest", nil),
		"groq":    NewGroqProvider("k", "llama-3.3-70b-versatile", nil),
		"aliyun":  NewAliyunProvider("k", "qwen-turbo", nil),
		"cohere":  NewCohereProvider("k", "command-r-plus-08-2024", nil),
	} {
		t.Run(name, func(t *testing.T) {
			p.SetOption("system_prompt", "PROVIDER_SYS")

			body, err := p.PrepareRequest("hi", map[string]interface{}{})
			if err != nil {
				t.Fatalf("PrepareRequest failed: %v", err)
			}
			decoded := decodeBody(t, body)
			assertAbsent(t, decoded, "system_prompt")
			messages, ok := decoded["messages"].([]interface{})
			if !ok || len(messages) != 2 {
				t.Fatalf("expected a system turn and a user turn, got %v", decoded["messages"])
			}
			first, _ := messages[0].(map[string]interface{})
			if first["role"] != "system" || first["content"] != "PROVIDER_SYS" {
				t.Errorf("first message = %v, want the provider-level system prompt", first)
			}
		})
	}

	t.Run("per-request wins over provider level", func(t *testing.T) {
		p := NewMistralProvider("k", "mistral-large-latest", nil)
		p.SetOption("system_prompt", "PROVIDER_SYS")

		body, err := p.PrepareRequest("hi", map[string]interface{}{"system_prompt": "CALL_SYS"})
		if err != nil {
			t.Fatalf("PrepareRequest failed: %v", err)
		}
		messages, _ := decodeBody(t, body)["messages"].([]interface{})
		first, _ := messages[0].(map[string]interface{})
		if first["content"] != "CALL_SYS" {
			t.Errorf("first message = %v, want the per-request system prompt", first)
		}
	})
}

// TestPerRequestUnsupportedSamplingIsStripped covers the other door into the filter:
// options a caller sets for one call, rather than defaults from a table.
func TestPerRequestUnsupportedSamplingIsStripped(t *testing.T) {
	for name, tc := range map[string]struct {
		provider Provider
		keep     string
	}{
		"vllm":    {NewVLLMProvider("k", "m", nil), "min_p"},
		"mistral": {NewMistralProvider("k", "m", nil), "top_p"},
		"groq":    {NewGroqProvider("k", "m", nil), "top_p"},
	} {
		t.Run(name, func(t *testing.T) {
			body, err := tc.provider.PrepareRequest("hi", map[string]interface{}{
				"mirostat": 2, "tfs_z": 1.0, "repeat_last_n": 64,
				tc.keep: 0.5,
			})
			if err != nil {
				t.Fatalf("PrepareRequest failed: %v", err)
			}
			decoded := decodeBody(t, body)
			assertAbsent(t, decoded, "mirostat", "tfs_z", "repeat_last_n")
			assertNum(t, decoded, tc.keep, 0.5)
		})
	}
}

// TestGenericAnthropicShapeUsesSystemField verifies the Anthropic-shaped generic
// provider takes the system prompt in its top-level field. Anthropic accepts only
// user and assistant roles inside messages, so the OpenAI trick of prepending a
// "system"-role message would be rejected rather than honored.
func TestGenericAnthropicShapeUsesSystemField(t *testing.T) {
	registry := GetDefaultRegistry()
	cfg, ok := registry.GetProviderConfig("anthropic")
	if !ok {
		t.Fatal("anthropic provider config missing from the registry")
	}
	p := &GenericProvider{
		model:   "claude-3-5-sonnet-20241022",
		config:  cfg,
		options: map[string]interface{}{},
		logger:  utils.NewLogger(utils.LogLevelWarn),
	}

	body, err := p.PrepareRequest("hi", map[string]interface{}{"system_prompt": "SYS"})
	if err != nil {
		t.Fatalf("PrepareRequest failed: %v", err)
	}
	decoded := decodeBody(t, body)
	if decoded["system"] != "SYS" {
		t.Errorf("system = %v, want the system prompt in Anthropic's own field", decoded["system"])
	}
	assertAbsent(t, decoded, "system_prompt")

	messages, ok := decoded["messages"].([]interface{})
	if !ok || len(messages) != 1 {
		t.Fatalf("expected only the user turn, got %v", decoded["messages"])
	}
	if first, _ := messages[0].(map[string]interface{}); first["role"] != "user" {
		t.Errorf("messages[0].role = %v, want user; Anthropic rejects a system role here", first["role"])
	}
}

// TestAnthropicThinkingSkippedBeforeClaude37 covers models with no thinking mode.
// Extended thinking arrived with Claude 3.7, so a reasoning_effort on anything older
// must not turn into a thinking configuration: both shapes are rejected there, and
// the previous model list never matched a Claude 3 id, so every one of them was sent
// adaptive thinking.
func TestAnthropicThinkingSkippedBeforeClaude37(t *testing.T) {
	for _, model := range []string{
		"claude-2.1", "claude-instant-1.2",
		"claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
		"claude-3-5-sonnet-20241022", "claude-3-5-haiku-latest",
	} {
		t.Run(model, func(t *testing.T) {
			if anthropicSupportsThinking(model) {
				t.Fatalf("%s predates extended thinking", model)
			}

			p := NewAnthropicProvider("k", model, nil)
			p.SetOption("max_tokens", 4096)

			body, err := p.PrepareRequest("hi", map[string]interface{}{"reasoning_effort": "high"})
			if err != nil {
				t.Fatalf("PrepareRequest failed: %v", err)
			}
			assertAbsent(t, decodeBody(t, body), "thinking", "output_config")
		})
	}
}

// TestAnthropicThinkingModeByModel pins which thinking mode each generation gets.
func TestAnthropicThinkingModeByModel(t *testing.T) {
	cases := []struct {
		model string
		mode  string // "budget", "adaptive", or "" for none
	}{
		{"claude-3-5-sonnet-20241022", ""},
		{"claude-3-7-sonnet-20250219", "budget"},
		{"claude-sonnet-4-5", "budget"},
		{"claude-haiku-4-5", "budget"},
		{"claude-opus-4-6", "adaptive"},
		{"claude-opus-5", "adaptive"},
		{"claude-something-future", "adaptive"},
	}

	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			p := NewAnthropicProvider("k", tc.model, nil)
			p.SetOption("max_tokens", 8192)

			body, err := p.PrepareRequest("hi", map[string]interface{}{"reasoning_effort": "high"})
			if err != nil {
				t.Fatalf("PrepareRequest failed: %v", err)
			}
			decoded := decodeBody(t, body)

			if tc.mode == "" {
				assertAbsent(t, decoded, "thinking")
				return
			}
			thinking, ok := decoded["thinking"].(map[string]interface{})
			if !ok {
				t.Fatalf("thinking = %v, want an object", decoded["thinking"])
			}
			switch tc.mode {
			case "budget":
				if thinking["type"] != "enabled" {
					t.Errorf("thinking.type = %v, want enabled", thinking["type"])
				}
				if _, has := thinking["budget_tokens"]; !has {
					t.Error("extended thinking needs budget_tokens")
				}
			case "adaptive":
				if thinking["type"] != "adaptive" {
					t.Errorf("thinking.type = %v, want adaptive", thinking["type"])
				}
				if _, has := thinking["budget_tokens"]; has {
					t.Error("adaptive thinking rejects budget_tokens")
				}
			}
		})
	}
}
