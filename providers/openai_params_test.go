package providers

import (
	"encoding/json"
	"testing"

	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

func decodeRequest(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()
	var req map[string]interface{}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("unmarshal request: %v", err)
	}
	return req
}

// tool_choice was stripped for every o-series and GPT-5 model on the belief that they reject it.
// They don't — the support matrix marks Functions/Tools available across both families, and GPT-5
// extends tool_choice rather than dropping it. Stripping it degraded "required" to auto silently:
// the request succeeds and returns a plausible answer that simply ignored the forced tool.
func TestToolChoiceSurvivesOnReasoningModels(t *testing.T) {
	tools := []utils.Tool{{
		Type: "function",
		Function: utils.Function{
			Name:        "get_weather",
			Description: "w",
			Parameters:  map[string]interface{}{"type": "object"},
		},
	}}

	for _, model := range []string{"gpt-5", "gpt-5.4", "o3", "o4-mini", "gpt-4o"} {
		t.Run(model, func(t *testing.T) {
			p := NewOpenAIProvider("key", model, nil).(*OpenAIProvider)
			body, err := p.PrepareRequest("hi", map[string]interface{}{
				"tools": tools, "tool_choice": "required",
			})
			if err != nil {
				t.Fatalf("PrepareRequest: %v", err)
			}
			if got := decodeRequest(t, body)["tool_choice"]; got != "required" {
				t.Errorf("tool_choice = %v, want %q — a dropped tool_choice fails silently", got, "required")
			}
		})
	}

	// The first-generation o1 previews genuinely have no tool support.
	p := NewOpenAIProvider("key", "o1-mini", nil).(*OpenAIProvider)
	body, _ := p.PrepareRequest("hi", map[string]interface{}{"tools": tools, "tool_choice": "required"})
	if _, present := decodeRequest(t, body)["tool_choice"]; present {
		t.Error("tool_choice reached o1-mini, which does not support tools")
	}
}

// Reasoning models reject the whole sampling family, not temperature alone, and any one of them
// fails the entire request.
func TestReasoningModelsStripEverySamplingParam(t *testing.T) {
	sampling := map[string]interface{}{
		"temperature": 0.7, "top_p": 0.9, "presence_penalty": 0.1,
		"frequency_penalty": 0.2, "logprobs": true, "top_logprobs": 3,
		"logit_bias": map[string]interface{}{"1": 1},
	}

	p := NewOpenAIProvider("key", "gpt-5", nil).(*OpenAIProvider)
	body, err := p.PrepareRequest("hi", sampling)
	if err != nil {
		t.Fatalf("PrepareRequest: %v", err)
	}
	req := decodeRequest(t, body)
	for key := range sampling {
		if _, present := req[key]; present {
			t.Errorf("%q reached a reasoning model that rejects it: %s", key, body)
		}
	}

	// A non-reasoning model must keep them, or this is just deleting useful parameters.
	p4 := NewOpenAIProvider("key", "gpt-4o", nil).(*OpenAIProvider)
	body4, _ := p4.PrepareRequest("hi", sampling)
	if got := decodeRequest(t, body4)["top_p"]; got != 0.9 {
		t.Errorf("top_p = %v on gpt-4o, want 0.9 — non-reasoning models still accept sampling", got)
	}
}

// The effort scale is cross-provider; which levels an OpenAI model accepts depends on its minor
// version. Sending an unsupported level is a 400, so the ends clamp inward — behaviour the enum
// has always documented and nothing implemented.
func TestReasoningEffortClampsPerModel(t *testing.T) {
	cases := []struct {
		model, requested, want string
	}{
		{"gpt-5.6", "max", "max"},               // max exists only from 5.6
		{"gpt-5.4", "max", "xhigh"},             // no max; xhigh is the nearest it takes
		{"gpt-5", "max", "high"},                // neither; clamp to high
		{"gpt-5.4", "xhigh", "xhigh"},           // xhigh from 5.4
		{"gpt-5.1", "xhigh", "high"},            // not on 5.1
		{"o3", "xhigh", "high"},                 // o-series tops out at high
		{"gpt-5.1-codex-max", "xhigh", "xhigh"}, // the model that introduced it
		{"gpt-5", "minimal", "minimal"},         // minimal only on the original GPT-5
		{"gpt-5.1", "minimal", "low"},           // dropped from 5.1 onward
		{"gpt-5-codex", "minimal", "low"},       // codex never had it
		{"gpt-5.1", "none", "none"},             // none from 5.1
		{"gpt-5", "none", "minimal"},            // no none; minimal is the closest
		{"o3", "none", "low"},                   // neither; lowest real level
		{"gpt-5-pro", "low", "high"},            // pro runs at high whatever is asked
		{"o3", "medium", "medium"},              // the middle of the scale is universal
	}

	for _, tc := range cases {
		got, ok := normalizeOpenAIReasoningEffort(tc.model, tc.requested)
		if !ok || got != tc.want {
			t.Errorf("normalize(%q, %q) = %q,%v; want %q,true", tc.model, tc.requested, got, ok, tc.want)
		}
	}

	// Models that take no effort at all drop the parameter rather than clamp it.
	for _, model := range []string{"gpt-4o", "o1-mini", "gpt-5-chat-latest"} {
		if _, ok := normalizeOpenAIReasoningEffort(model, "high"); ok {
			t.Errorf("%q accepted reasoning_effort, but it does not support the parameter", model)
		}
	}
}

// gpt-5-chat-latest is the non-reasoning GPT-5: the API answers "Invalid 'reasoning_effort' for
// non-reasoning model". Matching on the gpt-5 prefix alone swept it in.
func TestGPT5ChatVariantsRejectReasoningParams(t *testing.T) {
	if modelNeedsReasoningEffort("gpt-5-chat-latest") {
		t.Error("gpt-5-chat-latest is non-reasoning; reasoning_effort is invalid for it")
	}
	if modelSupportsVerbosity("gpt-5-chat") {
		t.Error("verbosity is a GPT-5 reasoning feature; gpt-5-chat rejects it")
	}
	// Later chat variants DO reason, so the exclusion must not be a blanket "chat" match.
	if !modelNeedsReasoningEffort("gpt-5.1-chat") {
		t.Error("gpt-5.1-chat supports reasoning effort and must keep it")
	}
}

// codex-mini reasons but matches none of the naming patterns; o1-mini is the one reasoning model
// with no effort parameter at all.
func TestReasoningEffortEdgeModels(t *testing.T) {
	if !modelNeedsReasoningEffort("codex-mini-latest") {
		t.Error("codex-mini supports reasoning_effort; dropping it silently loses the setting")
	}
	if modelNeedsReasoningEffort("o1-mini") {
		t.Error("o1-mini does not accept reasoning_effort")
	}
}

// The Responses API takes reasoning.effort, not a flat reasoning_effort. Sent flat it is an
// unknown parameter and the whole request fails.
func TestResponsesAPINestsReasoningEffort(t *testing.T) {
	p := NewOpenAIResponsesProvider("key", "gpt-5", nil).(*OpenAIResponsesProvider)
	p.SetOption("reasoning_effort", types.ReasoningEffortHigh)

	body, err := p.PrepareRequest("hi", nil)
	if err != nil {
		t.Fatalf("PrepareRequest: %v", err)
	}
	req := decodeRequest(t, body)
	if _, flat := req["reasoning_effort"]; flat {
		t.Errorf("reasoning_effort sent flat; the Responses API rejects it there: %s", body)
	}
	reasoning, ok := req["reasoning"].(map[string]interface{})
	if !ok {
		t.Fatalf("no reasoning object: %s", body)
	}
	if reasoning["effort"] != "high" {
		t.Errorf("reasoning.effort = %v, want %q", reasoning["effort"], "high")
	}
}

// Non-OpenAI providers embed OpenAIProvider for its wire format but bring their own catalogue.
// OpenAI's per-model parameter rules say nothing about their model ids and must not be applied.
func TestEmbeddedProvidersKeepTheirOwnParameters(t *testing.T) {
	for _, model := range []string{"gemini-2.5-flash", "deepseek-reasoner"} {
		if got, ok := normalizeOpenAIReasoningEffort(model, "high"); !ok || got != "high" {
			t.Errorf("normalize(%q, high) = %q,%v; a non-OpenAI model must pass through untouched", model, got, ok)
		}
		if isOpenAIFamilyModel(model) {
			t.Errorf("%q must not be treated as an OpenAI model", model)
		}
	}
}

// o3-pro and codex-mini are Responses-only models that do not support parallel tool calls, so the
// gate matters more on that path than on Chat Completions — where it was originally applied.
func TestParallelToolCallsGatedOnResponsesPath(t *testing.T) {
	for _, model := range []string{"o3-pro", "codex-mini-latest", "o4-mini"} {
		merged := mergeOpenAIResponsesOptions(model, nil,
			map[string]interface{}{"parallel_tool_calls": true}, nil)
		if _, present := merged["parallel_tool_calls"]; present {
			t.Errorf("parallel_tool_calls reached %q, which does not support it", model)
		}
	}
	// GPT-5 does support it, so the gate must not be a blanket strip.
	merged := mergeOpenAIResponsesOptions("gpt-5", nil,
		map[string]interface{}{"parallel_tool_calls": true}, nil)
	if merged["parallel_tool_calls"] != true {
		t.Error("parallel_tool_calls dropped for gpt-5, which supports it")
	}
}

// gpt-5-pro runs only at "high", but pinning must not turn a nonsense value into a valid request —
// every other model drops an unrecognized level, and this one has to agree.
func TestPinnedEffortModelStillValidatesTheLevel(t *testing.T) {
	if got, ok := normalizeOpenAIReasoningEffort("gpt-5-pro", "low"); !ok || got != "high" {
		t.Errorf("normalize(gpt-5-pro, low) = %q,%v; want high,true", got, ok)
	}
	if _, ok := normalizeOpenAIReasoningEffort("gpt-5-pro", "bogus"); ok {
		t.Error("gpt-5-pro accepted an unrecognized effort level; it must be dropped like anywhere else")
	}
}

// Google and DeepSeek embed OpenAIProvider for its wire format but bring their own catalogue.
// OpenAI's per-model rules must not delete their options — the same guard reasoning_effort and the
// sampling family already have.
func TestVerbosityUntouchedForEmbeddedProviders(t *testing.T) {
	for _, model := range []string{"gemini-2.5-flash", "deepseek-reasoner"} {
		opts := map[string]interface{}{"verbosity": "low"}
		applyOpenAIVerbosity(model, opts)
		if opts["verbosity"] != "low" {
			t.Errorf("verbosity dropped for %q by OpenAI's rules, which do not govern it", model)
		}
	}
	// It is still enforced for OpenAI's own models, or the guard would be a blanket bypass.
	opts := map[string]interface{}{"verbosity": "low"}
	applyOpenAIVerbosity("gpt-4o", opts)
	if _, present := opts["verbosity"]; present {
		t.Error("verbosity reached gpt-4o, which rejects it")
	}
}
