package providers

import (
	"encoding/json"
	"testing"

	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

func functionTool(name string) utils.Tool {
	return utils.Tool{Type: "function", Function: utils.Function{Name: name, Description: "d"}}
}

// reasoningEffortIn returns the body's reasoning_effort and whether it was present
// at all. Presence matters: for gpt-5.6, omitting is not the same as "none".
func reasoningEffortIn(t *testing.T, body []byte) (string, bool) {
	t.Helper()
	var request map[string]interface{}
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatalf("body is not valid JSON: %v", err)
	}
	v, present := request["reasoning_effort"]
	if !present {
		return "", false
	}
	s, _ := v.(string)
	return s, true
}

// TestChatToolsCarveOutPinsEffortNone: a tool-carrying request must not be rejected
// with "Function tools with reasoning_effort are not supported for <model> in
// /v1/chat/completions".
func TestChatToolsCarveOutPinsEffortNone(t *testing.T) {
	tools := map[string]interface{}{"tools": []utils.Tool{functionTool("lookup")}}

	// An explicit effort is what these models reject, so it must become "none".
	t.Run("caller set an effort", func(t *testing.T) {
		for _, model := range []string{"gpt-5.4", "gpt-5.4-mini", "gpt-5.5", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"} {
			t.Run(model, func(t *testing.T) {
				p := NewOpenAIProvider("sk-test", model, nil)
				opts := map[string]interface{}{"reasoning_effort": "high"}
				for k, v := range tools {
					opts[k] = v
				}
				body, err := p.PrepareRequest("hi", opts)
				if err != nil {
					t.Fatalf("PrepareRequest returned error: %v", err)
				}
				effort, present := reasoningEffortIn(t, body)
				if !present || effort != string(types.ReasoningEffortNone) {
					t.Errorf("reasoning_effort = %q (present=%v), want none", effort, present)
				}
			})
		}
	})

	// The gpt-5.6 line reasons by default, so silence is not a carve-out: the
	// parameter has to be sent even though the caller set nothing.
	t.Run("no effort set, model reasons by default", func(t *testing.T) {
		for _, model := range []string{"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"} {
			t.Run(model, func(t *testing.T) {
				p := NewOpenAIProvider("sk-test", model, nil)
				body, err := p.PrepareRequest("hi", tools)
				if err != nil {
					t.Fatalf("PrepareRequest returned error: %v", err)
				}
				effort, present := reasoningEffortIn(t, body)
				if !present || effort != string(types.ReasoningEffortNone) {
					t.Errorf("reasoning_effort = %q (present=%v), want none", effort, present)
				}
			})
		}
	})

	// gpt-5.4 and 5.5 fail only on an explicit effort. With none supplied the
	// request already succeeds, so pinning "none" would disable the model's default
	// reasoning to avoid an error that would not have happened.
	t.Run("no effort set, model needs no carve-out", func(t *testing.T) {
		for _, model := range []string{"gpt-5.4", "gpt-5.4-mini", "gpt-5.5"} {
			t.Run(model, func(t *testing.T) {
				p := NewOpenAIProvider("sk-test", model, nil)
				body, err := p.PrepareRequest("hi", tools)
				if err != nil {
					t.Fatalf("PrepareRequest returned error: %v", err)
				}
				if effort, present := reasoningEffortIn(t, body); present {
					t.Errorf("sent reasoning_effort=%q; default reasoning must be left alone", effort)
				}
			})
		}
	})
}

// TestChatToolsCarveOutLeavesReasoningAlone: no tools, or a model outside the
// affected set, must keep full reasoning.
func TestChatToolsCarveOutLeavesReasoningAlone(t *testing.T) {
	tests := []struct {
		name    string
		model   string
		options map[string]interface{}
	}{
		{
			// The overwhelmingly common case, and the whole point of not routing:
			// a tool-free request keeps full reasoning on the fast transport.
			name:    "affected model without tools",
			model:   "gpt-5.6-sol",
			options: map[string]interface{}{"reasoning_effort": "high"},
		},
		{
			name:    "below the bound with tools",
			model:   "gpt-5.3",
			options: map[string]interface{}{"tools": []utils.Tool{functionTool("lookup")}, "reasoning_effort": "high"},
		},
		{
			name:    "o-series with tools",
			model:   "o3",
			options: map[string]interface{}{"tools": []utils.Tool{functionTool("lookup")}, "reasoning_effort": "high"},
		},
		{
			// web_search is filtered out of the Chat body, so such a request sends
			// no tools and is not subject to the restriction.
			name:    "affected model with only web_search",
			model:   "gpt-5.6-sol",
			options: map[string]interface{}{"tools": []utils.Tool{{Type: "web_search"}}, "reasoning_effort": "high"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := NewOpenAIProvider("sk-test", tt.model, nil)
			body, err := p.PrepareRequest("hi", tt.options)
			if err != nil {
				t.Fatalf("PrepareRequest returned error: %v", err)
			}
			effort, present := reasoningEffortIn(t, body)
			if !present {
				t.Fatalf("reasoning_effort was dropped entirely; want the caller's level preserved")
			}
			if effort == string(types.ReasoningEffortNone) {
				t.Errorf("reasoning was disabled for a request that does not need the carve-out")
			}
		})
	}
}

// TestChatToolsCarveOutCoversEveryRequestPath: a tool-carrying request must be safe
// however it was built, so the carve-out cannot be wired into only some paths.
func TestChatToolsCarveOutCoversEveryRequestPath(t *testing.T) {
	opts := map[string]interface{}{
		"tools":            []utils.Tool{functionTool("lookup")},
		"reasoning_effort": "high",
	}
	schema := map[string]interface{}{"type": "object", "properties": map[string]interface{}{}}
	msgs := []types.MemoryMessage{{Role: "user", Content: "hi"}}

	paths := map[string]func(p Provider) ([]byte, error){
		"PrepareRequest": func(p Provider) ([]byte, error) {
			return p.PrepareRequest("hi", opts)
		},
		"PrepareRequestWithSchema": func(p Provider) ([]byte, error) {
			return p.PrepareRequestWithSchema("hi", opts, schema)
		},
		"PrepareRequestWithMessages": func(p Provider) ([]byte, error) {
			return p.PrepareRequestWithMessages(msgs, opts)
		},
		"PrepareRequestWithMessagesAndSchema": func(p Provider) ([]byte, error) {
			return p.PrepareRequestWithMessagesAndSchema(msgs, opts, schema)
		},
		"PrepareStreamRequest": func(p Provider) ([]byte, error) {
			return p.PrepareStreamRequest("hi", opts)
		},
	}

	for name, build := range paths {
		t.Run(name, func(t *testing.T) {
			p := NewOpenAIProvider("sk-test", "gpt-5.6-sol", nil)
			body, err := build(p)
			if err != nil {
				t.Fatalf("%s returned error: %v", name, err)
			}
			effort, present := reasoningEffortIn(t, body)
			if !present || effort != string(types.ReasoningEffortNone) {
				t.Errorf("%s: reasoning_effort = %q (present=%v), want none; "+
					"this request would be rejected by the API. body=%s",
					name, effort, present, body)
			}
		})
	}
}
