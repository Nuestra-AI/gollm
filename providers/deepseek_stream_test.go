package providers

import (
	"encoding/json"
	"testing"
)

// Fix #4: DeepSeek (which embeds OpenAIProvider) must emit the system prompt with
// role "system", not the inherited "developer" default DeepSeek doesn't recognize.
func TestDeepSeekSystemRole(t *testing.T) {
	p := NewDeepSeekProvider("k", "deepseek-chat", nil)
	msgs, opts := structuredFixture()
	body, err := p.PrepareRequestWithMessages(msgs, opts)
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	var req struct {
		Messages []struct {
			Role string `json:"role"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(req.Messages) == 0 || req.Messages[0].Role != "system" {
		t.Errorf("expected leading system message, got %+v", req.Messages)
	}
}
