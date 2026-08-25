package gollm

import (
	"testing"

	"github.com/teilomillet/gollm/config"
)

// TestWithOpenAIToolReasoningOption: the exported option and its re-exports must
// set the field callers expect.
func TestWithOpenAIToolReasoningOption(t *testing.T) {
	cfg := &config.Config{}
	WithOpenAIToolReasoning(ToolReasoningPreferQuality)(cfg)
	if cfg.ToolReasoning != config.ToolReasoningPreferQuality {
		t.Errorf("ToolReasoning = %q, want %q", cfg.ToolReasoning, config.ToolReasoningPreferQuality)
	}

	WithOpenAIToolReasoning(ToolReasoningPreferSpeed)(cfg)
	if cfg.ToolReasoning != config.ToolReasoningPreferSpeed {
		t.Errorf("ToolReasoning = %q, want %q", cfg.ToolReasoning, config.ToolReasoningPreferSpeed)
	}
}
