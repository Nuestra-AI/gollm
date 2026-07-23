package gollm

import (
	"context"
	"testing"

	"github.com/teilomillet/gollm/types"
)

// A usage recorder is only useful if a downstream service can actually install one. The observer
// hangs off *llm.LLMImpl, but callers hold a gollm.LLM, and enabling memory swaps the concrete type
// underneath them — so this pins the hook to the public interface for both constructions.
func TestUsageObserverReachableFromPublicAPI(t *testing.T) {
	cases := []struct {
		name string
		opts []ConfigOption
	}{
		{"plain", []ConfigOption{SetProvider("vllm"), SetModel("qwen")}},
		// Memory wraps the base LLM in a different type; the hook has to survive the wrapping.
		{"with memory", []ConfigOption{SetProvider("vllm"), SetModel("qwen"), SetMemory(1000)}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			client, err := NewLLM(tc.opts...)
			if err != nil {
				t.Fatalf("NewLLM: %v", err)
			}

			var seen []UsageEvent
			// The capability is deliberately off the LLM interface so downstream implementations
			// keep compiling; attaching post-construction goes through the optional interface.
			if !AttachUsageObserver(client, func(_ context.Context, e UsageEvent) { seen = append(seen, e) }) {
				t.Fatal("client does not accept a usage observer")
			}

			// The aliases must be usable without importing the internal llm package.
			var (
				_ UsageObserver = func(context.Context, UsageEvent) {}
				_ UsageOutcome  = UsageOutcomeSchemaFail
				_ TokenUsage    = types.TokenUsage{}
			)
			if len(seen) != 0 {
				t.Errorf("observer fired before any generation: %+v", seen)
			}
		})
	}
}
