package gollm

import (
	"context"
	"reflect"
	"testing"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/llm"
)

// observerInstalled reports whether a constructed client actually carries a usage observer.
// Function values aren't comparable, so identity can't be asserted; presence on the client that was
// built — rather than on the config that built it — is the property that matters, and reflection is
// the only way to see it from outside the llm package.
func observerInstalled(t *testing.T, client interface{}) bool {
	t.Helper()

	v := reflect.ValueOf(client)
	for v.Kind() == reflect.Ptr || v.Kind() == reflect.Interface {
		if v.IsNil() {
			t.Fatal("nil client")
		}
		// Unwrap memory and other decorators that embed an LLM until the impl is reached.
		if v.Kind() == reflect.Ptr && v.Elem().Kind() == reflect.Struct {
			if f := v.Elem().FieldByName("usageObserver"); f.IsValid() {
				return !f.IsNil()
			}
			if inner := v.Elem().FieldByName("LLM"); inner.IsValid() {
				v = inner
				continue
			}
		}
		v = v.Elem()
	}
	t.Fatalf("no usageObserver field found on %T", client)
	return false
}

// An observer attached at configuration time must reach the client built from that config, without
// the caller having to hold the client and call a setter.
func TestWithUsageObserverInstalledAtConstruction(t *testing.T) {
	observer := func(context.Context, UsageEvent) {}

	cases := []struct {
		name string
		opts []ConfigOption
	}{
		{"plain", []ConfigOption{SetProvider("vllm"), SetModel("qwen"), WithUsageObserver(observer)}},
		{"with memory", []ConfigOption{SetProvider("vllm"), SetModel("qwen"), SetMemory(1000), WithUsageObserver(observer)}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			client, err := NewLLM(tc.opts...)
			if err != nil {
				t.Fatalf("NewLLM: %v", err)
			}
			if !observerInstalled(t, client) {
				t.Error("client was built without the configured usage observer")
			}
		})
	}

	// Without the option, nothing is installed — the hook stays opt-in.
	bare, err := NewLLM(SetProvider("vllm"), SetModel("qwen"))
	if err != nil {
		t.Fatalf("NewLLM: %v", err)
	}
	if observerInstalled(t, bare) {
		t.Error("an observer was installed without WithUsageObserver")
	}
}

// MOA is the highest fan-out spend in the library — one Generate is (models x iterations)+1 billed
// calls across clients the caller never holds. Passing the observer once must cover every one of
// them, layer models included, not just the aggregator.
func TestMOAPropagatesUsageObserverToEveryModel(t *testing.T) {
	observer := func(context.Context, UsageEvent) {}

	moa, err := NewMOA(
		MOAConfig{
			Iterations: 1,
			Models: []ConfigOption{
				func(c *config.Config) { SetProvider("vllm")(c); SetModel("model-a")(c) },
				func(c *config.Config) { SetProvider("vllm")(c); SetModel("model-b")(c) },
			},
		},
		SetProvider("vllm"), SetModel("aggregator"), WithUsageObserver(observer),
	)
	if err != nil {
		t.Fatalf("NewMOA: %v", err)
	}

	for i, layer := range moa.Layers {
		for j, model := range layer.Models {
			if !observerInstalled(t, model) {
				t.Errorf("layer %d model %d has no usage observer: its calls would go unrecorded", i, j)
			}
		}
	}
	if !observerInstalled(t, moa.Aggregator) {
		t.Error("aggregator has no usage observer")
	}
}

// A layer model that configures its own observer keeps it rather than being overwritten by the
// aggregator's, so usage can be routed per model when a suite needs that.
func TestMOAModelKeepsItsOwnObserver(t *testing.T) {
	perModel := func(context.Context, UsageEvent) {}
	aggregatorObserver := func(context.Context, UsageEvent) {}

	moa, err := NewMOA(
		MOAConfig{
			Iterations: 1,
			Models: []ConfigOption{
				func(c *config.Config) {
					SetProvider("vllm")(c)
					SetModel("model-a")(c)
					WithUsageObserver(perModel)(c)
				},
				// This one sets none and must inherit the aggregator's.
				func(c *config.Config) { SetProvider("vllm")(c); SetModel("model-b")(c) },
			},
		},
		SetProvider("vllm"), SetModel("aggregator"), WithUsageObserver(aggregatorObserver),
	)
	if err != nil {
		t.Fatalf("NewMOA: %v", err)
	}

	for i, layer := range moa.Layers {
		if !observerInstalled(t, layer.Models[0]) {
			t.Errorf("layer %d lost its observer", i)
		}
	}
}

// The option must be usable through the public surface without importing the internal llm package.
func TestUsageObserverTypesAreAliased(t *testing.T) {
	var (
		_ UsageObserver     = func(context.Context, UsageEvent) {}
		_ UsageOutcome      = UsageOutcomeParseFail
		_ llm.UsageObserver = UsageObserver(nil) // llm and gollm names denote the same type
	)
}
