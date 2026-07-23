package types

import "testing"

// ComputedTotal is the single definition of TotalTokens that every path relies on, so the two
// accounting styles are pinned here rather than in each caller.
func TestComputedTotalCoversEveryBilledToken(t *testing.T) {
	cases := []struct {
		name  string
		usage TokenUsage
		want  int
	}{
		{
			// OpenAI reports a total that already counts the whole input, cached part included.
			// Recomputing it would be wrong; adding CachedPromptTokens on top double-counts.
			"openai total is trusted as reported",
			TokenUsage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150, CachedPromptTokens: 80, ReasoningTokens: 40},
			150,
		},
		{
			// Anthropic sends no total, and its cache counts sit alongside PromptTokens rather
			// than inside it. Omitting them under-reports this call by two thirds.
			"anthropic sums uncached input, both cache counts and output",
			TokenUsage{PromptTokens: 10, CompletionTokens: 5, CacheCreationInputTokens: 3, CacheReadInputTokens: 7},
			25,
		},
		{
			"no cache counts is plain input plus output",
			TokenUsage{PromptTokens: 18, CompletionTokens: 7},
			25,
		},
		{"nothing reported totals nothing", TokenUsage{}, 0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.usage.ComputedTotal(); got != tc.want {
				t.Errorf("ComputedTotal() = %d, want %d", got, tc.want)
			}
		})
	}
}

// Add must stay inclusive across provider styles: an OpenAI-style total plus a
// components-only Anthropic-style one (TotalTokens 0) must not drop the latter.
func TestAddIsInclusiveAcrossMixedProviderStyles(t *testing.T) {
	openai := TokenUsage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150}
	anthropic := TokenUsage{PromptTokens: 10, CompletionTokens: 50, CacheCreationInputTokens: 30, CacheReadInputTokens: 7}
	if anthropic.ComputedTotal() != 97 {
		t.Fatalf("precondition: anthropic ComputedTotal = %d, want 97", anthropic.ComputedTotal())
	}

	for _, c := range []struct {
		name string
		sum  TokenUsage
	}{
		{"openai+anthropic", openai.Add(anthropic)},
		{"anthropic+openai", anthropic.Add(openai)},
	} {
		if c.sum.TotalTokens != 247 {
			t.Errorf("%s: TotalTokens = %d, want 247 (150+97)", c.name, c.sum.TotalTokens)
		}
		if got := c.sum.ComputedTotal(); got != 247 {
			t.Errorf("%s: ComputedTotal() = %d, want 247", c.name, got)
		}
	}

	// Homogeneous sums are unchanged by the fix.
	if got := openai.Add(openai).ComputedTotal(); got != 300 {
		t.Errorf("openai+openai ComputedTotal = %d, want 300", got)
	}
	if got := anthropic.Add(anthropic).ComputedTotal(); got != 194 {
		t.Errorf("anthropic+anthropic ComputedTotal = %d, want 194", got)
	}
}
