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
