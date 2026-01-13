package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/utils"
)

func mainAnthropic() {
	fmt.Println("=== Anthropic Web Search Examples ===")
	fmt.Println()

	// Example 1: Basic web search with Anthropic
	fmt.Println("Example 1: Basic Web Search (Anthropic)")
	anthropicBasicWebSearch()

	fmt.Println("\n=== Example 2: Web Search with Max Uses (Anthropic) ===")
	anthropicMaxUsesSearch()

	fmt.Println("\n=== Example 3: Web Search with Allowed Domains (Anthropic) ===")
	anthropicAllowedDomainsSearch()

	fmt.Println("\n=== Example 4: Web Search with Blocked Domains (Anthropic) ===")
	anthropicBlockedDomainsSearch()

	fmt.Println("\n=== Example 5: Web Search with User Location (Anthropic) ===")
	anthropicLocationBasedSearch()
}

// anthropicBasicWebSearch demonstrates basic web search with Anthropic
func anthropicBasicWebSearch() {
	llm, err := gollm.NewLLM(
		gollm.SetProvider("anthropic"),
		gollm.SetModel("claude-sonnet-4-5"),
		gollm.SetAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
		gollm.SetMaxTokens(1024),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	// Create a web_search tool for Anthropic
	webSearchTool := utils.Tool{
		Type: "web_search",
	}

	prompt := gollm.NewPrompt("What is the weather in NYC today?",
		gollm.WithTools([]utils.Tool{webSearchTool}),
		gollm.WithToolChoice("auto"),
	)

	ctx := context.Background()
	response, err := llm.Generate(ctx, prompt)
	if err != nil {
		log.Fatalf("Failed to generate response: %v", err)
	}

	fmt.Printf("Response: %s\n", response)
}

// anthropicMaxUsesSearch demonstrates limiting the number of web searches
func anthropicMaxUsesSearch() {
	llm, err := gollm.NewLLM(
		gollm.SetProvider("anthropic"),
		gollm.SetModel("claude-sonnet-4-5"),
		gollm.SetAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
		gollm.SetMaxTokens(1024),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	// Create a web_search tool with max_uses limit
	webSearchTool := utils.Tool{
		Type:    "web_search",
		MaxUses: 3, // Limit to 3 searches per request
	}

	prompt := gollm.NewPrompt("Compare the latest news from tech companies: Apple, Google, and Microsoft",
		gollm.WithTools([]utils.Tool{webSearchTool}),
		gollm.WithToolChoice("auto"),
	)

	ctx := context.Background()
	response, err := llm.Generate(ctx, prompt)
	if err != nil {
		log.Fatalf("Failed to generate response: %v", err)
	}

	fmt.Printf("Response: %s\n", response)
}

// anthropicAllowedDomainsSearch demonstrates web search with domain filtering
func anthropicAllowedDomainsSearch() {
	llm, err := gollm.NewLLM(
		gollm.SetProvider("anthropic"),
		gollm.SetModel("claude-sonnet-4-5"),
		gollm.SetAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
		gollm.SetMaxTokens(1024),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	// Create a web_search tool with allowed domains
	webSearchTool := utils.Tool{
		Type: "web_search",
		AllowedDomains: []string{
			"bbc.com",
			"reuters.com",
			"apnews.com",
		},
		MaxUses: 5,
	}

	prompt := gollm.NewPrompt("What are the top news stories today?",
		gollm.WithTools([]utils.Tool{webSearchTool}),
		gollm.WithToolChoice("auto"),
	)

	ctx := context.Background()
	response, err := llm.Generate(ctx, prompt)
	if err != nil {
		log.Fatalf("Failed to generate response: %v", err)
	}

	fmt.Printf("Response: %s\n", response)
}

// anthropicBlockedDomainsSearch demonstrates web search with blocked domains
func anthropicBlockedDomainsSearch() {
	llm, err := gollm.NewLLM(
		gollm.SetProvider("anthropic"),
		gollm.SetModel("claude-sonnet-4-5"),
		gollm.SetAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
		gollm.SetMaxTokens(1024),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	// Create a web_search tool with blocked domains
	webSearchTool := utils.Tool{
		Type: "web_search",
		BlockedDomains: []string{
			"example-spam.com",
			"untrusted-source.net",
		},
		MaxUses: 5,
	}

	prompt := gollm.NewPrompt("What are the latest AI developments?",
		gollm.WithTools([]utils.Tool{webSearchTool}),
		gollm.WithToolChoice("auto"),
	)

	ctx := context.Background()
	response, err := llm.Generate(ctx, prompt)
	if err != nil {
		log.Fatalf("Failed to generate response: %v", err)
	}

	fmt.Printf("Response: %s\n", response)
}

// anthropicLocationBasedSearch demonstrates web search with user location
func anthropicLocationBasedSearch() {
	llm, err := gollm.NewLLM(
		gollm.SetProvider("anthropic"),
		gollm.SetModel("claude-sonnet-4-5"),
		gollm.SetAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
		gollm.SetMaxTokens(1024),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	// Create a web_search tool with user location
	webSearchTool := utils.Tool{
		Type: "web_search",
		UserLocation: map[string]interface{}{
			"type":     "approximate",
			"city":     "San Francisco",
			"region":   "California",
			"country":  "US",
			"timezone": "America/Los_Angeles",
		},
		MaxUses: 5,
	}

	prompt := gollm.NewPrompt("What restaurants near me have the best reviews?",
		gollm.WithTools([]utils.Tool{webSearchTool}),
		gollm.WithToolChoice("auto"),
	)

	ctx := context.Background()
	response, err := llm.Generate(ctx, prompt)
	if err != nil {
		log.Fatalf("Failed to generate response: %v", err)
	}

	fmt.Printf("Response: %s\n", response)
}
