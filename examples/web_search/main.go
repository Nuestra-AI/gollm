package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/teilomillet/gollm"
	"github.com/teilomillet/gollm/utils"
)

func main() {
	// Example 1: Basic web search
	fmt.Println("=== Example 1: Basic Web Search ===")
	basicWebSearch()

	fmt.Println("\n=== Example 2: Web Search with Domain Filtering ===")
	domainFilteredSearch()

	fmt.Println("\n=== Example 3: Web Search with User Location ===")
	locationBasedSearch()

	fmt.Println("\n=== Example 4: Offline/Cache-Only Web Search ===")
	cacheOnlySearch()
}

// basicWebSearch demonstrates basic web search usage
func basicWebSearch() {
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o"),
		gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		gollm.SetMaxTokens(500),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	// Create a web_search tool
	webSearchTool := utils.Tool{
		Type: "web_search",
	}

	prompt := gollm.NewPrompt("What was a positive news story from today?",
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

// domainFilteredSearch demonstrates web search with domain filtering
func domainFilteredSearch() {
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o"),
		gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		gollm.SetMaxTokens(500),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	// Create a web_search tool with allowed domains filter
	webSearchTool := utils.Tool{
		Type: "web_search",
		Filters: map[string]interface{}{
			"allowed_domains": []string{
				"pubmed.ncbi.nlm.nih.gov",
				"clinicaltrials.gov",
				"www.who.int",
				"www.cdc.gov",
				"www.fda.gov",
			},
		},
	}

	prompt := gollm.NewPrompt("What are the latest guidelines for diabetes treatment?",
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

// locationBasedSearch demonstrates web search with user location
func locationBasedSearch() {
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o"),
		gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		gollm.SetMaxTokens(500),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	// Create a web_search tool with user location
	webSearchTool := utils.Tool{
		Type: "web_search",
		UserLocation: map[string]interface{}{
			"type":     "approximate",
			"country":  "GB",
			"city":     "London",
			"region":   "London",
			"timezone": "Europe/London",
		},
	}

	prompt := gollm.NewPrompt("What are the best restaurants near me?",
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

// cacheOnlySearch demonstrates offline/cache-only web search
func cacheOnlySearch() {
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o"),
		gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		gollm.SetMaxTokens(500),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	// Create a web_search tool with external_web_access disabled (cache-only mode)
	webSearchTool := utils.Tool{
		Type:              "web_search",
		ExternalWebAccess: false,
	}

	prompt := gollm.NewPrompt("What is the Eiffel Tower's height?",
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
