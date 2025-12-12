package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/teilomillet/gollm"
)

func main() {
	// Check for API key
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("Error: OPENAI_API_KEY environment variable not set")
		fmt.Println("\nTo run this example, set your OpenAI API key:")
		fmt.Println("  PowerShell: $env:OPENAI_API_KEY=\"your-key-here\"")
		fmt.Println("  Bash:       export OPENAI_API_KEY=\"your-key-here\"")
		fmt.Println("\nOr use a different provider by changing SetProvider() in the code")
		os.Exit(1)
	}

	// Example 1: Basic usage tracking with Generate
	fmt.Println("=== Example 1: Basic Token Usage Tracking ===")
	basicUsageExample()

	fmt.Println()

	// Example 2: Usage tracking with schema validation
	fmt.Println("=== Example 2: Token Usage with Schema ===")
	schemaUsageExample()
}

func basicUsageExample() {
	// Create LLM instance
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetMaxTokens(150),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	// Create a prompt
	prompt := gollm.NewPrompt("Explain quantum computing in one sentence.")

	// Generate with usage tracking
	ctx := context.Background()
	response, details, err := llm.GenerateWithUsage(ctx, prompt)
	if err != nil {
		log.Fatalf("Failed to generate: %v", err)
	}

	// Display results
	fmt.Printf("Response: %s\n\n", response)

	if details != nil {
		// Display response metadata
		if details.ID != "" {
			fmt.Printf("Response ID: %s\n", details.ID)
		}
		if details.Model != "" {
			fmt.Printf("Model: %s\n", details.Model)
		}
		fmt.Printf("\nToken Usage:\n")
		fmt.Printf("  Prompt tokens:     %d\n", details.TokenUsage.PromptTokens)
		fmt.Printf("  Completion tokens: %d\n", details.TokenUsage.CompletionTokens)
		fmt.Printf("  Total tokens:      %d\n", details.TokenUsage.TotalTokens)

		// Display cache information if available (Anthropic-specific)
		if details.TokenUsage.CacheCreationInputTokens > 0 || details.TokenUsage.CacheReadInputTokens > 0 {
			fmt.Printf("\nCache Information:\n")
			fmt.Printf("  Cache creation tokens: %d\n", details.TokenUsage.CacheCreationInputTokens)
			fmt.Printf("  Cache read tokens:     %d\n", details.TokenUsage.CacheReadInputTokens)
		}
	} else {
		fmt.Println("Note: Response details not available for this provider")
	}
}

func schemaUsageExample() {
	// Create LLM instance
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetMaxTokens(300),
		gollm.SetTemperature(0.0), // Use deterministic output
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	// Define a schema for structured output
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"language": map[string]interface{}{
				"type":        "string",
				"description": "Programming language name",
			},
			"year": map[string]interface{}{
				"type":        "integer",
				"description": "Year of first release",
			},
			"paradigm": map[string]interface{}{
				"type":        "string",
				"description": "Primary programming paradigm",
			},
		},
		"required": []string{"language", "year", "paradigm"},
	}

	// Create a prompt
	prompt := gollm.NewPrompt("Provide information about the Go programming language in JSON format")

	// Generate with schema and usage tracking
	ctx := context.Background()
	response, details, err := llm.GenerateWithSchemaAndUsage(ctx, prompt, schema)
	if err != nil {
		log.Fatalf("Failed to generate: %v", err)
	}

	// Display results
	fmt.Printf("Structured Response:\n%s\n\n", response)

	if details != nil {
		// Display response metadata
		if details.ID != "" {
			fmt.Printf("Response ID: %s\n", details.ID)
		}
		if details.Model != "" {
			fmt.Printf("Model: %s\n", details.Model)
		}
		fmt.Printf("\nToken Usage:\n")
		fmt.Printf("  Prompt tokens:     %d\n", details.TokenUsage.PromptTokens)
		fmt.Printf("  Completion tokens: %d\n", details.TokenUsage.CompletionTokens)
		fmt.Printf("  Total tokens:      %d\n", details.TokenUsage.TotalTokens)

		// Calculate estimated cost (example rates for gpt-4o-mini)
		inputCostPer1K := 0.00015 // $0.15 per 1K input tokens
		outputCostPer1K := 0.0006 // $0.60 per 1K output tokens

		inputCost := float64(details.TokenUsage.PromptTokens) / 1000 * inputCostPer1K
		outputCost := float64(details.TokenUsage.CompletionTokens) / 1000 * outputCostPer1K
		totalCost := inputCost + outputCost

		fmt.Printf("\nEstimated Cost (gpt-4o-mini):\n")
		fmt.Printf("  Input cost:  $%.6f\n", inputCost)
		fmt.Printf("  Output cost: $%.6f\n", outputCost)
		fmt.Printf("  Total cost:  $%.6f\n", totalCost)
	} else {
		fmt.Println("Note: Response details not available for this provider")
	}
}
