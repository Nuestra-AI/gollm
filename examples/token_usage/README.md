# Token Usage Tracking Example

This example demonstrates how to use the new `GenerateWithUsage` and `GenerateWithSchemaAndUsage` methods to track token consumption and response details when making LLM API calls.

## Features

The example showcases:

1. **Basic Token Usage Tracking** - Track tokens used in standard text generation
2. **Schema-Based Usage Tracking** - Track tokens when using structured output with JSON schemas
3. **Cost Estimation** - Calculate estimated costs based on token usage
4. **Response Details** - Access message IDs and other response metadata

## What You'll Learn

- How to use `GenerateWithUsage()` to get response details including token usage
- How to use `GenerateWithSchemaAndUsage()` for structured output with usage tracking
- How to access different token metrics (prompt, completion, total)
- How to access response metadata like message ID and model used
- How to calculate API costs based on token usage
- Provider-specific features like cache usage (Anthropic)

## Response Details Information

The `types.ResponseDetails` struct provides comprehensive information:

```go
type ResponseDetails struct {
    ID         string      // Message/response ID from the provider
    TokenUsage TokenUsage  // Token consumption details
    Model      string      // Model used for the response
    Metadata   map[string]interface{} // Additional provider-specific metadata
}

type TokenUsage struct {
    PromptTokens             int  // Input tokens used
    CompletionTokens         int  // Output tokens generated
    TotalTokens              int  // Total tokens consumed
    CacheCreationInputTokens int  // Tokens written to cache (Anthropic)
    CacheReadInputTokens     int  // Tokens read from cache (Anthropic)
}
```

## Provider Support

Token usage tracking is supported across all providers:

- **OpenAI** - Full support with prompt/completion/total tokens
- **Anthropic** - Full support including cache metrics
- **Groq** - Full support (OpenAI-compatible format)
- **Mistral** - Full support (OpenAI-compatible format)  
- **OpenRouter** - Full support (OpenAI-compatible format)
- **Cohere** - Full support with normalized metrics
- **DeepSeek** - Full support (OpenAI-compatible format)
- **Google (Gemini)** - Full support (OpenAI-compatible format)
- **Ollama** - Returns `nil` for usage (local models don't track usage)
- **Generic Providers** - Support depends on underlying API format

## Running the Example

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Run the example
go run main.go
```

## Expected Output

```
=== Example 1: Basic Token Usage Tracking ===
Response: Quantum computing uses quantum bits (qubits) to perform calculations...

Token Usage:
  Prompt tokens:     12
  Completion tokens: 28
  Total tokens:      40

=== Example 2: Token Usage with Schema ===
Structured Response: {"language":"Go","description":"A statically typed...","year":2009}

Token Usage:
  Prompt tokens:     45
  Completion tokens: 35
  Total tokens:      80

Estimated Cost:
  Input cost:  $0.000007
  Output cost: $0.000021
  Total cost:  $0.000028
```

## Use Cases

Token usage tracking is valuable for:

1. **Cost Monitoring** - Track and optimize API costs
2. **Budget Management** - Stay within token budgets
3. **Performance Optimization** - Identify expensive operations
4. **Cache Optimization** - Monitor cache effectiveness (Anthropic)
5. **Usage Analytics** - Understand token consumption patterns
6. **Billing** - Implement usage-based billing in applications

## Notes

- The `GenerateWithUsage` and `GenerateWithSchemaAndUsage` methods are backward compatible
- If a provider doesn't support usage tracking, the usage parameter will be `nil`
- Cache metrics are only available for Anthropic's Claude models
- Cost calculations are estimates - check your provider's actual pricing
- Token counts may vary slightly between providers for the same prompt
