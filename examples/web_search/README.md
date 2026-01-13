# Web Search Example

This example demonstrates how to use web_search tools with both **OpenAI** and **Anthropic** providers in the gollm library. The web_search tool allows models to access up-to-date information from the internet and provide answers with sourced citations.

## Supported Providers

### OpenAI
- Models: `gpt-4o`, `gpt-5`, o-series reasoning models
- Specialized models for Chat Completions API: `gpt-5-search-api`, `gpt-4o-search-preview`, `gpt-4o-mini-search-preview`
- **Important**: When you specify a `web_search` tool with OpenAI, gollm automatically:
  1. Switches to an appropriate search-enabled model (e.g., `gpt-4o` → `gpt-4o-search-preview`)
  2. Filters out the `web_search` tool from the API request (search models have built-in search capabilities)
- The Chat Completions API does NOT accept `"type": "web_search"` in the tools array - only "function" and "custom" types are supported
- Search models perform web searches automatically based on the conversation context

### Anthropic
- Models: `claude-sonnet-4-5`, `claude-sonnet-4`, `claude-haiku-4-5`, `claude-opus-4-5`, `claude-opus-4-1`, `claude-opus-4`

## Features Demonstrated

### OpenAI Examples (main.go)

#### 1. Basic Web Search
Simple web search for current information:
```go
webSearchTool := utils.Tool{
    Type: "web_search",
}
```

#### 2. Domain Filtering
Limit search results to specific trusted domains (up to 100 URLs):
```go
webSearchTool := utils.Tool{
    Type: "web_search",
    Filters: map[string]interface{}{
        "allowed_domains": []string{
            "pubmed.ncbi.nlm.nih.gov",
            "clinicaltrials.gov",
        },
    },
}
```

#### 3. User Location
Refine search results based on geographic location:
```go
webSearchTool := utils.Tool{
    Type: "web_search",
    UserLocation: map[string]interface{}{
        "type":     "approximate",
        "country":  "GB",
        "city":     "London",
        "timezone": "Europe/London",
    },
}
```

#### 4. Cache-Only/Offline Mode (OpenAI only)
Use cached/indexed results without fetching live content:
```go
webSearchTool := utils.Tool{
    Type:              "web_search",
    ExternalWebAccess: false,
}
```

### Anthropic Examples (anthropic_example.go)

#### 1. Basic Web Search
```go
webSearchTool := utils.Tool{
    Type: "web_search",
}
```

#### 2. Max Uses Limit (Anthropic-specific)
Limit the number of searches per request:
```go
webSearchTool := utils.Tool{
    Type:    "web_search",
    MaxUses: 3, // Maximum 3 searches
}
```

#### 3. Allowed Domains
```go
webSearchTool := utils.Tool{
    Type: "web_search",
    AllowedDomains: []string{
        "bbc.com",
        "reuters.com",
    },
    MaxUses: 5,
}
```

#### 4. Blocked Domains (Anthropic-specific)
Exclude specific domains from results:
```go
webSearchTool := utils.Tool{
    Type: "web_search",
    BlockedDomains: []string{
        "untrusted-source.net",
    },
    MaxUses: 5,
}
```

**Note**: `allowed_domains` and `blocked_domains` are mutually exclusive.

#### 5. User Location
Same as OpenAI location-based search.

## Provider Differences

| Feature | OpenAI | Anthropic |
|---------|--------|-----------|
| Tool Type | `"web_search"` | `"web_search"` |
| Internal Type | `"web_search"` | `"web_search_20250305"` (versioned) |
| Domain Filtering | `allowed_domains` only | `allowed_domains` OR `blocked_domains` |
| Max Uses | Not supported | `max_uses` parameter |
| External Web Access | `external_web_access` (true/false) | Always live |
| User Location | Supported | Supported |
| Response Format | `web_search_call` + annotations | `server_tool_use` + `web_search_tool_result` + citations |
| Citation Format | `url_citation` in annotations | `web_search_result_location` in text blocks |

## Domain Filtering Rules

### OpenAI
- Domains without HTTP/HTTPS prefix: `example.com` not `https://example.com`
- Subdomains automatically included
- Up to 100 domains allowed

### Anthropic
- Domains without HTTP/HTTPS scheme
- Subdomains automatically included (`example.com` covers `docs.example.com`)
- Specific subdomains restrict to that subdomain only
- Subpaths supported: `example.com/blog` matches `example.com/blog/post-1`
- Wildcard support: One `*` allowed in path only
  - ✅ Valid: `example.com/*`, `example.com/*/articles`
  - ❌ Invalid: `*.example.com`, `ex*.com`

## Response Format

### OpenAI Response
```go
// Access from response metadata
details.Metadata["web_search_calls"]  // []WebSearchCall
details.Metadata["annotations"]       // []Annotation
details.Metadata["citations"]         // []URLCitation
```

### Anthropic Response
```go
// Access from response metadata
details.Metadata["server_tool_uses"]     // []AnthropicServerToolUse
details.Metadata["web_search_results"]   // []AnthropicWebSearchResult
details.Metadata["citations"]            // []AnthropicWebSearchResultLocation
details.Metadata["web_search_requests"]  // int (usage count)
```

## Web Search Types

### OpenAI
1. **Non-reasoning web search**: Fast, simple lookups
2. **Agentic search with reasoning models**: Model manages search process, can perform multiple searches
3. **Deep research**: In-depth investigations with hundreds of sources

### Anthropic
Claude decides when to search based on the prompt. The API may execute multiple searches throughout a single request, providing cited sources in the final response.

## Usage

### OpenAI
```bash
export OPENAI_API_KEY=your-api-key
go run main.go
```

### Anthropic
```bash
export ANTHROPIC_API_KEY=your-api-key
# Note: anthropic_example.go contains functions but doesn't have a main() entry point
# You can copy the functions to main.go or create a separate test file
```

## Important Notes

1. **Citations must be visible**: When displaying web results to end users, inline citations must be clearly visible and clickable.

2. **Pricing**:
   - **OpenAI**: Standard token costs plus tool usage (varies by model)
   - **Anthropic**: $10 per 1,000 searches plus standard token costs

3. **Rate limits**: Same tiered rate limits as the underlying model

4. **Organization settings**: Anthropic requires administrators to enable web search in Console

5. **Context window**:
   - **OpenAI**: Limited to 128,000 tokens
   - **Anthropic**: Normal context limits apply

## Abstraction Benefits

The gollm library provides a unified interface for web_search across providers:

```go
// Same tool definition works for both OpenAI and Anthropic
webSearchTool := utils.Tool{
    Type: "web_search",
    AllowedDomains: []string{"example.com"},
    UserLocation: map[string]interface{}{
        "type": "approximate",
        "country": "US",
    },
}
```

Provider-specific features (like `MaxUses` for Anthropic or `ExternalWebAccess` for OpenAI) are gracefully ignored by providers that don't support them.

## Reference

- [OpenAI Web Search Documentation](https://platform.openai.com/docs/guides/tools-web-search)
- [Anthropic Web Search Documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)
- [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses)
- [Anthropic Messages API](https://platform.claude.com/docs/en/api-reference/messages)
