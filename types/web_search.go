// Package types contains shared type definitions used across the gollm library.
package types

// WebSearchCall represents a web search tool invocation in the response.
// This is specific to OpenAI's web_search tool feature.
type WebSearchCall struct {
	ID     string           `json:"id"`               // Unique identifier for the web search call
	Type   string           `json:"type"`             // Always "web_search_call"
	Status string           `json:"status"`           // Status of the search call (e.g., "completed")
	Action *WebSearchAction `json:"action,omitempty"` // Details about the action taken
}

// WebSearchAction describes the specific action taken during a web search.
// Actions can be "search", "open_page", or "find_in_page".
type WebSearchAction struct {
	Type    string   `json:"type"`              // Type of action: "search", "open_page", or "find_in_page"
	Query   string   `json:"query,omitempty"`   // Search query (for "search" actions)
	Domains []string `json:"domains,omitempty"` // Domains searched (for "search" actions)
	Sources []Source `json:"sources,omitempty"` // All URLs retrieved during search
}

// Source represents a URL source that was consulted during web search.
// This is returned in the sources field and includes all URLs, not just cited ones.
type Source struct {
	URL   string `json:"url"`             // URL of the source
	Title string `json:"title,omitempty"` // Title of the page/source
	Type  string `json:"type,omitempty"`  // Type of source (e.g., "oai-sports", "oai-weather", "oai-finance")
}

// URLCitation represents a citation annotation in the response content.
// Citations link specific parts of the response text to their source URLs.
type URLCitation struct {
	Type       string `json:"type"`        // Always "url_citation"
	StartIndex int    `json:"start_index"` // Starting character index in the text
	EndIndex   int    `json:"end_index"`   // Ending character index in the text
	URL        string `json:"url"`         // URL of the cited source
	Title      string `json:"title"`       // Title of the cited page
}

// Annotation is a generic annotation that can be a URL citation or other types.
// This allows for extensibility as OpenAI adds more annotation types.
type Annotation struct {
	Type       string `json:"type"`                  // Type of annotation (e.g., "url_citation")
	StartIndex int    `json:"start_index,omitempty"` // Starting character index in the text
	EndIndex   int    `json:"end_index,omitempty"`   // Ending character index in the text
	URL        string `json:"url,omitempty"`         // URL (for url_citation type)
	Title      string `json:"title,omitempty"`       // Title (for url_citation type)
}

// WebSearchResponse extends ResponseDetails with web search specific information.
// This includes web search calls and annotations extracted from the response.
type WebSearchResponse struct {
	ResponseDetails                 // Embedded standard response details
	WebSearchCalls  []WebSearchCall `json:"web_search_calls,omitempty"` // Web search calls made
	Annotations     []Annotation    `json:"annotations,omitempty"`      // Citations and other annotations
	Citations       []URLCitation   `json:"citations,omitempty"`        // Extracted URL citations for convenience
}

// Anthropic-specific web search types

// AnthropicServerToolUse represents a server-side tool use in Anthropic responses.
// This is used for web_search and other server-executed tools.
type AnthropicServerToolUse struct {
	Type  string                 `json:"type"`  // Always "server_tool_use"
	ID    string                 `json:"id"`    // Unique identifier
	Name  string                 `json:"name"`  // Tool name (e.g., "web_search")
	Input map[string]interface{} `json:"input"` // Tool input parameters (e.g., query)
}

// AnthropicWebSearchResult represents a single search result from Anthropic's web search.
type AnthropicWebSearchResult struct {
	Type             string `json:"type"`                        // Always "web_search_result"
	URL              string `json:"url"`                         // URL of the source page
	Title            string `json:"title"`                       // Title of the source page
	PageAge          string `json:"page_age,omitempty"`          // When the site was last updated
	EncryptedContent string `json:"encrypted_content,omitempty"` // Encrypted content for multi-turn conversations
}

// AnthropicWebSearchToolResult contains the results of a web search tool execution.
type AnthropicWebSearchToolResult struct {
	Type      string                     `json:"type"`            // Always "web_search_tool_result"
	ToolUseID string                     `json:"tool_use_id"`     // ID of the tool use this result corresponds to
	Content   []AnthropicWebSearchResult `json:"content"`         // Array of search results
	Error     *AnthropicWebSearchError   `json:"error,omitempty"` // Error if search failed
}

// AnthropicWebSearchError represents an error in web search execution.
type AnthropicWebSearchError struct {
	Type      string `json:"type"`       // Always "web_search_tool_result_error"
	ErrorCode string `json:"error_code"` // Error code (e.g., "max_uses_exceeded", "too_many_requests")
}

// AnthropicWebSearchResultLocation represents a citation in Anthropic's response.
// This appears in the citations array of text blocks.
type AnthropicWebSearchResultLocation struct {
	Type           string `json:"type"`            // Always "web_search_result_location"
	URL            string `json:"url"`             // URL of the cited source
	Title          string `json:"title"`           // Title of the cited source
	EncryptedIndex string `json:"encrypted_index"` // Reference for multi-turn conversations
	CitedText      string `json:"cited_text"`      // Up to 150 characters of cited content
}
