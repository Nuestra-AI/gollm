// Package types contains shared type definitions used across the gollm library.
// It helps avoid import cycles while providing common data structures.
package types

// MemoryMessage represents a single message in the conversation history.
// It includes the role of the speaker, the content of the message,
// and the number of tokens in the message for efficient memory management.
type MemoryMessage struct {
	Role         string                 // Role of the message sender (e.g., "user", "assistant")
	Content      string                 // The actual message content
	Tokens       int                    // Number of tokens in the message
	CacheControl string                 // Caching strategy for this message ("ephemeral", "persistent", etc.)
	Metadata     map[string]interface{} // Additional provider-specific metadata
}

// TokenUsage contains token consumption information from an LLM API call.
// This structure is normalized across different providers to provide a consistent interface.
// Providers map their specific usage fields to these normalized field names.
type TokenUsage struct {
	PromptTokens             int `json:"prompt_tokens"`                         // Input tokens (normalized from provider-specific field)
	CompletionTokens         int `json:"completion_tokens"`                     // Output tokens (normalized from provider-specific field)
	TotalTokens              int `json:"total_tokens"`                          // Total tokens used in the request
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"` // Tokens written to cache (Anthropic-specific)
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`     // Tokens read from cache (Anthropic-specific)
}

// ResponseDetails contains comprehensive response metadata from an LLM API call.
// This includes the message ID, token usage information, and other provider-specific details.
type ResponseDetails struct {
	ID         string                 `json:"id,omitempty"`       // Message/response ID from the provider
	TokenUsage TokenUsage             `json:"token_usage"`        // Token consumption details
	Model      string                 `json:"model,omitempty"`    // Model used for the response
	Metadata   map[string]interface{} `json:"metadata,omitempty"` // Additional provider-specific metadata
}
