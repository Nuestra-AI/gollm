// Package types contains shared type definitions used across the gollm library.
// It helps avoid import cycles while providing common data structures.
package types

// ContentPartType represents the type of content in a multimodal message.
type ContentPartType string

const (
	// ContentTypeText represents text content.
	ContentTypeText ContentPartType = "text"
	// ContentTypeImageURL represents an image referenced by URL.
	ContentTypeImageURL ContentPartType = "image_url"
	// ContentTypeImage represents an image with embedded data (base64).
	ContentTypeImage ContentPartType = "image"
)

// ImageURL represents an image referenced by URL.
// Used by OpenAI and similar providers.
type ImageURL struct {
	URL    string `json:"url"`              // URL of the image or base64 data URI
	Detail string `json:"detail,omitempty"` // Detail level: "auto", "low", or "high"
}

// ImageSource represents an image with embedded data.
// Used by Anthropic and similar providers.
type ImageSource struct {
	Type      string `json:"type"`       // Source type: "base64"
	MediaType string `json:"media_type"` // MIME type: "image/jpeg", "image/png", "image/gif", "image/webp"
	Data      string `json:"data"`       // Base64-encoded image data
}

// ContentPart represents a single part of multimodal content.
// A message can contain multiple parts (e.g., text and images).
type ContentPart struct {
	Type     ContentPartType `json:"type"`                // Type of content: "text", "image_url", or "image"
	Text     string          `json:"text,omitempty"`      // Text content (when Type is "text")
	ImageURL *ImageURL       `json:"image_url,omitempty"` // Image URL (when Type is "image_url")
	Source   *ImageSource    `json:"source,omitempty"`    // Image source (when Type is "image", used by Anthropic)
}

// NewTextContent creates a text content part.
func NewTextContent(text string) ContentPart {
	return ContentPart{
		Type: ContentTypeText,
		Text: text,
	}
}

// NewImageURLContent creates an image content part from a URL.
// The detail parameter can be "auto", "low", or "high" (empty defaults to "auto").
func NewImageURLContent(url string, detail string) ContentPart {
	if detail == "" {
		detail = "auto"
	}
	return ContentPart{
		Type: ContentTypeImageURL,
		ImageURL: &ImageURL{
			URL:    url,
			Detail: detail,
		},
	}
}

// NewImageBase64Content creates an image content part from base64-encoded data.
// mediaType should be "image/jpeg", "image/png", "image/gif", or "image/webp".
func NewImageBase64Content(base64Data, mediaType string) ContentPart {
	return ContentPart{
		Type: ContentTypeImage,
		Source: &ImageSource{
			Type:      "base64",
			MediaType: mediaType,
			Data:      base64Data,
		},
	}
}

// MemoryMessage represents a single message in the conversation history.
// It includes the role of the speaker, the content of the message,
// and the number of tokens in the message for efficient memory management.
//
// For tool calling support:
// - Assistant messages may include ToolCalls (requests to use tools)
// - Tool messages contain ToolCallID (linking result to the original call)
//
// For multimodal support:
// - Use MultiContent for messages with images or mixed content
// - When MultiContent is set, Content is ignored by providers that support multimodal
type MemoryMessage struct {
	Role         string                 // Role of the message sender (e.g., "user", "assistant", "tool")
	Content      string                 // The actual message content (text-only)
	MultiContent []ContentPart          // Multimodal content (text + images); takes precedence over Content
	Tokens       int                    // Number of tokens in the message
	CacheControl string                 // Caching strategy for this message ("ephemeral", "persistent", etc.)
	Metadata     map[string]interface{} // Additional provider-specific metadata
	ToolCalls    []ToolCall             // Tool calls requested by the assistant (only for role="assistant")
	ToolCallID   string                 // ID of the tool call this message responds to (only for role="tool")
}

// HasMultiContent returns true if the message contains multimodal content.
func (m *MemoryMessage) HasMultiContent() bool {
	return len(m.MultiContent) > 0
}

// GetTextContent returns the text content of the message.
// If MultiContent is set, it concatenates all text parts.
// Otherwise, it returns the Content field.
func (m *MemoryMessage) GetTextContent() string {
	if !m.HasMultiContent() {
		return m.Content
	}
	var text string
	for _, part := range m.MultiContent {
		if part.Type == ContentTypeText {
			text += part.Text
		}
	}
	return text
}

// TokenUsage contains token consumption information from an LLM API call.
// This structure is normalized across different providers to provide a consistent interface.
// Providers map their specific usage fields to these normalized field names.
//
// Inclusion semantics differ between the two cache accounting styles, and cost
// math depends on getting them right:
//
//   - Anthropic style: PromptTokens counts only uncached input;
//     CacheCreationInputTokens and CacheReadInputTokens are reported *alongside*
//     it, so billable input is the sum of all three (at their own rates).
//   - OpenAI style: PromptTokens is the full input count and already *includes*
//     CachedPromptTokens, which is the discounted portion of it.
//
// ReasoningTokens is likewise a subset of CompletionTokens, not an addition to
// it — billed at the output rate but broken out so callers can see the split.
// It carries OpenAI's reasoning_tokens and Anthropic's thinking_tokens, which
// name the same thing.
//
// Several fields are finer breakdowns of a field above them, never additions:
//
//   - CacheCreation5mInputTokens and CacheCreation1hInputTokens partition
//     CacheCreationInputTokens by cache lifetime, which is what distinguishes a
//     1.25x write from a 2x one.
//   - CacheWritePromptTokens is the OpenAI-style write, inside PromptTokens.
//   - AcceptedPredictionTokens and RejectedPredictionTokens are inside
//     CompletionTokens; rejected predictions are billed in full despite never
//     appearing in the response.
//   - AudioPromptTokens and AudioCompletionTokens are inside PromptTokens and
//     CompletionTokens respectively, and are priced far above text.
//
// TotalTokens counts every billed token in the round-trip, so it is comparable
// across providers: for the OpenAI style that is the provider's own total (whose
// input side already covers the cached portion), and for the Anthropic style it
// is uncached input plus both cache fields plus output. The cache and reasoning
// fields are breakdowns of what TotalTokens already contains, never additions to
// it — summing them into it double-counts. See ComputedTotal, which is the single
// definition every path uses, so the same round-trip totals the same whether it
// was read by a provider parser, a stream accumulator, or raw-body recovery.
type TokenUsage struct {
	PromptTokens             int `json:"prompt_tokens"`                         // Input tokens (normalized from provider-specific field)
	CompletionTokens         int `json:"completion_tokens"`                     // Output tokens (normalized from provider-specific field)
	TotalTokens              int `json:"total_tokens"`                          // Every billed token in the request; see ComputedTotal
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"` // Tokens written to cache, excluded from PromptTokens (Anthropic style)
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`     // Tokens read from cache, excluded from PromptTokens (Anthropic style)
	CachedPromptTokens       int `json:"cached_prompt_tokens,omitempty"`        // Cache-discounted input tokens, included in PromptTokens (OpenAI style: prompt_tokens_details.cached_tokens)
	ReasoningTokens          int `json:"reasoning_tokens,omitempty"`            // Reasoning/thinking output tokens, included in CompletionTokens

	// Breakdowns of the fields above, each billed at its own multiplier.
	CacheCreation5mInputTokens int `json:"cache_creation_5m_input_tokens,omitempty"` // 5-minute cache writes, billed 1.25x input; subset of CacheCreationInputTokens
	CacheCreation1hInputTokens int `json:"cache_creation_1h_input_tokens,omitempty"` // 1-hour cache writes, billed 2x input; subset of CacheCreationInputTokens
	CacheWritePromptTokens     int `json:"cache_write_prompt_tokens,omitempty"`      // Cache writes billed 1.25x input, included in PromptTokens (OpenAI style: prompt_tokens_details.cache_write_tokens)
	AcceptedPredictionTokens   int `json:"accepted_prediction_tokens,omitempty"`     // Predicted-output tokens that appeared in the completion; subset of CompletionTokens
	RejectedPredictionTokens   int `json:"rejected_prediction_tokens,omitempty"`     // Predicted-output tokens that did not appear yet are billed as output; subset of CompletionTokens
	AudioPromptTokens          int `json:"audio_prompt_tokens,omitempty"`            // Audio input tokens, subset of PromptTokens, priced well above text
	AudioCompletionTokens      int `json:"audio_completion_tokens,omitempty"`        // Audio output tokens, subset of CompletionTokens, priced well above text
}

// Add returns the element-wise sum of two usage records. It is the correct way
// to accumulate the cost of a multi-round-trip operation (retries, tool loops,
// mixture-of-agents), where each round-trip is billed independently.
func (u TokenUsage) Add(other TokenUsage) TokenUsage {
	return TokenUsage{
		PromptTokens:             u.PromptTokens + other.PromptTokens,
		CompletionTokens:         u.CompletionTokens + other.CompletionTokens,
		TotalTokens:              u.TotalTokens + other.TotalTokens,
		CacheCreationInputTokens: u.CacheCreationInputTokens + other.CacheCreationInputTokens,
		CacheReadInputTokens:     u.CacheReadInputTokens + other.CacheReadInputTokens,
		CachedPromptTokens:       u.CachedPromptTokens + other.CachedPromptTokens,
		ReasoningTokens:          u.ReasoningTokens + other.ReasoningTokens,

		CacheCreation5mInputTokens: u.CacheCreation5mInputTokens + other.CacheCreation5mInputTokens,
		CacheCreation1hInputTokens: u.CacheCreation1hInputTokens + other.CacheCreation1hInputTokens,
		CacheWritePromptTokens:     u.CacheWritePromptTokens + other.CacheWritePromptTokens,
		AcceptedPredictionTokens:   u.AcceptedPredictionTokens + other.AcceptedPredictionTokens,
		RejectedPredictionTokens:   u.RejectedPredictionTokens + other.RejectedPredictionTokens,
		AudioPromptTokens:          u.AudioPromptTokens + other.AudioPromptTokens,
		AudioCompletionTokens:      u.AudioCompletionTokens + other.AudioCompletionTokens,
	}
}

// ComputedTotal is the one definition of TotalTokens, used by every path that
// produces a TokenUsage so that a round-trip totals the same however its usage
// was read.
//
// A provider-reported total is trusted as-is: the APIs that send one (the
// OpenAI-shaped ones) already count the whole input, cached portion included.
// Otherwise every billed component is summed — which for the Anthropic style
// means the cache fields too, since those sit alongside PromptTokens rather than
// inside it. Leaving them out would under-report a cache-heavy call by most of
// what it cost, and would make Anthropic's total mean something different from
// OpenAI's.
func (u TokenUsage) ComputedTotal() int {
	if u.TotalTokens > 0 {
		return u.TotalTokens
	}
	return u.PromptTokens + u.CompletionTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens
}

// IsZero reports whether no tokens were recorded at all, which distinguishes a
// round-trip whose usage the provider never reported from one that reported it.
func (u TokenUsage) IsZero() bool {
	return u == TokenUsage{}
}

// ResponseDetails contains comprehensive response metadata from an LLM API call.
// This includes the message ID, token usage information, and other provider-specific details.
type ResponseDetails struct {
	ID         string                 `json:"id,omitempty"`         // Message/response ID from the provider
	TokenUsage TokenUsage             `json:"token_usage"`          // Token consumption details
	Model      string                 `json:"model,omitempty"`      // Model used for the response
	Metadata   map[string]interface{} `json:"metadata,omitempty"`   // Additional provider-specific metadata
	ToolCalls  []ToolCall             `json:"tool_calls,omitempty"` // Structured tool calls from the response (preserves ID, type, and arguments)
	// ServiceTier is the tier the provider actually served the request on
	// ("standard", "priority", "batch", "flex", "scale", …). It is not a token
	// count but it multiplies every token's price — batch is roughly half rate,
	// priority a premium — so a cost record without it can be wrong by 2x on
	// counts that are perfectly accurate. Empty when the provider reports none.
	ServiceTier string `json:"service_tier,omitempty"`
}
