package llm

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/teilomillet/gollm/types"
)

// ---------------------------------------------------------------------------
// hasStructuredMessages
// ---------------------------------------------------------------------------

func TestHasStructuredMessagesEmpty(t *testing.T) {
	p := &Prompt{}
	assert.False(t, p.hasStructuredMessages())
}

func TestHasStructuredMessagesSingleUser(t *testing.T) {
	// A single user message with no tool calls should return false
	p := &Prompt{
		Messages: []PromptMessage{
			{Role: "user", Content: "hello"},
		},
	}
	assert.False(t, p.hasStructuredMessages())
}

func TestHasStructuredMessagesMultiTurn(t *testing.T) {
	p := &Prompt{
		Messages: []PromptMessage{
			{Role: "user", Content: "hello"},
			{Role: "assistant", Content: "hi"},
		},
	}
	assert.True(t, p.hasStructuredMessages())
}

func TestHasStructuredMessagesSingleAssistant(t *testing.T) {
	// Single message with non-user role should return true
	p := &Prompt{
		Messages: []PromptMessage{
			{Role: "assistant", Content: "I'll help you"},
		},
	}
	assert.True(t, p.hasStructuredMessages())
}

func TestHasStructuredMessagesSingleWithToolCallID(t *testing.T) {
	p := &Prompt{
		Messages: []PromptMessage{
			{Role: "user", Content: "result", ToolCallID: "call_123"},
		},
	}
	assert.True(t, p.hasStructuredMessages())
}

func TestHasStructuredMessagesSingleWithToolCalls(t *testing.T) {
	p := &Prompt{
		Messages: []PromptMessage{
			{
				Role: "assistant",
				ToolCalls: []types.ToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: struct {
							Name      string          `json:"name"`
							Arguments json.RawMessage `json:"arguments"`
						}{
							Name:      "get_weather",
							Arguments: json.RawMessage(`{}`),
						},
					},
				},
			},
		},
	}
	assert.True(t, p.hasStructuredMessages())
}

// ---------------------------------------------------------------------------
// promptMessagesToMemoryMessages
// ---------------------------------------------------------------------------

func TestPromptMessagesToMemoryMessages(t *testing.T) {
	promptMsgs := []PromptMessage{
		{Role: "system", Content: "You are helpful", CacheType: CacheTypeEphemeral},
		{Role: "user", Content: "What is Go?"},
		{
			Role: "assistant",
			ToolCalls: []types.ToolCall{
				{
					ID:   "call_abc",
					Type: "function",
					Function: struct {
						Name      string          `json:"name"`
						Arguments json.RawMessage `json:"arguments"`
					}{
						Name:      "search",
						Arguments: json.RawMessage(`{"q":"Go programming"}`),
					},
				},
			},
		},
		{Role: "tool", ToolCallID: "call_abc", Content: "Go is a programming language"},
	}

	result := promptMessagesToMemoryMessages(promptMsgs)

	require.Len(t, result, 4)

	// System message with cache control
	assert.Equal(t, "system", result[0].Role)
	assert.Equal(t, "You are helpful", result[0].Content)
	assert.Equal(t, "ephemeral", result[0].CacheControl)

	// User message
	assert.Equal(t, "user", result[1].Role)
	assert.Equal(t, "What is Go?", result[1].Content)

	// Assistant with tool calls
	assert.Equal(t, "assistant", result[2].Role)
	require.Len(t, result[2].ToolCalls, 1)
	assert.Equal(t, "call_abc", result[2].ToolCalls[0].ID)
	assert.Equal(t, "search", result[2].ToolCalls[0].Function.Name)

	// Tool response
	assert.Equal(t, "tool", result[3].Role)
	assert.Equal(t, "call_abc", result[3].ToolCallID)
	assert.Equal(t, "Go is a programming language", result[3].Content)
}

func TestPromptMessagesToMemoryMessagesEmpty(t *testing.T) {
	result := promptMessagesToMemoryMessages(nil)
	assert.Empty(t, result)
}

func TestPromptMessagesToMemoryMessagesWithMultiContent(t *testing.T) {
	promptMsgs := []PromptMessage{
		{
			Role: "user",
			MultiContent: []types.ContentPart{
				{Type: "text", Text: "Describe this image"},
				{Type: "image_url", ImageURL: &types.ImageURL{URL: "https://example.com/img.png"}},
			},
		},
	}

	result := promptMessagesToMemoryMessages(promptMsgs)

	require.Len(t, result, 1)
	assert.Equal(t, "user", result[0].Role)
	require.Len(t, result[0].MultiContent, 2)
	assert.Equal(t, types.ContentPartType("text"), result[0].MultiContent[0].Type)
	assert.Equal(t, types.ContentPartType("image_url"), result[0].MultiContent[1].Type)
}

// ---------------------------------------------------------------------------
// NewPrompt
// ---------------------------------------------------------------------------

func TestNewPromptDefaultMessage(t *testing.T) {
	p := NewPrompt("hello")
	require.Len(t, p.Messages, 1)
	assert.Equal(t, "user", p.Messages[0].Role)
	assert.Equal(t, "hello", p.Messages[0].Content)
	assert.Equal(t, "hello", p.Input)
}

func TestNewPromptWithSystemPrompt(t *testing.T) {
	p := NewPrompt("hello", WithSystemPrompt("Be brief", CacheTypeEphemeral))
	assert.Equal(t, "Be brief", p.SystemPrompt)
	assert.Equal(t, CacheTypeEphemeral, p.SystemCacheType)
}

func TestNewPromptWithMessages(t *testing.T) {
	msgs := []PromptMessage{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	}
	p := NewPrompt("hi", WithMessages(msgs))
	assert.Len(t, p.Messages, 2)
}

func TestNewPromptWithDirectives(t *testing.T) {
	p := NewPrompt("hello", WithDirectives("be concise", "use JSON"))
	assert.Equal(t, []string{"be concise", "use JSON"}, p.Directives)
}

func TestNewPromptWithMaxLength(t *testing.T) {
	p := NewPrompt("hello", WithMaxLength(100))
	assert.Equal(t, 100, p.MaxLength)
}

func TestNewPromptWithContext(t *testing.T) {
	p := NewPrompt("hello", WithContext("background info"))
	assert.Equal(t, "background info", p.Context)
}

func TestNewPromptWithOutput(t *testing.T) {
	p := NewPrompt("hello", WithOutput("JSON"))
	assert.Equal(t, "JSON", p.Output)
}

// ---------------------------------------------------------------------------
// Prompt.String()
// ---------------------------------------------------------------------------

func TestPromptStringIncludesAllParts(t *testing.T) {
	p := NewPrompt("What is Go?",
		WithSystemPrompt("You are an expert", CacheTypeEphemeral),
		WithContext("Programming languages"),
		WithDirectives("Be brief", "Use examples"),
		WithOutput("Markdown"),
		WithMaxLength(50),
	)

	s := p.String()
	assert.Contains(t, s, "System: You are an expert")
	assert.Contains(t, s, "(Cache: ephemeral)")
	assert.Contains(t, s, "Context: Programming languages")
	assert.Contains(t, s, "- Be brief")
	assert.Contains(t, s, "- Use examples")
	assert.Contains(t, s, "What is Go?")
	assert.Contains(t, s, "Expected Output Format:")
	assert.Contains(t, s, "Markdown")
	assert.Contains(t, s, "approximately 50 words")
}

// ---------------------------------------------------------------------------
// HasImages
// ---------------------------------------------------------------------------

func TestHasImagesTrue(t *testing.T) {
	p := NewPrompt("describe", WithImageURL("https://example.com/img.png", "auto"))
	assert.True(t, p.HasImages())
}

func TestHasImagesFalse(t *testing.T) {
	p := NewPrompt("hello")
	assert.False(t, p.HasImages())
}

// ---------------------------------------------------------------------------
// CacheOption
// ---------------------------------------------------------------------------

func TestCacheOption(t *testing.T) {
	p := NewPrompt("hello", CacheOption(CacheTypeEphemeral))
	require.Len(t, p.Messages, 1)
	assert.Equal(t, CacheTypeEphemeral, p.Messages[0].CacheType)
}
