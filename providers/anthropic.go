// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

// AnthropicProvider implements the Provider interface for Anthropic's Claude API.
// It supports Claude models and provides access to Anthropic's language model capabilities,
// including structured output and system prompts.
type AnthropicProvider struct {
	apiKey       string                 // API key for authentication
	model        string                 // Model identifier (e.g., "claude-3-opus", "claude-3-sonnet")
	extraHeaders map[string]string      // Additional HTTP headers
	options      map[string]interface{} // Model-specific options
	logger       utils.Logger           // Logger instance
}

// NewAnthropicProvider creates a new Anthropic provider instance.
// It initializes the provider with the given API key, model, and optional headers.
//
// Parameters:
//   - apiKey: Anthropic API key for authentication
//   - model: The model to use (e.g., "claude-3-opus", "claude-3-sonnet")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured Anthropic Provider instance
func NewAnthropicProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	provider := &AnthropicProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: make(map[string]string),
		options:      make(map[string]interface{}),
		logger:       utils.NewLogger(utils.LogLevelInfo), // Default logger
	}

	// Copy the provided extraHeaders
	for k, v := range extraHeaders {
		provider.extraHeaders[k] = v
	}

	// Add the caching header if it's not already present
	if _, exists := provider.extraHeaders["anthropic-beta"]; !exists {
		provider.extraHeaders["anthropic-beta"] = "prompt-caching-2024-07-31"
	}

	return provider
}

// SetLogger configures the logger for the Anthropic provider.
// This is used for debugging and monitoring API interactions.
func (p *AnthropicProvider) SetLogger(logger utils.Logger) {
	p.logger = logger
}

// SetOption sets a specific option for the Anthropic provider.
// Supported options include:
//   - temperature: Controls randomness (0.0 to 1.0)
//   - max_tokens: Maximum tokens in the response
//   - top_p: Nucleus sampling parameter
//   - top_k: Top-k sampling parameter
//   - stop_sequences: Custom stop sequences
func (p *AnthropicProvider) SetOption(key string, value interface{}) {
	p.options[key] = value
}

// SetDefaultOptions configures standard options from the global configuration.
// This includes temperature, max tokens, and sampling parameters.
func (p *AnthropicProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}
}

// Name returns "anthropic" as the provider identifier.
func (p *AnthropicProvider) Name() string {
	return "anthropic"
}

// Endpoint returns the Anthropic API endpoint URL.
// For API version 2024-02-15, this is "https://api.anthropic.com/v1/messages".
func (p *AnthropicProvider) Endpoint() string {
	return "https://api.anthropic.com/v1/messages"
}

// SupportsJSONSchema indicates that Anthropic supports structured output
// through its system prompts and response formatting capabilities.
func (p *AnthropicProvider) SupportsJSONSchema() bool {
	return true
}

// Headers returns the required HTTP headers for Anthropic API requests.
// This includes:
//   - x-api-key: API key for authentication
//   - anthropic-version: API version identifier
//   - Content-Type: application/json
//   - Any additional headers specified via SetExtraHeaders
func (p *AnthropicProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":      "application/json",
		"x-api-key":         p.apiKey,
		"anthropic-version": "2023-06-01",
		"anthropic-beta":    "prompt-caching-2024-07-31",
	}
	return headers
}

// PrepareRequest creates the request body for an Anthropic API call.
// It handles:
//   - Message formatting
//   - System prompts
//   - Response formatting
//   - Model-specific options
//
// Parameters:
//   - prompt: The input text or conversation
//   - options: Additional parameters for the request
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *AnthropicProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	requestBody := map[string]interface{}{
		"model":      p.model,
		"max_tokens": p.options["max_tokens"],
		"system":     []map[string]interface{}{},
		"messages":   []map[string]interface{}{},
	}

	// Handle system prompt
	systemPrompt := ""
	if sp, ok := options["system_prompt"].(string); ok && sp != "" {
		systemPrompt = sp
	}

	// If we have tools, add tool usage instructions to the system prompt
	if tools, ok := options["tools"].([]utils.Tool); ok && len(tools) > 0 {
		anthropicTools := make([]map[string]interface{}, len(tools))
		for i, tool := range tools {
			// Handle web_search as a special built-in tool
			if tool.Type == "web_search" {
				webSearchTool := map[string]interface{}{
					"type": "web_search_20250305",
					"name": "web_search",
				}

				// Add max_uses if provided
				if maxUses, ok := tool.MaxUses.(int); ok && maxUses > 0 {
					webSearchTool["max_uses"] = maxUses
				}

				// Add allowed_domains from Filters or AllowedDomains
				var allowedDomains []string
				if tool.AllowedDomains != nil {
					if domains, ok := tool.AllowedDomains.([]string); ok {
						allowedDomains = domains
					}
				} else if tool.Filters != nil {
					if filters, ok := tool.Filters.(map[string]interface{}); ok {
						if domains, ok := filters["allowed_domains"].([]string); ok {
							allowedDomains = domains
						}
					}
				}
				if len(allowedDomains) > 0 {
					webSearchTool["allowed_domains"] = allowedDomains
				}

				// Add blocked_domains if provided (mutually exclusive with allowed_domains)
				if tool.BlockedDomains != nil {
					if domains, ok := tool.BlockedDomains.([]string); ok && len(domains) > 0 {
						// Only add blocked_domains if allowed_domains wasn't set
						if len(allowedDomains) == 0 {
							webSearchTool["blocked_domains"] = domains
						}
					}
				}

				// Add user_location if provided
				if userLoc, ok := tool.UserLocation.(map[string]interface{}); ok {
					webSearchTool["user_location"] = userLoc
				}

				anthropicTools[i] = webSearchTool
			} else {
				// Handle regular function tools
				anthropicTools[i] = map[string]interface{}{
					"name":         tool.Function.Name,
					"description":  tool.Function.Description,
					"input_schema": tool.Function.Parameters,
				}
			}
		}
		requestBody["tools"] = anthropicTools

		// Add tool usage instructions to system prompt
		if len(tools) > 1 {
			toolUsagePrompt := "When multiple tools are needed to answer a question, you should identify all required tools upfront and use them all at once in your response, rather than using them sequentially. Do not wait for tool results before calling other tools."
			if systemPrompt != "" {
				systemPrompt = toolUsagePrompt + "\n\n" + systemPrompt
			} else {
				systemPrompt = toolUsagePrompt
			}
		}

		// Only set tool_choice when tools are provided
		if toolChoice, ok := options["tool_choice"].(string); ok {
			requestBody["tool_choice"] = map[string]interface{}{
				"type": toolChoice,
			}
		} else {
			// Default to auto for tool choice when tools are provided
			requestBody["tool_choice"] = map[string]interface{}{
				"type": "auto",
			}
		}
	}

	// Add system prompt if we have one
	if systemPrompt != "" {
		parts := splitSystemPrompt(systemPrompt, 3)
		for i, part := range parts {
			systemMessage := map[string]interface{}{
				"type": "text",
				"text": part,
			}
			if i > 0 {
				systemMessage["cache_control"] = map[string]string{"type": "ephemeral"}
			}
			requestBody["system"] = append(requestBody["system"].([]map[string]interface{}), systemMessage)
		}
	}

	// Build user message content
	userContent := []map[string]interface{}{}

	// Check if we have images to include - use shared helper for conversion
	images, hasImages := options["images"].([]types.ContentPart)
	if hasImages && len(images) > 0 {
		// Add images first, then text (Anthropic prefers this order)
		userContent = append(userContent, ConvertImagesToAnthropicContent(images)...)
	}

	// Add text content
	textContent := map[string]interface{}{
		"type": "text",
		"text": prompt,
	}
	// Add cache_control only if caching is enabled
	if caching, ok := options["enable_caching"].(bool); ok && caching {
		textContent["cache_control"] = map[string]string{"type": "ephemeral"}
	}
	userContent = append(userContent, textContent)

	userMessage := map[string]interface{}{
		"role":    "user",
		"content": userContent,
	}

	requestBody["messages"] = append(requestBody["messages"].([]map[string]interface{}), userMessage)

	// Add other options
	for k, v := range options {
		if k != "system_prompt" && k != "max_tokens" && k != "tools" && k != "tool_choice" && k != "enable_caching" && k != "reasoning_effort" && k != "strict_tools" && k != "images" {
			requestBody[k] = v
		}
	}

	// Configure thinking from reasoning_effort (adaptive vs. budgeted by model).
	// Runs after the passthrough so it sees the final max_tokens and wins over
	// any raw thinking/output_config option.
	p.applyThinking(requestBody, options)

	return json.Marshal(requestBody)
}

// anthropicUsesLegacyThinking reports models that predate adaptive thinking and
// must use manual extended thinking (thinking:{type:"enabled",budget_tokens:N}).
// This is a deliberately frozen set: Anthropic will not ship new sub-4.6 models,
// so newer and unknown models default to adaptive — the forward-compatible
// direction, since budget_tokens is being removed across the line and is already
// rejected with a 400 on Opus 4.7+/Sonnet 5/Fable 5. Failing open to adaptive
// keeps the next Opus/Sonnet release working without a code change.
func anthropicUsesLegacyThinking(model string) bool {
	m := strings.ToLower(model)
	for _, legacy := range []string{
		"opus-4-5", "opus-4-1", "opus-4-0", "opus-3",
		"sonnet-4-5", "sonnet-4-0", "sonnet-3",
		"haiku-4-5", "haiku-3",
		"claude-2",
	} {
		if strings.Contains(m, legacy) {
			return true
		}
	}
	return false
}

// anthropicSupportsXhighEffort reports whether the model accepts the "xhigh"
// effort level, which Opus 4.7 introduced. Opus 4.6 and Sonnet 4.6 use adaptive
// thinking but reject "xhigh" (they top out at "max"), so it must be demoted for
// them.
func anthropicSupportsXhighEffort(model string) bool {
	m := strings.ToLower(model)
	return strings.Contains(m, "opus-4-7") || strings.Contains(m, "opus-4-8") ||
		strings.Contains(m, "sonnet-5") || strings.Contains(m, "fable") || strings.Contains(m, "mythos")
}

// normalizeAnthropicEffort maps a reasoning_effort hint onto the effort levels
// Anthropic's output_config accepts (low/medium/high/xhigh/max). OpenAI's
// "minimal" collapses to "low"; anything unrecognized defaults to "medium".
func normalizeAnthropicEffort(effort string) string {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "minimal", "low":
		return "low"
	case "high":
		return "high"
	case "xhigh":
		return "xhigh"
	case "max":
		return "max"
	default:
		return "medium"
	}
}

// anthropicThinkingBudget maps a reasoning_effort level onto a budget_tokens
// value for legacy models that use manual extended thinking. The API requires
// 1024 <= budget_tokens < max_tokens; ok is false when that constraint cannot be
// enforced — max_tokens unknown (<= 0) or too small (<= 1024) — signalling the
// caller to skip thinking rather than emit a budget that may violate the API.
func anthropicThinkingBudget(effort string, maxTokens int) (int, bool) {
	// Without a known, sufficient max_tokens we cannot guarantee
	// budget_tokens < max_tokens, so decline rather than risk a 400.
	if maxTokens <= 1024 {
		return 0, false
	}
	var budget int
	switch normalizeAnthropicEffort(effort) {
	case "low":
		budget = 4096
	case "high", "xhigh", "max":
		budget = 16384
	default: // medium
		budget = 8192
	}
	if budget >= maxTokens {
		budget = maxTokens - 1
	}
	if budget < 1024 {
		budget = 1024
	}
	return budget, true
}

// applyThinking configures Anthropic thinking on the request body from the
// reasoning_effort option. It is a no-op when no reasoning effort is requested,
// so thinking stays off by default (matching Anthropic's default on Opus 4.7+).
// It must run AFTER the caller's option passthrough so it reads the final
// max_tokens and takes precedence over any raw thinking/output_config option.
//
// Adaptive-thinking models get thinking:{type:"adaptive"} with depth steered via
// output_config.effort — never budget_tokens, which they reject. Legacy models
// get manual extended thinking with an effort-derived budget_tokens instead.
func (p *AnthropicProvider) applyThinking(requestBody map[string]interface{}, options map[string]interface{}) {
	// A per-request reasoning_effort takes precedence over a provider-level
	// default (set via SetOption), mirroring how the OpenAI provider merges
	// request options over provider defaults.
	effort, ok := optionString(options["reasoning_effort"])
	if !ok || effort == "" {
		effort, ok = optionString(p.options["reasoning_effort"])
	}
	if !ok || effort == "" {
		return
	}

	if !anthropicUsesLegacyThinking(p.model) {
		requestBody["thinking"] = map[string]interface{}{"type": "adaptive"}
		eff := normalizeAnthropicEffort(effort)
		if eff == "xhigh" && !anthropicSupportsXhighEffort(p.model) {
			eff = "high"
		}
		// Merge into any caller-supplied output_config rather than clobbering it.
		oc, _ := requestBody["output_config"].(map[string]interface{})
		if oc == nil {
			oc = map[string]interface{}{}
		}
		oc["effort"] = eff
		requestBody["output_config"] = oc
		return
	}

	// Validate the budget against the max_tokens actually in the request body
	// (toInt handles int/float64/int64; any other type yields 0 → thinking is
	// skipped, never sent with an unenforceable budget).
	if budget, ok := anthropicThinkingBudget(effort, toInt(requestBody["max_tokens"])); ok {
		requestBody["thinking"] = map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": budget,
		}
	}
}

// Helper function to split the system prompt into a maximum of n parts
func splitSystemPrompt(prompt string, n int) []string {
	if n <= 1 {
		return []string{prompt}
	}

	// Split the prompt into paragraphs
	paragraphs := strings.Split(prompt, "\n\n")

	if len(paragraphs) <= n {
		return paragraphs
	}

	// If we have more paragraphs than allowed parts, we need to combine some
	result := make([]string, n)
	paragraphsPerPart := len(paragraphs) / n
	extraParagraphs := len(paragraphs) % n

	currentIndex := 0
	for i := 0; i < n; i++ {
		end := currentIndex + paragraphsPerPart
		if i < extraParagraphs {
			end++
		}
		result[i] = strings.Join(paragraphs[currentIndex:end], "\n\n")
		currentIndex = end
	}

	return result
}

// PrepareRequestWithSchema creates a request that includes structured output formatting.
// This uses Anthropic's system prompts to enforce response structure.
//
// Parameters:
//   - prompt: The input text or conversation
//   - options: Additional request parameters
//   - schema: JSON schema for response validation
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *AnthropicProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	schemaObj, err := normalizeSchema(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to normalize schema: %w", err)
	}
	schemaJSON, err := json.Marshal(schemaObj)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %w", err)
	}

	// Create a system message that enforces the JSON schema
	systemMsg := fmt.Sprintf("You must respond with a JSON object that strictly adheres to this schema:\n%s\nDo not include any explanatory text, only output valid JSON.", string(schemaJSON))

	requestBody := map[string]interface{}{
		"model":  p.model,
		"system": systemMsg,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}

	// Add any additional options
	for k, v := range options {
		if k != "system_prompt" && k != "reasoning_effort" && k != "strict_tools" { // Skip system_prompt as we're using it for schema, skip OpenAI-specific params
			requestBody[k] = v
		}
	}

	// Configure thinking after the passthrough so budget_tokens can be validated
	// against the max_tokens the caller supplied via options.
	p.applyThinking(requestBody, options)

	return json.Marshal(requestBody)
}

// ParseResponse extracts the generated text from the Anthropic API response.
// It handles various response formats and error cases.
//
// Parameters:
//   - body: Raw API response body
//
// Returns:
//   - Generated text content
//   - Any error encountered during parsing
func (p *AnthropicProvider) ParseResponse(body []byte) (string, error) {
	p.logger.Debug("Raw API response: %s", string(body))

	var response struct {
		ID      string `json:"id"`
		Type    string `json:"type"`
		Role    string `json:"role"`
		Model   string `json:"model"`
		Content []struct {
			Type  string          `json:"type"`
			Text  string          `json:"text,omitempty"`
			ID    string          `json:"id,omitempty"`
			Name  string          `json:"name,omitempty"`
			Input json.RawMessage `json:"input,omitempty"`
		} `json:"content"`
		StopReason string  `json:"stop_reason"`
		StopSeq    *string `json:"stop_sequence"`
		Usage      struct {
			InputTokens              int `json:"input_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			OutputTokens             int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return "", fmt.Errorf("error parsing response: %w", err)
	}
	if len(response.Content) == 0 {
		// Anthropic's native field is stop_reason; write it under the finish_reason:
		// label (and keep the backend sentinel intact) so classification is uniform
		// across providers.
		return "", fmt.Errorf("no content or tool calls in response (finish_reason: %q, completion_tokens: %d)",
			response.StopReason, response.Usage.OutputTokens)
	}

	p.logger.Debug("Number of content blocks: %d", len(response.Content))
	p.logger.Debug("Stop reason: %s", response.StopReason)

	var finalResponse strings.Builder
	var functionCalls []string
	var pendingText strings.Builder
	var lastType string

	// First pass: collect all function calls and text
	for i, content := range response.Content {
		p.logger.Debug("Processing content block %d: type=%s", i, content.Type)

		switch content.Type {
		case "text":
			// If we have pending text and this is also text, add a space
			if lastType == "text" && pendingText.Len() > 0 {
				pendingText.WriteString(" ")
			}
			pendingText.WriteString(content.Text)
			p.logger.Debug("Added text content: %s", content.Text)

		case "tool_use", "tool_calls":
			// If we have any pending text, add it to the final response
			if pendingText.Len() > 0 {
				if finalResponse.Len() > 0 {
					finalResponse.WriteString("\n")
				}
				finalResponse.WriteString(pendingText.String())
				pendingText.Reset()
			}

			// Parse input as raw JSON to preserve the exact format
			var args interface{}
			if err := json.Unmarshal(content.Input, &args); err != nil {
				p.logger.Debug("Error parsing tool input: %v, raw input: %s", err, string(content.Input))
				return "", fmt.Errorf("error parsing tool input: %w", err)
			}

			functionCall, err := utils.FormatFunctionCall(content.Name, args)
			if err != nil {
				p.logger.Debug("Error formatting function call: %v", err)
				return "", fmt.Errorf("error formatting function call: %w", err)
			}
			functionCalls = append(functionCalls, functionCall)
			p.logger.Debug("Added function call: %s", functionCall)
		}
		lastType = content.Type
	}

	// Add any remaining pending text
	if pendingText.Len() > 0 {
		if finalResponse.Len() > 0 {
			finalResponse.WriteString("\n")
		}
		finalResponse.WriteString(pendingText.String())
	}

	p.logger.Debug("Number of function calls collected: %d", len(functionCalls))
	for i, call := range functionCalls {
		p.logger.Debug("Function call %d: %s", i, call)
	}

	// Add all function calls at the end
	if len(functionCalls) > 0 {
		if finalResponse.Len() > 0 {
			finalResponse.WriteString("\n")
		}
		finalResponse.WriteString(strings.Join(functionCalls, "\n"))
	}

	result := finalResponse.String()
	p.logger.Debug("Final response: %s", result)
	return result, nil
}

// ParseResponseWithUsage extracts both the generated text and response details from the Anthropic API response.
// It handles Anthropic-specific response formats and normalizes data to a common structure.
//
// Parameters:
//   - body: Raw API response body
//
// Returns:
//   - Generated text content
//   - Response details (or nil if not available)
//   - Any error encountered during parsing
func (p *AnthropicProvider) ParseResponseWithUsage(body []byte) (string, *types.ResponseDetails, error) {
	p.logger.Debug("Raw API response: %s", string(body))

	var response struct {
		ID      string `json:"id"`
		Type    string `json:"type"`
		Role    string `json:"role"`
		Model   string `json:"model"`
		Content []struct {
			Type      string          `json:"type"`
			Text      string          `json:"text,omitempty"`
			ID        string          `json:"id,omitempty"`
			Name      string          `json:"name,omitempty"`
			Input     json.RawMessage `json:"input,omitempty"`
			ToolUseID string          `json:"tool_use_id,omitempty"` // For web_search_tool_result
			Content   json.RawMessage `json:"content,omitempty"`     // For web_search_tool_result content
			Citations []struct {
				Type           string `json:"type"`
				URL            string `json:"url"`
				Title          string `json:"title"`
				EncryptedIndex string `json:"encrypted_index"`
				CitedText      string `json:"cited_text"`
			} `json:"citations,omitempty"` // For text blocks with citations
		} `json:"content"`
		StopReason string  `json:"stop_reason"`
		StopSeq    *string `json:"stop_sequence"`
		Usage      struct {
			InputTokens              int `json:"input_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			OutputTokens             int `json:"output_tokens"`
			// Cache writes split by lifetime: 5-minute entries bill at 1.25x input,
			// 1-hour entries at 2x, so the aggregate above cannot price them.
			CacheCreation *struct {
				Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens"`
				Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens"`
			} `json:"cache_creation,omitempty"`
			// Extended thinking output. Anthropic's name for what OpenAI calls
			// reasoning tokens; billed at the output rate and included in OutputTokens.
			OutputTokensDetails *struct {
				ThinkingTokens int `json:"thinking_tokens"`
			} `json:"output_tokens_details,omitempty"`
			ServerToolUse *struct {
				WebSearchRequests int `json:"web_search_requests"`
				WebFetchRequests  int `json:"web_fetch_requests"`
			} `json:"server_tool_use,omitempty"` // For server-side tool usage tracking
			// Batch is roughly half rate and priority a premium, so the tier scales
			// every count above.
			ServiceTier string `json:"service_tier"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", nil, fmt.Errorf("error parsing response: %w", err)
	}

	if len(response.Content) == 0 {
		// Anthropic's native field is stop_reason; write it under the finish_reason:
		// label (and keep the backend sentinel intact) so classification is uniform
		// across providers.
		return "", nil, fmt.Errorf("no content or tool calls in response (finish_reason: %q, completion_tokens: %d)",
			response.StopReason, response.Usage.OutputTokens)
	}

	// Extract and normalize response details including ID and usage information
	details := &types.ResponseDetails{
		ID:    response.ID,
		Model: response.Model,
		TokenUsage: types.TokenUsage{
			PromptTokens:     response.Usage.InputTokens,
			CompletionTokens: response.Usage.OutputTokens,
			// Total is computed below, once the cache counts are in place: Anthropic reports
			// no total of its own and bills cache reads and writes on top of InputTokens.
			CacheCreationInputTokens: response.Usage.CacheCreationInputTokens,
			CacheReadInputTokens:     response.Usage.CacheReadInputTokens,
		},
		ServiceTier: response.Usage.ServiceTier,
	}
	if d := response.Usage.CacheCreation; d != nil {
		details.TokenUsage.CacheCreation5mInputTokens = d.Ephemeral5mInputTokens
		details.TokenUsage.CacheCreation1hInputTokens = d.Ephemeral1hInputTokens
		// Older responses carry only the aggregate; newer ones carry only the split.
		// Derive whichever is missing so both are always available to price against.
		if details.TokenUsage.CacheCreationInputTokens == 0 {
			details.TokenUsage.CacheCreationInputTokens = d.Ephemeral5mInputTokens + d.Ephemeral1hInputTokens
		}
	}
	if d := response.Usage.OutputTokensDetails; d != nil {
		details.TokenUsage.ReasoningTokens = d.ThinkingTokens
	}
	details.TokenUsage.TotalTokens = details.TokenUsage.ComputedTotal()

	// Initialize metadata for web search data
	if details.Metadata == nil {
		details.Metadata = make(map[string]interface{})
	}

	// Add server-side tool usage if present. These are billed per request, separately
	// from tokens, so they have to be recorded to cost a web-search turn correctly.
	if stu := response.Usage.ServerToolUse; stu != nil {
		if stu.WebSearchRequests > 0 {
			details.Metadata["web_search_requests"] = stu.WebSearchRequests
		}
		if stu.WebFetchRequests > 0 {
			details.Metadata["web_fetch_requests"] = stu.WebFetchRequests
		}
	}

	p.logger.Debug("Number of content blocks: %d", len(response.Content))
	p.logger.Debug("Stop reason: %s", response.StopReason)

	var finalResponse strings.Builder
	var functionCalls []string
	var pendingText strings.Builder
	var lastType string

	// Collect web search specific data
	var serverToolUses []types.AnthropicServerToolUse
	var webSearchResults []types.AnthropicWebSearchResult
	var citations []types.AnthropicWebSearchResultLocation

	// First pass: collect all function calls, text, and web search data
	for i, content := range response.Content {
		p.logger.Debug("Processing content block %d: type=%s", i, content.Type)

		switch content.Type {
		case "text":
			// If we have pending text and this is also text, add a space
			if lastType == "text" && pendingText.Len() > 0 {
				pendingText.WriteString(" ")
			}
			pendingText.WriteString(content.Text)
			p.logger.Debug("Added text content: %s", content.Text)

			// Extract citations if present
			if len(content.Citations) > 0 {
				for _, cit := range content.Citations {
					citations = append(citations, types.AnthropicWebSearchResultLocation{
						Type:           cit.Type,
						URL:            cit.URL,
						Title:          cit.Title,
						EncryptedIndex: cit.EncryptedIndex,
						CitedText:      cit.CitedText,
					})
				}
			}

		case "server_tool_use":
			// Handle web search tool invocations
			var input map[string]interface{}
			if len(content.Input) > 0 {
				if err := json.Unmarshal(content.Input, &input); err != nil {
					p.logger.Debug("Error parsing server_tool_use input: %v", err)
				}
			}

			serverToolUses = append(serverToolUses, types.AnthropicServerToolUse{
				Type:  content.Type,
				ID:    content.ID,
				Name:  content.Name,
				Input: input,
			})
			p.logger.Debug("Added server_tool_use: %s (id: %s)", content.Name, content.ID)

		case "web_search_tool_result":
			// Handle web search results
			if len(content.Content) > 0 {
				var resultContent []types.AnthropicWebSearchResult
				if err := json.Unmarshal(content.Content, &resultContent); err != nil {
					p.logger.Debug("Error parsing web_search_tool_result content: %v", err)
				} else {
					webSearchResults = append(webSearchResults, resultContent...)
				}
			}
			p.logger.Debug("Added web_search_tool_result for tool_use_id: %s", content.ToolUseID)

		case "tool_use", "tool_calls":
			// If we have any pending text, add it to the final response
			if pendingText.Len() > 0 {
				if finalResponse.Len() > 0 {
					finalResponse.WriteString("\n")
				}
				finalResponse.WriteString(pendingText.String())
				pendingText.Reset()
			}

			// Preserve structured tool call data on ResponseDetails
			details.ToolCalls = append(details.ToolCalls, types.NewToolCall(content.ID, content.Name, content.Input))

			// Parse input as raw JSON to preserve the exact format
			var args interface{}
			if err := json.Unmarshal(content.Input, &args); err != nil {
				p.logger.Debug("Error parsing tool input: %v, raw input: %s", err, string(content.Input))
				return "", nil, fmt.Errorf("error parsing tool input: %w", err)
			}

			functionCall, err := utils.FormatFunctionCall(content.Name, args)
			if err != nil {
				p.logger.Debug("Error formatting function call: %v", err)
				return "", nil, fmt.Errorf("error formatting function call: %w", err)
			}
			functionCalls = append(functionCalls, functionCall)
			p.logger.Debug("Added function call: %s", functionCall)
		}
		lastType = content.Type
	}

	// Store web search data in metadata
	if len(serverToolUses) > 0 {
		details.Metadata["server_tool_uses"] = serverToolUses
	}
	if len(webSearchResults) > 0 {
		details.Metadata["web_search_results"] = webSearchResults
	}
	if len(citations) > 0 {
		details.Metadata["citations"] = citations
	}

	// Add any remaining pending text
	if pendingText.Len() > 0 {
		if finalResponse.Len() > 0 {
			finalResponse.WriteString("\n")
		}
		finalResponse.WriteString(pendingText.String())
	}

	p.logger.Debug("Number of function calls collected: %d", len(functionCalls))
	for i, call := range functionCalls {
		p.logger.Debug("Function call %d: %s", i, call)
	}

	// Add all function calls at the end
	if len(functionCalls) > 0 {
		if finalResponse.Len() > 0 {
			finalResponse.WriteString("\n")
		}
		finalResponse.WriteString(strings.Join(functionCalls, "\n"))
	}

	result := finalResponse.String()
	p.logger.Debug("Final response: %s", result)
	return result, details, nil
}

// HandleFunctionCalls processes structured output in the response.
// This supports Anthropic's response formatting capabilities.
func (p *AnthropicProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	p.logger.Debug("Handling function calls from response")
	response := string(body)

	functionCalls, err := utils.ExtractFunctionCalls(response)
	if err != nil {
		return nil, fmt.Errorf("error extracting function calls: %w", err)
	}

	if len(functionCalls) == 0 {
		p.logger.Debug("No function calls found in the response")
		return nil, nil
	}

	p.logger.Debug("Function calls to handle: %v", functionCalls)
	return json.Marshal(functionCalls)
}

// SetExtraHeaders configures additional HTTP headers for API requests.
// This allows for custom headers needed for specific features or requirements.
func (p *AnthropicProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
}

// SupportsStreaming indicates whether streaming is supported
func (p *AnthropicProvider) SupportsStreaming() bool {
	return true
}

// PrepareStreamRequest creates a request body for streaming API calls
func (p *AnthropicProvider) PrepareStreamRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	requestBody := map[string]interface{}{
		"model":  p.model,
		"stream": true,
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": prompt,
			},
		},
		"max_tokens": 1024, // Default max tokens
	}

	// Add system prompt if present
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		requestBody["system"] = systemPrompt
		delete(options, "system_prompt")
	}

	// Add max tokens if present
	if maxTokens, ok := options["max_tokens"].(int); ok {
		requestBody["max_tokens"] = maxTokens
		delete(options, "max_tokens")
	}

	// Add temperature if present
	if temperature, ok := options["temperature"].(float64); ok {
		requestBody["temperature"] = temperature
		delete(options, "temperature")
	}

	// Add other options
	for k, v := range options {
		if k != "stream" && k != "reasoning_effort" && k != "strict_tools" { // Don't override stream setting, skip OpenAI-specific params
			requestBody[k] = v
		}
	}

	// Configure thinking after the passthrough so it sees the final max_tokens
	// and wins over any raw thinking/output_config option.
	p.applyThinking(requestBody, options)

	return json.Marshal(requestBody)
}

// PrepareStreamRequestWithMessages preserves structured (system + multi-turn)
// messages instead of flattening to one user turn: it reuses
// PrepareRequestWithMessages for parity and enables streaming.
func (p *AnthropicProvider) PrepareStreamRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error) {
	body, err := p.PrepareRequestWithMessages(messages, options)
	if err != nil {
		return nil, err
	}
	var requestBody map[string]interface{}
	if err := json.Unmarshal(body, &requestBody); err != nil {
		return nil, err
	}
	requestBody["stream"] = true
	return json.Marshal(requestBody)
}

// ParseStreamResponse processes a single chunk from a streaming response
func (p *AnthropicProvider) ParseStreamResponse(chunk []byte) (string, error) {
	// Skip empty lines
	if len(bytes.TrimSpace(chunk)) == 0 {
		return "", types.ErrStreamSkip
	}

	// Check for [DONE] marker
	if bytes.Equal(bytes.TrimSpace(chunk), []byte("[DONE]")) {
		return "", io.EOF
	}

	// Parse the event
	var event struct {
		Type  string `json:"type"`
		Index int    `json:"index"`
		Delta struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"delta"`
	}

	if err := json.Unmarshal(chunk, &event); err != nil {
		return "", fmt.Errorf("malformed event: %w", err)
	}

	// Handle different event types
	switch event.Type {
	case "content_block_delta":
		if event.Delta.Type == "text_delta" {
			if event.Delta.Text == "" {
				return "", types.ErrStreamSkip
			}
			return event.Delta.Text, nil
		}
		return "", types.ErrStreamSkip
	case "message_stop":
		return "", io.EOF
	default:
		return "", types.ErrStreamSkip
	}
}

// ParseStreamResponseRich extends ParseStreamResponse with token usage and the
// stop reason. Anthropic splits usage across events: message_start carries input
// tokens, and the final message_delta carries output tokens plus stop_reason —
// so a consumer must accumulate usage across chunks.
func (p *AnthropicProvider) ParseStreamResponseRich(chunk []byte) (types.StreamChunk, error) {
	return parseAnthropicStreamChunk(chunk)
}

// anthropicStreamUsage is the usage shape Anthropic sends on both message_start (input side)
// and message_delta (output side). One type covers both because unknown keys decode to zero.
type anthropicStreamUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	CacheCreation            *struct {
		Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens"`
		Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens"`
	} `json:"cache_creation,omitempty"`
	OutputTokensDetails *struct {
		ThinkingTokens int `json:"thinking_tokens"`
	} `json:"output_tokens_details,omitempty"`
	ServiceTier string `json:"service_tier"`
}

func (u anthropicStreamUsage) normalize() types.TokenUsage {
	usage := types.TokenUsage{
		PromptTokens:             u.InputTokens,
		CompletionTokens:         u.OutputTokens,
		CacheCreationInputTokens: u.CacheCreationInputTokens,
		CacheReadInputTokens:     u.CacheReadInputTokens,
	}
	if d := u.CacheCreation; d != nil {
		usage.CacheCreation5mInputTokens = d.Ephemeral5mInputTokens
		usage.CacheCreation1hInputTokens = d.Ephemeral1hInputTokens
		if usage.CacheCreationInputTokens == 0 {
			usage.CacheCreationInputTokens = d.Ephemeral5mInputTokens + d.Ephemeral1hInputTokens
		}
	}
	if d := u.OutputTokensDetails; d != nil {
		usage.ReasoningTokens = d.ThinkingTokens
	}
	return usage
}

// parseAnthropicStreamChunk is the Anthropic-format stream parser, shared with the generic
// provider so Anthropic-compatible gateways report usage the same way the first-party client does.
func parseAnthropicStreamChunk(chunk []byte) (types.StreamChunk, error) {
	trimmed := bytes.TrimSpace(chunk)
	if len(trimmed) == 0 {
		return types.StreamChunk{}, types.ErrStreamSkip
	}
	if bytes.Equal(trimmed, []byte("[DONE]")) {
		return types.StreamChunk{}, io.EOF
	}

	var event struct {
		Type    string `json:"type"`
		Index   int    `json:"index"`
		Message struct {
			// message_start names the model that actually served the stream, which a
			// "-latest" alias resolves to something more specific than what was requested.
			Model string               `json:"model"`
			Usage anthropicStreamUsage `json:"usage"`
		} `json:"message"`
		ContentBlock struct {
			Type string `json:"type"`
			ID   string `json:"id"`
			Name string `json:"name"`
		} `json:"content_block"`
		Delta struct {
			Type        string `json:"type"`
			Text        string `json:"text"`
			PartialJSON string `json:"partial_json"`
			StopReason  string `json:"stop_reason"`
		} `json:"delta"`
		Usage anthropicStreamUsage `json:"usage"`
	}
	if err := json.Unmarshal(trimmed, &event); err != nil {
		return types.StreamChunk{}, fmt.Errorf("malformed event: %w", err)
	}

	switch event.Type {
	case "message_start":
		u := event.Message.Usage.normalize()
		return types.StreamChunk{
			Kind:        "usage",
			Model:       event.Message.Model,
			ServiceTier: event.Message.Usage.ServiceTier,
			Usage:       &u,
		}, nil
	case "content_block_start":
		// A tool_use block opens with its id+name; arguments stream in later as
		// input_json_delta fragments on content_block_delta events. The block's
		// position is the event index, used by the consumer to assemble per-call.
		if event.ContentBlock.Type == "tool_use" {
			return types.StreamChunk{Kind: "tool_call_delta", ToolCallDelta: &types.ToolCallDelta{
				Index: event.Index,
				ID:    event.ContentBlock.ID,
				Name:  event.ContentBlock.Name,
			}}, nil
		}
		return types.StreamChunk{}, types.ErrStreamSkip
	case "content_block_delta":
		if event.Delta.Type == "text_delta" && event.Delta.Text != "" {
			return types.StreamChunk{Kind: "text", Text: event.Delta.Text}, nil
		}
		if event.Delta.Type == "input_json_delta" {
			return types.StreamChunk{Kind: "tool_call_delta", ToolCallDelta: &types.ToolCallDelta{
				Index:        event.Index,
				ArgsFragment: event.Delta.PartialJSON,
			}}, nil
		}
		return types.StreamChunk{}, types.ErrStreamSkip
	case "message_delta":
		// Final event: stop reason, cumulative output tokens, and — when extended
		// thinking ran — the share of those output tokens spent on it.
		u := event.Usage.normalize()
		return types.StreamChunk{
			Kind:         "finish",
			FinishReason: event.Delta.StopReason,
			ServiceTier:  event.Usage.ServiceTier,
			Usage:        &u,
		}, nil
	case "message_stop":
		return types.StreamChunk{}, io.EOF
	default:
		return types.StreamChunk{}, types.ErrStreamSkip
	}
}

// PrepareRequestWithMessages creates a request body using structured message objects
// rather than a flattened prompt string. This enables more efficient caching and
// better preserves conversation structure for the Claude API.
//
// Parameters:
//   - messages: Slice of MemoryMessage objects representing the conversation
//   - options: Additional options for the request
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *AnthropicProvider) PrepareRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error) {
	requestBody := map[string]interface{}{
		"model":      p.model,
		"max_tokens": p.options["max_tokens"],
		"system":     []map[string]interface{}{},
		"messages":   []map[string]interface{}{},
	}

	// Extract system prompt if present in options
	systemPrompt := ""
	if sp, ok := options["system_prompt"].(string); ok && sp != "" {
		systemPrompt = sp
	}

	// Handle system prompt
	if systemPrompt != "" {
		parts := splitSystemPrompt(systemPrompt, 3)
		for i, part := range parts {
			systemMessage := map[string]interface{}{
				"type": "text",
				"text": part,
			}
			if i > 0 {
				systemMessage["cache_control"] = map[string]string{"type": "ephemeral"}
			}
			requestBody["system"] = append(requestBody["system"].([]map[string]interface{}), systemMessage)
		}
	}

	// Process tools if present
	if tools, ok := options["tools"].([]utils.Tool); ok && len(tools) > 0 {
		anthropicTools := make([]map[string]interface{}, len(tools))
		for i, tool := range tools {
			// Handle web_search as a special built-in tool
			if tool.Type == "web_search" {
				webSearchTool := map[string]interface{}{
					"type": "web_search_20250305",
					"name": "web_search",
				}

				// Add max_uses if provided
				if maxUses, ok := tool.MaxUses.(int); ok && maxUses > 0 {
					webSearchTool["max_uses"] = maxUses
				}

				// Add allowed_domains from Filters or AllowedDomains
				var allowedDomains []string
				if tool.AllowedDomains != nil {
					if domains, ok := tool.AllowedDomains.([]string); ok {
						allowedDomains = domains
					}
				} else if tool.Filters != nil {
					if filters, ok := tool.Filters.(map[string]interface{}); ok {
						if domains, ok := filters["allowed_domains"].([]string); ok {
							allowedDomains = domains
						}
					}
				}
				if len(allowedDomains) > 0 {
					webSearchTool["allowed_domains"] = allowedDomains
				}

				// Add blocked_domains if provided (mutually exclusive with allowed_domains)
				if tool.BlockedDomains != nil {
					if domains, ok := tool.BlockedDomains.([]string); ok && len(domains) > 0 {
						// Only add blocked_domains if allowed_domains wasn't set
						if len(allowedDomains) == 0 {
							webSearchTool["blocked_domains"] = domains
						}
					}
				}

				// Add user_location if provided
				if userLoc, ok := tool.UserLocation.(map[string]interface{}); ok {
					webSearchTool["user_location"] = userLoc
				}

				anthropicTools[i] = webSearchTool
			} else {
				// Handle regular function tools
				anthropicTools[i] = map[string]interface{}{
					"name":         tool.Function.Name,
					"description":  tool.Function.Description,
					"input_schema": tool.Function.Parameters,
				}
			}
		}
		requestBody["tools"] = anthropicTools

		// Add tool usage instructions to system prompt if needed
		if len(tools) > 1 {
			toolUsagePrompt := "When multiple tools are needed to answer a question, you should identify all required tools upfront and use them all at once in your response, rather than using them sequentially. Do not wait for tool results before calling other tools."
			// This is separate from the existing system messages
			systemMessage := map[string]interface{}{
				"type": "text",
				"text": toolUsagePrompt,
			}
			requestBody["system"] = append(requestBody["system"].([]map[string]interface{}), systemMessage)
		}

		// Only set tool_choice when tools are provided
		if toolChoice, ok := options["tool_choice"].(string); ok {
			requestBody["tool_choice"] = map[string]interface{}{
				"type": toolChoice,
			}
		} else {
			// Default to auto for tool choice when tools are provided
			requestBody["tool_choice"] = map[string]interface{}{
				"type": "auto",
			}
		}
	}

	// Convert MemoryMessage objects to Anthropic messages
	for _, msg := range messages {
		var content []map[string]interface{}

		// Handle tool result messages (role=tool becomes user with tool_result content)
		if msg.Role == "tool" && msg.ToolCallID != "" {
			// Anthropic expects tool results as user messages with tool_result content block
			content = []map[string]interface{}{
				{
					"type":        "tool_result",
					"tool_use_id": msg.ToolCallID,
					"content":     msg.Content,
				},
			}
			// Check if this is an error result
			if isError, ok := msg.Metadata["is_error"].(bool); ok && isError {
				content[0]["is_error"] = true
			}
			message := map[string]interface{}{
				"role":    "user", // Anthropic uses "user" role for tool results
				"content": content,
			}
			requestBody["messages"] = append(requestBody["messages"].([]map[string]interface{}), message)
			continue
		}

		// Handle assistant messages with tool calls
		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			// Add text content if present
			if msg.Content != "" {
				content = append(content, map[string]interface{}{
					"type": "text",
					"text": msg.Content,
				})
			}
			// Add tool_use content blocks for each tool call
			for _, tc := range msg.ToolCalls {
				// Parse the arguments JSON
				var args interface{}
				if err := json.Unmarshal(tc.Function.Arguments, &args); err != nil {
					args = map[string]interface{}{} // Empty object on parse error
				}
				content = append(content, map[string]interface{}{
					"type":  "tool_use",
					"id":    tc.ID,
					"name":  tc.Function.Name,
					"input": args,
				})
			}
			message := map[string]interface{}{
				"role":    "assistant",
				"content": content,
			}
			requestBody["messages"] = append(requestBody["messages"].([]map[string]interface{}), message)
			continue
		}

		// Check if message has multimodal content
		if msg.HasMultiContent() {
			content = BuildAnthropicContentFromParts(msg.MultiContent)
		} else {
			// Regular text message
			content = []map[string]interface{}{
				{
					"type": "text",
					"text": msg.Content,
				},
			}
		}

		// Add cache_control if specified (to the last content block)
		if len(content) > 0 {
			if msg.CacheControl != "" {
				content[len(content)-1]["cache_control"] = map[string]string{"type": msg.CacheControl}
			} else if caching, ok := options["enable_caching"].(bool); ok && caching {
				// Add default caching if enabled globally
				content[len(content)-1]["cache_control"] = map[string]string{"type": "ephemeral"}
			}
		}

		message := map[string]interface{}{
			"role":    msg.Role,
			"content": content,
		}

		requestBody["messages"] = append(requestBody["messages"].([]map[string]interface{}), message)
	}

	// Add other options
	for k, v := range options {
		if k != "system_prompt" && k != "max_tokens" && k != "tools" && k != "tool_choice" && k != "enable_caching" && k != "structured_messages" && k != "reasoning_effort" && k != "strict_tools" && k != "images" {
			requestBody[k] = v
		}
	}

	// Configure thinking after the passthrough so it sees the final max_tokens
	// and wins over any raw thinking/output_config option.
	p.applyThinking(requestBody, options)

	return json.Marshal(requestBody)
}

// PrepareRequestWithMessagesAndSchema creates a request body using structured message objects
// and a JSON schema for response validation. Since Anthropic doesn't natively support
// structured output schemas, the schema is embedded in the system prompt.
//
// Parameters:
//   - messages: Slice of MemoryMessage objects representing the conversation
//   - options: Additional options for the request
//   - schema: JSON schema for response validation
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *AnthropicProvider) PrepareRequestWithMessagesAndSchema(messages []types.MemoryMessage, options map[string]interface{}, schema interface{}) ([]byte, error) {
	schemaJSON, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %w", err)
	}

	schemaInstruction := fmt.Sprintf("You must respond with a JSON object that strictly adheres to this schema:\n%s\nDo not include any explanatory text, only output valid JSON.", string(schemaJSON))

	// Combine user system prompt with schema instruction
	sp := ""
	if existing, ok := options["system_prompt"].(string); ok && existing != "" {
		sp = existing
	}
	combinedSystemPrompt := schemaInstruction
	if sp != "" {
		combinedSystemPrompt = sp + "\n\n" + schemaInstruction
	}

	newOptions := make(map[string]interface{}, len(options)+1)
	for k, v := range options {
		newOptions[k] = v
	}
	newOptions["system_prompt"] = combinedSystemPrompt

	return p.PrepareRequestWithMessages(messages, newOptions)
}
