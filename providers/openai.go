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

// OpenAIProvider implements the Provider interface for OpenAI's API.
// It supports GPT models and provides access to OpenAI's language model capabilities,
// including function calling, JSON mode, and structured output validation.
type OpenAIProvider struct {
	apiKey       string                 // API key for authentication
	model        string                 // Model identifier (e.g., "gpt-4", "gpt-4o-mini")
	extraHeaders map[string]string      // Additional HTTP headers
	options      map[string]interface{} // Model-specific options
	logger       utils.Logger           // Logger instance
}

// NewOpenAIProvider creates a new OpenAI provider instance.
// It initializes the provider with the given API key, model, and optional headers.
//
// Parameters:
//   - apiKey: OpenAI API key for authentication
//   - model: The model to use (e.g., "gpt-4", "gpt-3.5-turbo")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured OpenAI Provider instance
func NewOpenAIProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	return &OpenAIProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]interface{}),
		logger:       utils.NewLogger(utils.LogLevelInfo),
	}
}

// SetLogger configures the logger for the OpenAI provider.
// This is used for debugging and monitoring API interactions.
func (p *OpenAIProvider) SetLogger(logger utils.Logger) {
	p.logger = logger
}

// needsMaxCompletionTokens checks if the model requires max_completion_tokens instead of max_tokens
func (p *OpenAIProvider) needsMaxCompletionTokens() bool {
	// Check for models that start with "o"
	if strings.HasPrefix(p.model, "o") {
		return true
	}

	// Check for gpt-4o and similar models
	if strings.Contains(p.model, "4o") || strings.Contains(p.model, "-o") || strings.Contains(p.model, "-5") {
		return true
	}

	return false
}

func (p *OpenAIProvider) needsReasoningEffort() bool {
	// Check for models that start with "o"
	if strings.HasPrefix(p.model, "o") {
		return true
	}

	// Check for gpt-5
	if strings.Contains(p.model, "-5") {
		return true
	}

	return false
}

func (p *OpenAIProvider) needsNoTemperature() bool {
	return modelNeedsNoTemperature(p.model)
}

// modelNeedsNoTemperature checks if a given model doesn't support temperature
func modelNeedsNoTemperature(model string) bool {
	// Check for models that start with "o3"
	if strings.HasPrefix(model, "o3") {
		return true
	}

	if strings.Contains(model, "-5") {
		return true
	}

	return false
}

func (p *OpenAIProvider) needsNoToolChoice() bool {
	return modelNeedsNoToolChoice(p.model)
}

// modelNeedsNoToolChoice checks if a given model doesn't support tool_choice
func modelNeedsNoToolChoice(model string) bool {
	// O-series models (o1, o3, o4) don't support tool_choice
	if strings.HasPrefix(model, "o1") || strings.HasPrefix(model, "o3") || strings.HasPrefix(model, "o4") {
		return true
	}

	// GPT-5 models don't support tool_choice
	if strings.Contains(model, "-5") {
		return true
	}

	return false
}

// SetOption sets a specific option for the OpenAI provider.
// Supported options include:
//   - temperature: Controls randomness (0.0 to 2.0)
//   - max_tokens: Maximum tokens in the response (automatically converted to max_completion_tokens for "o" models)
//   - top_p: Nucleus sampling parameter
//   - frequency_penalty: Repetition reduction
//   - presence_penalty: Topic steering
//   - seed: Deterministic sampling seed
func (p *OpenAIProvider) SetOption(key string, value interface{}) {
	// Handle max_tokens conversion for "o" models
	if key == "max_tokens" {
		if p.needsMaxCompletionTokens() {
			// For models requiring max_completion_tokens, use that instead
			key = "max_completion_tokens"
			// Delete max_tokens if it was previously set
			delete(p.options, "max_tokens")
		} else {
			// For models using max_tokens, make sure max_completion_tokens is not set
			delete(p.options, "max_completion_tokens")
		}
	} else if key == "max_completion_tokens" {
		// If explicitly setting max_completion_tokens, remove max_tokens to avoid conflicts
		delete(p.options, "max_tokens")
	}

	// if the option is reasoning_effort, check if the model supports it
	if key == "reasoning_effort" {
		// if it doesn't, remove it from options
		if !p.needsReasoningEffort() {
			delete(p.options, "reasoning_effort")
		}
	}

	if key == "temperature" {
		if p.needsNoTemperature() {
			delete(p.options, "temperature")
		}
	}

	p.options[key] = value
	p.logger.Debug("Option set", "key", key, "value", value)
}

// SetDefaultOptions configures standard options from the global configuration.
// This includes temperature, max tokens, and sampling parameters.
func (p *OpenAIProvider) SetDefaultOptions(config *config.Config) {
	p.SetOption("temperature", config.Temperature)
	p.SetOption("max_tokens", config.MaxTokens)
	if config.Seed != nil {
		p.SetOption("seed", *config.Seed)
	}
	p.logger.Debug("Default options set", "temperature", config.Temperature, "max_tokens", config.MaxTokens, "seed", config.Seed)
}

// Name returns "openai" as the provider identifier.
func (p *OpenAIProvider) Name() string {
	return "openai"
}

// Endpoint returns the OpenAI API endpoint URL.
// For API version 1, this is "https://api.openai.com/v1/chat/completions".
func (p *OpenAIProvider) Endpoint() string {
	return "https://api.openai.com/v1/chat/completions"
}

// SupportsJSONSchema indicates that OpenAI supports native JSON schema validation
// through its function calling and JSON mode capabilities.
func (p *OpenAIProvider) SupportsJSONSchema() bool {
	return true
}

// Headers returns the required HTTP headers for OpenAI API requests.
// This includes:
//   - Authorization: Bearer token using the API key
//   - Content-Type: application/json
//   - Any additional headers specified via SetExtraHeaders
func (p *OpenAIProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + p.apiKey,
	}

	for key, value := range p.extraHeaders {
		headers[key] = value
	}

	p.logger.Debug("Headers prepared", "headers", headers)
	return headers
}

// PrepareRequest creates the request body for an OpenAI API call.
// It handles:
//   - Message formatting
//   - System messages
//   - Function/tool definitions
//   - Model-specific options
//   - Web search tool handling (automatically switches to search models and filters tools)
//
// Note: When web_search tools are detected, this function:
//  1. Automatically switches to an appropriate search model variant (e.g., gpt-4o-search-preview)
//  2. Filters out web_search tools from the tools array (they are NOT passed to the API)
//  3. Search models have built-in search capabilities and don't accept web_search as a tool type
//
// Parameters:
//   - prompt: The input text or conversation
//   - options: Additional parameters for the request
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *OpenAIProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	request := map[string]interface{}{
		"model":    p.model,
		"messages": []map[string]interface{}{},
	}

	// Handle system prompt as developer message
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["messages"] = append(request["messages"].([]map[string]interface{}), map[string]interface{}{
			"role":    "developer",
			"content": systemPrompt,
		})
	}

	// Add user message
	request["messages"] = append(request["messages"].([]map[string]interface{}), map[string]interface{}{
		"role":    "user",
		"content": prompt,
	})

	// Handle tools
	if tools, ok := options["tools"].([]utils.Tool); ok && len(tools) > 0 {
		// Filter out web_search tools - OpenAI doesn't support them
		openAITools := []map[string]interface{}{}
		for _, tool := range tools {
			// Skip web_search tools - OpenAI doesn't support them
			if tool.Type == "web_search" {
				p.logger.Debug("Skipping web_search tool - OpenAI doesn't support web_search")
				continue
			}

			// Handle regular function tools
			if !options["strict_tools"].(bool) {
				openAITools = append(openAITools, map[string]interface{}{
					"type": "function",
					"function": map[string]interface{}{
						"name":        tool.Function.Name,
						"description": tool.Function.Description,
						"parameters":  tool.Function.Parameters,
					},
				})
			} else {
				openAITools = append(openAITools, map[string]interface{}{
					"type": "function",
					"function": map[string]interface{}{
						"name":        tool.Function.Name,
						"description": tool.Function.Description,
						"parameters":  tool.Function.Parameters,
					},
					"strict": true, // Add this if you want strict mode
				})
			}
		}

		// Only add tools to request if we have any non-web_search tools
		if len(openAITools) > 0 {
			request["tools"] = openAITools

			// Handle tool_choice (only if model supports it and we have tools)
			if toolChoice, ok := options["tool_choice"].(string); ok && !p.needsNoToolChoice() {
				request["tool_choice"] = toolChoice
			}
		}
	}

	// Create a merged copy of options to handle token parameters properly
	mergedOptions := make(map[string]interface{})

	// First add options from provider (p.options)
	for k, v := range p.options {
		if k != "tools" && k != "tool_choice" && k != "system_prompt" {
			mergedOptions[k] = v
		}
	}

	// Then add options from the function parameters (may override provider options)
	for k, v := range options {
		if k != "tools" && k != "tool_choice" && k != "strict_tools" && k != "system_prompt" {
			mergedOptions[k] = v
		}
	}

	// Handle max_tokens/max_completion_tokens conflict
	// For models that need max_completion_tokens, ensure we use that and not max_tokens
	if p.needsMaxCompletionTokens() {
		if _, hasMaxTokens := mergedOptions["max_tokens"]; hasMaxTokens {
			// Move max_tokens value to max_completion_tokens
			mergedOptions["max_completion_tokens"] = mergedOptions["max_tokens"]
			delete(mergedOptions, "max_tokens")
		}
	} else {
		// For other models, ensure we use max_tokens and not max_completion_tokens
		if _, hasMaxCompletionTokens := mergedOptions["max_completion_tokens"]; hasMaxCompletionTokens {
			// Move max_completion_tokens value to max_tokens
			mergedOptions["max_tokens"] = mergedOptions["max_completion_tokens"]
			delete(mergedOptions, "max_completion_tokens")
		}
	}

	// Handle reasoning_effort option
	if !p.needsReasoningEffort() {
		delete(mergedOptions, "reasoning_effort")
	}

	// Remove temperature from mergedOptions if model doesn't support it
	if p.needsNoTemperature() {
		delete(mergedOptions, "temperature")
	}

	// Remove tool_choice from request if model doesn't support it
	if p.needsNoToolChoice() {
		delete(request, "tool_choice")
	}

	// Add merged options to the request
	for k, v := range mergedOptions {
		request[k] = v
	}

	return json.Marshal(request)
}

// PrepareRequestWithSchema creates a request that includes JSON schema validation.
// This uses OpenAI's function calling feature to enforce response structure.
//
// Parameters:
//   - prompt: The input text or conversation
//   - options: Additional request parameters
//   - schema: JSON schema for response validation
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *OpenAIProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	p.logger.Debug("Preparing request with schema", "prompt", prompt, "schema", schema)

	// First, ensure we have a proper object for the schema
	var schemaObj interface{}
	switch s := schema.(type) {
	case string:
		if err := json.Unmarshal([]byte(s), &schemaObj); err != nil {
			return nil, fmt.Errorf("failed to unmarshal schema string: %w", err)
		}
	case []byte:
		if err := json.Unmarshal(s, &schemaObj); err != nil {
			return nil, fmt.Errorf("failed to unmarshal schema bytes: %w", err)
		}
	case map[string]interface{}:
		schemaObj = s
	default:
		// Try to marshal and unmarshal to ensure we have a proper object
		schemaBytes, err := json.Marshal(schema)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal schema: %w", err)
		}
		if err := json.Unmarshal(schemaBytes, &schemaObj); err != nil {
			return nil, fmt.Errorf("failed to unmarshal schema: %w", err)
		}
	}

	// Clean the schema for OpenAI by removing unsupported validation rules
	cleanSchema := cleanSchemaForOpenAI(schemaObj)

	// Debug log the cleaned schema
	cleanSchemaJSON, _ := json.MarshalIndent(cleanSchema, "", "  ")
	p.logger.Debug("Cleaned schema for OpenAI", "schema", string(cleanSchemaJSON))

	request := map[string]interface{}{
		"model": p.model,
		"messages": []map[string]interface{}{
			{"role": "user", "content": prompt},
		},
		"response_format": map[string]interface{}{
			"type": "json_schema",
			"json_schema": map[string]interface{}{
				"name":   "structured_response",
				"schema": cleanSchema,
				"strict": true,
			},
		},
	}

	// Handle system prompt as system message
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["messages"] = append([]map[string]interface{}{
			{"role": "system", "content": systemPrompt},
		}, request["messages"].([]map[string]interface{})...)
	}

	// Create a merged copy of options to handle token parameters properly
	mergedOptions := make(map[string]interface{})

	// First add options from provider (p.options)
	for k, v := range p.options {
		if k != "system_prompt" && k != "strict_tools" {
			mergedOptions[k] = v
		}
	}

	// Then add options from the function parameters (may override provider options)
	for k, v := range options {
		if k != "system_prompt" && k != "strict_tools" {
			mergedOptions[k] = v
		}
	}

	// Handle max_tokens/max_completion_tokens conflict
	// For models that need max_completion_tokens, ensure we use that and not max_tokens
	if p.needsMaxCompletionTokens() {
		if _, hasMaxTokens := mergedOptions["max_tokens"]; hasMaxTokens {
			// Move max_tokens value to max_completion_tokens
			mergedOptions["max_completion_tokens"] = mergedOptions["max_tokens"]
			delete(mergedOptions, "max_tokens")
		}
	} else {
		// For other models, ensure we use max_tokens and not max_completion_tokens
		if _, hasMaxCompletionTokens := mergedOptions["max_completion_tokens"]; hasMaxCompletionTokens {
			// Move max_completion_tokens value to max_tokens
			mergedOptions["max_tokens"] = mergedOptions["max_completion_tokens"]
			delete(mergedOptions, "max_completion_tokens")
		}
	}

	// Handle reasoning_effort option
	if !p.needsReasoningEffort() {
		delete(mergedOptions, "reasoning_effort")
	}

	if p.needsNoTemperature() {
		delete(mergedOptions, "temperature")
	}

	// Add merged options to the request
	for k, v := range mergedOptions {
		request[k] = v
	}

	reqJSON, err := json.Marshal(request)
	if err != nil {
		p.logger.Error("Failed to marshal request with schema", "error", err)
		return nil, err
	}

	p.logger.Debug("Full request to OpenAI", "request", string(reqJSON))
	return reqJSON, nil
}

// cleanSchemaForOpenAI removes validation rules that OpenAI doesn't support
func cleanSchemaForOpenAI(schema interface{}) interface{} {
	if schemaMap, ok := schema.(map[string]interface{}); ok {
		result := make(map[string]interface{})
		for k, v := range schemaMap {
			switch k {
			case "type", "properties", "required", "items":
				if k == "properties" {
					props := make(map[string]interface{})
					if propsMap, ok := v.(map[string]interface{}); ok {
						for propName, propSchema := range propsMap {
							props[propName] = cleanSchemaForOpenAI(propSchema)
						}
					}
					result[k] = props
				} else if k == "items" {
					result[k] = cleanSchemaForOpenAI(v)
				} else {
					result[k] = v
				}
			}
		}
		// Add additionalProperties: false at each object level
		if schemaMap["type"] == "object" {
			result["additionalProperties"] = false
		}
		return result
	}
	return schema
}

// ParseResponse extracts the generated text from the OpenAI API response.
// It handles various response formats and error cases.
//
// Parameters:
//   - body: Raw API response body
//
// Returns:
//   - Generated text content
//   - Any error encountered during parsing
func (p *OpenAIProvider) ParseResponse(body []byte) (string, error) {
	var response struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string          `json:"name"`
						Arguments json.RawMessage `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", err
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("empty response from API")
	}

	message := response.Choices[0].Message
	if message.Content != "" {
		return message.Content, nil
	}

	if len(message.ToolCalls) > 0 {
		var functionCalls []string
		for _, call := range message.ToolCalls {
			// Parse arguments as raw JSON to preserve the exact format
			var args interface{}
			if err := json.Unmarshal(call.Function.Arguments, &args); err != nil {
				return "", fmt.Errorf("error parsing function arguments: %w", err)
			}

			functionCall, err := utils.FormatFunctionCall(call.Function.Name, args)
			if err != nil {
				return "", fmt.Errorf("error formatting function call: %w", err)
			}
			functionCalls = append(functionCalls, functionCall)
		}
		return strings.Join(functionCalls, "\n"), nil
	}

	return "", fmt.Errorf("no content or tool calls in response")
}

// ParseResponseWithUsage extracts both the generated text and response details from the OpenAI API response.
// It handles provider-specific response formats and normalizes data to a common structure.
// For web_search responses, it extracts web_search_call items, annotations, and citations.
//
// Parameters:
//   - body: Raw API response body
//
// Returns:
//   - Generated text content
//   - Response details (or nil if not available)
//   - Any error encountered during parsing
func (p *OpenAIProvider) ParseResponseWithUsage(body []byte) (string, *types.ResponseDetails, error) {
	var response struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string          `json:"name"`
						Arguments json.RawMessage `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
		// Web search specific fields (from Responses API format)
		Output []struct {
			Type   string `json:"type"`
			ID     string `json:"id,omitempty"`
			Status string `json:"status,omitempty"`
			Action *struct {
				Type    string   `json:"type,omitempty"`
				Query   string   `json:"query,omitempty"`
				Domains []string `json:"domains,omitempty"`
				Sources []struct {
					URL   string `json:"url"`
					Title string `json:"title,omitempty"`
					Type  string `json:"type,omitempty"`
				} `json:"sources,omitempty"`
			} `json:"action,omitempty"`
			Role    string `json:"role,omitempty"`
			Content []struct {
				Type        string `json:"type"`
				Text        string `json:"text,omitempty"`
				Annotations []struct {
					Type       string `json:"type"`
					StartIndex int    `json:"start_index,omitempty"`
					EndIndex   int    `json:"end_index,omitempty"`
					URL        string `json:"url,omitempty"`
					Title      string `json:"title,omitempty"`
				} `json:"annotations,omitempty"`
			} `json:"content,omitempty"`
		} `json:"output,omitempty"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", nil, fmt.Errorf("error parsing response: %w", err)
	}

	// Extract response details including ID and usage information
	details := &types.ResponseDetails{
		ID:    response.ID,
		Model: response.Model,
		TokenUsage: types.TokenUsage{
			PromptTokens:     response.Usage.PromptTokens,
			CompletionTokens: response.Usage.CompletionTokens,
			TotalTokens:      response.Usage.TotalTokens,
		},
	}

	// Check for web search response format (Responses API with output array)
	if len(response.Output) > 0 {
		// Convert structured output to interface{} slice for parsing
		outputInterfaces := make([]interface{}, len(response.Output))
		for i := range response.Output {
			item := response.Output[i]
			itemMap := map[string]interface{}{
				"type":   item.Type,
				"id":     item.ID,
				"status": item.Status,
			}

			if item.Action != nil {
				actionMap := map[string]interface{}{
					"type": item.Action.Type,
				}
				if item.Action.Query != "" {
					actionMap["query"] = item.Action.Query
				}
				if len(item.Action.Domains) > 0 {
					actionMap["domains"] = item.Action.Domains
				}
				if len(item.Action.Sources) > 0 {
					sources := make([]map[string]interface{}, len(item.Action.Sources))
					for j, src := range item.Action.Sources {
						sources[j] = map[string]interface{}{
							"url":   src.URL,
							"title": src.Title,
							"type":  src.Type,
						}
					}
					actionMap["sources"] = sources
				}
				itemMap["action"] = actionMap
			}

			if item.Role != "" {
				itemMap["role"] = item.Role
			}

			if len(item.Content) > 0 {
				contentMaps := make([]map[string]interface{}, len(item.Content))
				for j, c := range item.Content {
					contentMap := map[string]interface{}{
						"type": c.Type,
					}
					if c.Text != "" {
						contentMap["text"] = c.Text
					}
					if len(c.Annotations) > 0 {
						annMaps := make([]map[string]interface{}, len(c.Annotations))
						for k, ann := range c.Annotations {
							annMaps[k] = map[string]interface{}{
								"type":        ann.Type,
								"start_index": ann.StartIndex,
								"end_index":   ann.EndIndex,
								"url":         ann.URL,
								"title":       ann.Title,
							}
						}
						contentMap["annotations"] = annMaps
					}
					contentMaps[j] = contentMap
				}
				itemMap["content"] = contentMaps
			}

			outputInterfaces[i] = itemMap
		}
		return p.parseWebSearchResponse(outputInterfaces, details)
	}

	// Standard Chat Completions API format
	if len(response.Choices) == 0 {
		return "", nil, fmt.Errorf("empty response from API")
	}

	message := response.Choices[0].Message
	if message.Content != "" {
		return message.Content, details, nil
	}

	if len(message.ToolCalls) > 0 {
		var functionCalls []string
		for _, call := range message.ToolCalls {
			// Parse arguments as raw JSON to preserve the exact format
			var args interface{}
			if err := json.Unmarshal(call.Function.Arguments, &args); err != nil {
				return "", nil, fmt.Errorf("error parsing function arguments: %w", err)
			}

			functionCall, err := utils.FormatFunctionCall(call.Function.Name, args)
			if err != nil {
				return "", nil, fmt.Errorf("error formatting function call: %w", err)
			}
			functionCalls = append(functionCalls, functionCall)
		}
		return strings.Join(functionCalls, "\n"), details, nil
	}

	return "", nil, fmt.Errorf("no content or tool calls in response")
}

// parseWebSearchResponse handles the Responses API format which includes web_search_call items.
func (p *OpenAIProvider) parseWebSearchResponse(output []interface{}, details *types.ResponseDetails) (string, *types.ResponseDetails, error) {
	var webSearchCalls []types.WebSearchCall
	var annotations []types.Annotation
	var citations []types.URLCitation
	var content strings.Builder

	for _, item := range output {
		itemMap, ok := item.(map[string]interface{})
		if !ok {
			continue
		}

		itemType, _ := itemMap["type"].(string)

		switch itemType {
		case "web_search_call":
			// Extract web search call
			call := types.WebSearchCall{
				ID:     itemMap["id"].(string),
				Type:   itemType,
				Status: itemMap["status"].(string),
			}

			if actionData, ok := itemMap["action"].(map[string]interface{}); ok {
				action := &types.WebSearchAction{
					Type: actionData["type"].(string),
				}

				if query, ok := actionData["query"].(string); ok {
					action.Query = query
				}

				if domains, ok := actionData["domains"].([]interface{}); ok {
					for _, d := range domains {
						if domain, ok := d.(string); ok {
							action.Domains = append(action.Domains, domain)
						}
					}
				}

				if sources, ok := actionData["sources"].([]interface{}); ok {
					for _, s := range sources {
						if srcMap, ok := s.(map[string]interface{}); ok {
							source := types.Source{
								URL: srcMap["url"].(string),
							}
							if title, ok := srcMap["title"].(string); ok {
								source.Title = title
							}
							if srcType, ok := srcMap["type"].(string); ok {
								source.Type = srcType
							}
							action.Sources = append(action.Sources, source)
						}
					}
				}

				call.Action = action
			}

			webSearchCalls = append(webSearchCalls, call)

		case "message":
			// Extract message content and annotations
			if contentArray, ok := itemMap["content"].([]interface{}); ok {
				for _, c := range contentArray {
					if contentItem, ok := c.(map[string]interface{}); ok {
						if text, ok := contentItem["text"].(string); ok {
							content.WriteString(text)
						}

						if annArray, ok := contentItem["annotations"].([]interface{}); ok {
							for _, a := range annArray {
								if annMap, ok := a.(map[string]interface{}); ok {
									ann := types.Annotation{
										Type: annMap["type"].(string),
									}

									if startIdx, ok := annMap["start_index"].(float64); ok {
										ann.StartIndex = int(startIdx)
									}
									if endIdx, ok := annMap["end_index"].(float64); ok {
										ann.EndIndex = int(endIdx)
									}
									if url, ok := annMap["url"].(string); ok {
										ann.URL = url
									}
									if title, ok := annMap["title"].(string); ok {
										ann.Title = title
									}

									annotations = append(annotations, ann)

									// Also add to citations for convenience
									if ann.Type == "url_citation" {
										citations = append(citations, types.URLCitation{
											Type:       ann.Type,
											StartIndex: ann.StartIndex,
											EndIndex:   ann.EndIndex,
											URL:        ann.URL,
											Title:      ann.Title,
										})
									}
								}
							}
						}
					}
				}
			}
		}
	}

	// Store web search specific data in metadata
	if details.Metadata == nil {
		details.Metadata = make(map[string]interface{})
	}
	details.Metadata["web_search_calls"] = webSearchCalls
	details.Metadata["annotations"] = annotations
	details.Metadata["citations"] = citations

	return content.String(), details, nil
}

// HandleFunctionCalls processes function calling in the response.
// This supports OpenAI's function calling and JSON mode features.
func (p *OpenAIProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	response := string(body)
	functionCalls, err := utils.ExtractFunctionCalls(response)
	if err != nil {
		return nil, fmt.Errorf("error extracting function calls: %w", err)
	}

	if len(functionCalls) == 0 {
		return nil, fmt.Errorf("no function calls found in response")
	}

	p.logger.Debug("Function calls to handle", "calls", functionCalls)
	return json.Marshal(functionCalls)
}

// SetExtraHeaders configures additional HTTP headers for API requests.
// This allows for custom headers needed for specific features or requirements.
func (p *OpenAIProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
	p.logger.Debug("Extra headers set", "headers", extraHeaders)
}

// SupportsStreaming indicates whether streaming is supported
func (p *OpenAIProvider) SupportsStreaming() bool {
	return true
}

// PrepareStreamRequest creates a request body for streaming API calls
func (p *OpenAIProvider) PrepareStreamRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	// Start with regular request preparation
	requestBody := map[string]interface{}{
		"model": p.model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"stream": true,
	}

	// Create a merged copy of options to handle token parameters properly
	mergedOptions := make(map[string]interface{})

	// First add options from provider (p.options)
	for k, v := range p.options {
		if k != "stream" { // Don't override stream setting
			mergedOptions[k] = v
		}
	}

	// Then add options from the function parameters (may override provider options)
	for k, v := range options {
		if k != "stream" { // Don't override stream setting
			mergedOptions[k] = v
		}
	}

	// Handle max_tokens/max_completion_tokens conflict
	// For models that need max_completion_tokens, ensure we use that and not max_tokens
	if p.needsMaxCompletionTokens() {
		if _, hasMaxTokens := mergedOptions["max_tokens"]; hasMaxTokens {
			// Move max_tokens value to max_completion_tokens
			mergedOptions["max_completion_tokens"] = mergedOptions["max_tokens"]
			delete(mergedOptions, "max_tokens")
		}
	} else {
		// For other models, ensure we use max_tokens and not max_completion_tokens
		if _, hasMaxCompletionTokens := mergedOptions["max_completion_tokens"]; hasMaxCompletionTokens {
			// Move max_completion_tokens value to max_tokens
			mergedOptions["max_tokens"] = mergedOptions["max_completion_tokens"]
			delete(mergedOptions, "max_completion_tokens")
		}
	}

	// Handle reasoning_effort option
	if !p.needsReasoningEffort() {
		delete(mergedOptions, "reasoning_effort")
	}

	if p.needsNoTemperature() {
		delete(mergedOptions, "temperature")
	}

	// Add merged options to the request
	for k, v := range mergedOptions {
		requestBody[k] = v
	}

	return json.Marshal(requestBody)
}

// ParseStreamResponse processes a single chunk from a streaming response
func (p *OpenAIProvider) ParseStreamResponse(chunk []byte) (string, error) {
	// Skip empty lines
	if len(bytes.TrimSpace(chunk)) == 0 {
		return "", fmt.Errorf("empty chunk")
	}

	// Check for [DONE] marker
	if bytes.Equal(bytes.TrimSpace(chunk), []byte("[DONE]")) {
		return "", io.EOF
	}

	// Parse the chunk
	var response struct {
		Choices []struct {
			Delta struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"delta"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(chunk, &response); err != nil {
		return "", fmt.Errorf("malformed response: %w", err)
	}

	if len(response.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	// Handle finish reason
	if response.Choices[0].FinishReason != "" {
		return "", io.EOF
	}

	// Skip role-only messages
	if response.Choices[0].Delta.Role != "" && response.Choices[0].Delta.Content == "" {
		return "", fmt.Errorf("skip token")
	}

	return response.Choices[0].Delta.Content, nil
}

// PrepareRequestWithMessages creates a request body using structured message objects
// rather than a flattened prompt string. This enables more efficient caching and
// better preserves conversation structure for the OpenAI API.
//
// Parameters:
//   - messages: Slice of MemoryMessage objects representing the conversation
//   - options: Additional options for the request
//
// Returns:
//   - Serialized JSON request body
//   - Any error encountered during preparation
func (p *OpenAIProvider) PrepareRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error) {
	request := map[string]interface{}{
		"model":    p.model,
		"messages": []map[string]interface{}{},
	}

	// Handle system prompt as system message
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["messages"] = append(request["messages"].([]map[string]interface{}), map[string]interface{}{
			"role":    "system",
			"content": systemPrompt,
		})
	}

	// Convert MemoryMessage objects to OpenAI messages format
	for _, msg := range messages {
		message := map[string]interface{}{
			"role":    msg.Role,
			"content": msg.Content,
		}

		// Add metadata if present
		if len(msg.Metadata) > 0 {
			for k, v := range msg.Metadata {
				message[k] = v
			}
		}

		request["messages"] = append(request["messages"].([]map[string]interface{}), message)
	}

	// Handle tools
	if tools, ok := options["tools"].([]utils.Tool); ok && len(tools) > 0 {
		// Filter out web_search tools - OpenAI doesn't support them
		openAITools := []map[string]interface{}{}
		for _, tool := range tools {
			// Skip web_search tools - OpenAI doesn't support them
			if tool.Type == "web_search" {
				p.logger.Debug("Skipping web_search tool - OpenAI doesn't support web_search")
				continue
			}

			// Handle regular function tools
			if !options["strict_tools"].(bool) {
				openAITools = append(openAITools, map[string]interface{}{
					"type": "function",
					"function": map[string]interface{}{
						"name":        tool.Function.Name,
						"description": tool.Function.Description,
						"parameters":  tool.Function.Parameters,
					},
				})
			} else {
				openAITools = append(openAITools, map[string]interface{}{
					"type": "function",
					"function": map[string]interface{}{
						"name":        tool.Function.Name,
						"description": tool.Function.Description,
						"parameters":  tool.Function.Parameters,
					},
					"strict": true, // Add this if you want strict mode
				})
			}
		}

		// Only add tools to request if we have any non-web_search tools
		if len(openAITools) > 0 {
			request["tools"] = openAITools

			// Handle tool_choice (only if model supports it and we have tools)
			if toolChoice, ok := options["tool_choice"].(string); ok && !p.needsNoToolChoice() {
				request["tool_choice"] = toolChoice
			}
		}
	}

	// Create a merged copy of options to handle token parameters properly
	mergedOptions := make(map[string]interface{})

	// First add options from provider (p.options)
	for k, v := range p.options {
		if k != "tools" && k != "tool_choice" && k != "system_prompt" && k != "structured_messages" {
			mergedOptions[k] = v
		}
	}

	// Then add options from the function parameters (may override provider options)
	for k, v := range options {
		if k != "tools" && k != "tool_choice" && k != "strict_tools" && k != "system_prompt" && k != "structured_messages" {
			mergedOptions[k] = v
		}
	}

	// Handle max_tokens/max_completion_tokens conflict
	// For models that need max_completion_tokens, ensure we use that and not max_tokens
	if p.needsMaxCompletionTokens() {
		if _, hasMaxTokens := mergedOptions["max_tokens"]; hasMaxTokens {
			// Move max_tokens value to max_completion_tokens
			mergedOptions["max_completion_tokens"] = mergedOptions["max_tokens"]
			delete(mergedOptions, "max_tokens")
		}
	} else {
		// For other models, ensure we use max_tokens and not max_completion_tokens
		if _, hasMaxCompletionTokens := mergedOptions["max_completion_tokens"]; hasMaxCompletionTokens {
			// Move max_completion_tokens value to max_tokens
			mergedOptions["max_tokens"] = mergedOptions["max_completion_tokens"]
			delete(mergedOptions, "max_completion_tokens")
		}
	}

	// Handle reasoning_effort option
	if !p.needsReasoningEffort() {
		delete(mergedOptions, "reasoning_effort")
	}

	// Remove temperature from mergedOptions if model doesn't support it
	if p.needsNoTemperature() {
		delete(mergedOptions, "temperature")
	}

	// Remove tool_choice from request if model doesn't support it
	if p.needsNoToolChoice() {
		delete(request, "tool_choice")
	}

	// Add merged options to the request
	for k, v := range mergedOptions {
		request[k] = v
	}

	return json.Marshal(request)
}
