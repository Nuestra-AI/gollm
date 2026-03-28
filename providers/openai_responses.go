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

// OpenAIResponsesProvider implements the Provider interface for OpenAI's
// Responses API (/v1/responses). It is registered as "openai-responses" and
// lives in its own file to avoid merge conflicts with upstream openai.go.
type OpenAIResponsesProvider struct {
	apiKey       string
	model        string
	extraHeaders map[string]string
	options      map[string]interface{}
	logger       utils.Logger
}

// NewOpenAIResponsesProvider creates a new OpenAI Responses API provider.
func NewOpenAIResponsesProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	if extraHeaders == nil {
		extraHeaders = make(map[string]string)
	}
	return &OpenAIResponsesProvider{
		apiKey:       apiKey,
		model:        model,
		extraHeaders: extraHeaders,
		options:      make(map[string]interface{}),
		logger:       utils.NewLogger(utils.LogLevelInfo),
	}
}

// ---------------------------------------------------------------------------
// Identity & config
// ---------------------------------------------------------------------------

func (p *OpenAIResponsesProvider) Name() string     { return "openai-responses" }
func (p *OpenAIResponsesProvider) Endpoint() string  { return "https://api.openai.com/v1/responses" }
func (p *OpenAIResponsesProvider) SupportsJSONSchema() bool { return true }
func (p *OpenAIResponsesProvider) SupportsStreaming() bool  { return true }

func (p *OpenAIResponsesProvider) Headers() map[string]string {
	headers := map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + p.apiKey,
	}
	for k, v := range p.extraHeaders {
		headers[k] = v
	}
	return headers
}

func (p *OpenAIResponsesProvider) SetExtraHeaders(extraHeaders map[string]string) {
	p.extraHeaders = extraHeaders
}

func (p *OpenAIResponsesProvider) SetLogger(logger utils.Logger) {
	p.logger = logger
}

func (p *OpenAIResponsesProvider) SetOption(key string, value interface{}) {
	// Normalize max_tokens → max_output_tokens for the Responses API
	if key == "max_tokens" || key == "max_completion_tokens" {
		delete(p.options, "max_tokens")
		delete(p.options, "max_completion_tokens")
		key = "max_output_tokens"
	}

	if key == "reasoning_effort" && !modelNeedsReasoningEffort(p.model) {
		delete(p.options, "reasoning_effort")
		return
	}

	if key == "temperature" && modelNeedsNoTemperature(p.model) {
		delete(p.options, "temperature")
		return
	}

	p.options[key] = value
}

func (p *OpenAIResponsesProvider) SetDefaultOptions(cfg *config.Config) {
	p.SetOption("temperature", cfg.Temperature)
	p.SetOption("max_tokens", cfg.MaxTokens)
	if cfg.Seed != nil {
		p.SetOption("seed", *cfg.Seed)
	}
}

// ---------------------------------------------------------------------------
// Request building helpers
// ---------------------------------------------------------------------------

// responsesExcludeKeys lists option keys that are handled specially and
// should not be blindly copied into the top-level request body.
var responsesExcludeKeys = []string{
	"tools", "tool_choice", "strict_tools", "system_prompt",
	"structured_messages", "images", "stream",
}

// buildTools converts utils.Tool slice into the Responses API tools format.
// Built-in tools (web_search, file_search, code_interpreter, computer_use, mcp)
// are passed through with their native shape. Function tools are wrapped.
func (p *OpenAIResponsesProvider) buildTools(tools []utils.Tool, strictTools bool) []map[string]interface{} {
	var result []map[string]interface{}
	for _, tool := range tools {
		switch tool.Type {
		case "web_search":
			t := map[string]interface{}{"type": "web_search"}
			if tool.SearchContextSize != "" {
				t["search_context_size"] = tool.SearchContextSize
			}
			if tool.UserLocation != nil {
				t["user_location"] = tool.UserLocation
			}
			result = append(result, t)

		case "file_search":
			t := map[string]interface{}{"type": "file_search"}
			if len(tool.VectorStoreIDs) > 0 {
				t["vector_store_ids"] = tool.VectorStoreIDs
			}
			if tool.Filters != nil {
				t["filters"] = tool.Filters
			}
			result = append(result, t)

		case "code_interpreter":
			t := map[string]interface{}{"type": "code_interpreter"}
			if tool.Container != nil {
				t["container"] = tool.Container
			}
			result = append(result, t)

		case "computer_use_preview": // OpenAI names this tool "computer_use_preview" (not "computer_use") while in preview
			t := map[string]interface{}{"type": "computer_use_preview"}
			if tool.DisplayWidth > 0 {
				t["display_width"] = tool.DisplayWidth
			}
			if tool.DisplayHeight > 0 {
				t["display_height"] = tool.DisplayHeight
			}
			if tool.Environment != "" {
				t["environment"] = tool.Environment
			}
			result = append(result, t)

		case "mcp":
			t := map[string]interface{}{"type": "mcp"}
			if tool.ServerLabel != "" {
				t["server_label"] = tool.ServerLabel
			}
			if tool.ServerURL != "" {
				t["server_url"] = tool.ServerURL
			}
			if len(tool.ServerHeaders) > 0 {
				t["headers"] = tool.ServerHeaders
			}
			result = append(result, t)

		default:
			// Function tool
			funcDef := map[string]interface{}{
				"type": "function",
				"name": tool.Function.Name,
			}
			if tool.Function.Description != "" {
				funcDef["description"] = tool.Function.Description
			}
			if tool.Function.Parameters != nil {
				funcDef["parameters"] = tool.Function.Parameters
			}
			if strictTools {
				funcDef["strict"] = true
			}
			result = append(result, funcDef)
		}
	}
	return result
}

// addToolsToRequest adds tools and tool_choice to the request map.
func (p *OpenAIResponsesProvider) addToolsToRequest(request map[string]interface{}, options map[string]interface{}) {
	tools, ok := options["tools"].([]utils.Tool)
	if !ok || len(tools) == 0 {
		return
	}

	strictTools, _ := options["strict_tools"].(bool)
	apiTools := p.buildTools(tools, strictTools)
	if len(apiTools) > 0 {
		request["tools"] = apiTools
		if toolChoice, ok := options["tool_choice"].(string); ok && !modelNeedsNoToolChoice(p.model) {
			request["tool_choice"] = toolChoice
		}
	}
}

// ---------------------------------------------------------------------------
// PrepareRequest — simple prompt → Responses API format
// ---------------------------------------------------------------------------

func (p *OpenAIResponsesProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	request := map[string]interface{}{
		"model": p.model,
	}

	// System prompt → instructions
	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["instructions"] = systemPrompt
	}

	// Check for images
	images, hasImages := options["images"].([]types.ContentPart)
	if hasImages && len(images) > 0 {
		content := []map[string]interface{}{
			{"type": "input_text", "text": prompt},
		}
		// ConvertImagesToOpenAIContent is defined in vision_helpers.go
		content = append(content, ConvertImagesToOpenAIContent(images)...)
		request["input"] = []map[string]interface{}{
			{"role": "user", "content": content},
		}
	} else {
		request["input"] = prompt
	}

	// Tools
	p.addToolsToRequest(request, options)

	// Merge options
	merged := mergeOpenAIResponsesOptions(p.model, p.options, options, responsesExcludeKeys)
	for k, v := range merged {
		request[k] = v
	}

	return json.Marshal(request)
}

// ---------------------------------------------------------------------------
// PrepareRequestWithSchema — prompt + JSON schema
// ---------------------------------------------------------------------------

func (p *OpenAIResponsesProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	schemaObj, err := normalizeSchema(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to normalize schema: %w", err)
	}

	cleanSchema := cleanSchemaForOpenAI(schemaObj) // defined in openai.go

	request := map[string]interface{}{
		"model": p.model,
		"input": prompt,
		"text": map[string]interface{}{
			"format": map[string]interface{}{
				"type":   "json_schema",
				"name":   "structured_response",
				"schema": cleanSchema,
				"strict": true,
			},
		},
	}

	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["instructions"] = systemPrompt
	}

	p.addToolsToRequest(request, options)

	merged := mergeOpenAIResponsesOptions(p.model, p.options, options, responsesExcludeKeys)
	for k, v := range merged {
		request[k] = v
	}

	return json.Marshal(request)
}

// ---------------------------------------------------------------------------
// PrepareRequestWithMessages — multi-turn conversation
// ---------------------------------------------------------------------------

func (p *OpenAIResponsesProvider) PrepareRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error) {
	request := map[string]interface{}{
		"model": p.model,
	}

	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["instructions"] = systemPrompt
	}

	input := p.convertMessagesToInput(messages)
	request["input"] = input

	p.addToolsToRequest(request, options)

	merged := mergeOpenAIResponsesOptions(p.model, p.options, options, responsesExcludeKeys)
	for k, v := range merged {
		request[k] = v
	}

	return json.Marshal(request)
}

// ---------------------------------------------------------------------------
// PrepareRequestWithMessagesAndSchema — multi-turn + schema
// ---------------------------------------------------------------------------

func (p *OpenAIResponsesProvider) PrepareRequestWithMessagesAndSchema(messages []types.MemoryMessage, options map[string]interface{}, schema interface{}) ([]byte, error) {
	schemaObj, err := normalizeSchema(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to normalize schema: %w", err)
	}

	cleanSchema := cleanSchemaForOpenAI(schemaObj) // defined in openai.go

	request := map[string]interface{}{
		"model": p.model,
		"text": map[string]interface{}{
			"format": map[string]interface{}{
				"type":   "json_schema",
				"name":   "structured_response",
				"schema": cleanSchema,
				"strict": true,
			},
		},
	}

	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["instructions"] = systemPrompt
	}

	input := p.convertMessagesToInput(messages)
	request["input"] = input

	p.addToolsToRequest(request, options)

	merged := mergeOpenAIResponsesOptions(p.model, p.options, options, responsesExcludeKeys)
	for k, v := range merged {
		request[k] = v
	}

	return json.Marshal(request)
}

// convertMessagesToInput converts MemoryMessage slice to Responses API input items.
func (p *OpenAIResponsesProvider) convertMessagesToInput(messages []types.MemoryMessage) []map[string]interface{} {
	var input []map[string]interface{}

	for _, msg := range messages {
		item := map[string]interface{}{
			"role": msg.Role,
		}

		// Tool result messages
		if msg.Role == "tool" && msg.ToolCallID != "" {
			item["type"] = "function_call_output"
			item["call_id"] = msg.ToolCallID
			item["output"] = msg.Content
			delete(item, "role")
			input = append(input, item)
			continue
		}

		// Assistant messages with tool calls
		if len(msg.ToolCalls) > 0 {
			// Each tool call becomes a separate function_call input item
			for _, tc := range msg.ToolCalls {
				callItem := map[string]interface{}{
					"type":      "function_call",
					"call_id":   tc.ID,
					"name":      tc.Function.Name,
					"arguments": string(tc.Function.Arguments),
				}
				input = append(input, callItem)
			}
			// If the assistant also had text content, add it as a message
			if msg.Content != "" {
				item["content"] = msg.Content
				input = append(input, item)
			}
			continue
		}

		// Multimodal content
		if msg.HasMultiContent() {
			// BuildOpenAIContentFromParts is defined in vision_helpers.go
			item["content"] = BuildOpenAIContentFromParts(msg.MultiContent)
		} else {
			item["content"] = msg.Content
		}

		input = append(input, item)
	}

	return input
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

func (p *OpenAIResponsesProvider) ParseResponse(body []byte) (string, error) {
	text, _, err := p.parseResponseInternal(body)
	return text, err
}

func (p *OpenAIResponsesProvider) ParseResponseWithUsage(body []byte) (string, *types.ResponseDetails, error) {
	return p.parseResponseInternal(body)
}

func (p *OpenAIResponsesProvider) parseResponseInternal(body []byte) (string, *types.ResponseDetails, error) {
	var response struct {
		ID     string `json:"id"`
		Status string `json:"status"`
		Model  string `json:"model"`
		Output []struct {
			Type   string `json:"type"`
			ID     string `json:"id,omitempty"`
			Status string `json:"status,omitempty"`
			Role   string `json:"role,omitempty"`

			// message content
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

			// function_call fields
			Name      string `json:"name,omitempty"`
			CallID    string `json:"call_id,omitempty"`
			Arguments string `json:"arguments,omitempty"`

			// web_search_call fields
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
		} `json:"output"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.Unmarshal(body, &response); err != nil {
		return "", nil, fmt.Errorf("error parsing response: %w", err)
	}

	details := &types.ResponseDetails{
		ID:    response.ID,
		Model: response.Model,
		TokenUsage: types.TokenUsage{
			PromptTokens:     response.Usage.InputTokens,
			CompletionTokens: response.Usage.OutputTokens,
			TotalTokens:      response.Usage.TotalTokens,
		},
	}

	// Collect text, function calls, and web search data
	var textContent strings.Builder
	var functionCalls []string
	var hasWebSearch bool
	var webSearchOutput []interface{}

	for _, item := range response.Output {
		switch item.Type {
		case "message":
			for _, c := range item.Content {
				if c.Type == "output_text" && c.Text != "" {
					textContent.WriteString(c.Text)
				}
			}

		case "function_call":
			var args interface{}
			if item.Arguments != "" {
				if err := json.Unmarshal([]byte(item.Arguments), &args); err != nil {
					args = item.Arguments
				}
			}
			fc, err := utils.FormatFunctionCall(item.Name, args)
			if err != nil {
				return "", nil, fmt.Errorf("error formatting function call: %w", err)
			}
			functionCalls = append(functionCalls, fc)

		case "web_search_call":
			hasWebSearch = true
			itemMap := map[string]interface{}{
				"type":   item.Type,
				"id":     item.ID,
				"status": item.Status,
			}
			if item.Action != nil {
				actionMap := map[string]interface{}{"type": item.Action.Type}
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
			webSearchOutput = append(webSearchOutput, itemMap)
		}
	}

	// If we had web search output, also add the message items for parseWebSearchResponse
	if hasWebSearch {
		for _, item := range response.Output {
			if item.Type == "message" {
				itemMap := map[string]interface{}{
					"type": "message",
					"role": item.Role,
				}
				if len(item.Content) > 0 {
					contentSlice := make([]interface{}, len(item.Content))
					for j, c := range item.Content {
						cm := map[string]interface{}{"type": c.Type}
						if c.Text != "" {
							cm["text"] = c.Text
						}
						if len(c.Annotations) > 0 {
							annSlice := make([]interface{}, len(c.Annotations))
							for k, ann := range c.Annotations {
								annSlice[k] = map[string]interface{}{
									"type":        ann.Type,
									"start_index": ann.StartIndex,
									"end_index":   ann.EndIndex,
									"url":         ann.URL,
									"title":       ann.Title,
								}
							}
							cm["annotations"] = annSlice
						}
						contentSlice[j] = cm
					}
					itemMap["content"] = contentSlice
				}
				webSearchOutput = append(webSearchOutput, itemMap)
			}
		}
		return parseResponsesWebSearchOutput(webSearchOutput, details)
	}

	if textContent.Len() > 0 {
		return textContent.String(), details, nil
	}

	if len(functionCalls) > 0 {
		return strings.Join(functionCalls, "\n"), details, nil
	}

	return "", details, fmt.Errorf("no content in response")
}

// parseResponsesWebSearchOutput handles parsing output[] items containing
// web_search_call and message items from the Responses API format.
// This is separate from OpenAIProvider.parseWebSearchResponse (which handles
// the same format within Chat Completions) to avoid touching openai.go.
func parseResponsesWebSearchOutput(output []interface{}, details *types.ResponseDetails) (string, *types.ResponseDetails, error) {
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
			call := types.WebSearchCall{
				Type: itemType,
			}
			if id, ok := itemMap["id"].(string); ok {
				call.ID = id
			}
			if status, ok := itemMap["status"].(string); ok {
				call.Status = status
			}

			if actionData, ok := itemMap["action"].(map[string]interface{}); ok {
				action := &types.WebSearchAction{}
				if t, ok := actionData["type"].(string); ok {
					action.Type = t
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
							source := types.Source{}
							if u, ok := srcMap["url"].(string); ok {
								source.URL = u
							}
							if t, ok := srcMap["title"].(string); ok {
								source.Title = t
							}
							if st, ok := srcMap["type"].(string); ok {
								source.Type = st
							}
							action.Sources = append(action.Sources, source)
						}
					}
				}
				call.Action = action
			}
			webSearchCalls = append(webSearchCalls, call)

		case "message":
			if contentArray, ok := itemMap["content"].([]interface{}); ok {
				for _, c := range contentArray {
					if contentItem, ok := c.(map[string]interface{}); ok {
						if text, ok := contentItem["text"].(string); ok {
							content.WriteString(text)
						}
						if annArray, ok := contentItem["annotations"].([]interface{}); ok {
							for _, a := range annArray {
								if annMap, ok := a.(map[string]interface{}); ok {
									ann := types.Annotation{}
									if t, ok := annMap["type"].(string); ok {
										ann.Type = t
									}
									ann.StartIndex = toInt(annMap["start_index"])
									ann.EndIndex = toInt(annMap["end_index"])
									if u, ok := annMap["url"].(string); ok {
										ann.URL = u
									}
									if t, ok := annMap["title"].(string); ok {
										ann.Title = t
									}
									annotations = append(annotations, ann)
									if ann.Type == "url_citation" {
										citations = append(citations, types.URLCitation(ann))
									}
								}
							}
						}
					}
				}
			}
		}
	}

	if details.Metadata == nil {
		details.Metadata = make(map[string]interface{})
	}
	details.Metadata["web_search_calls"] = webSearchCalls
	details.Metadata["annotations"] = annotations
	details.Metadata["citations"] = citations

	return content.String(), details, nil
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

func (p *OpenAIResponsesProvider) PrepareStreamRequest(prompt string, options map[string]interface{}) ([]byte, error) {
	// Build a normal request, then add stream: true
	request := map[string]interface{}{
		"model":  p.model,
		"input":  prompt,
		"stream": true,
	}

	if systemPrompt, ok := options["system_prompt"].(string); ok && systemPrompt != "" {
		request["instructions"] = systemPrompt
	}

	p.addToolsToRequest(request, options)

	merged := mergeOpenAIResponsesOptions(p.model, p.options, options, responsesExcludeKeys)
	for k, v := range merged {
		request[k] = v
	}

	return json.Marshal(request)
}

func (p *OpenAIResponsesProvider) ParseStreamResponse(chunk []byte) (string, error) {
	trimmed := bytes.TrimSpace(chunk)
	if len(trimmed) == 0 {
		return "", fmt.Errorf("empty chunk")
	}

	// Check for [DONE] marker
	if bytes.Equal(trimmed, []byte("[DONE]")) {
		return "", io.EOF
	}

	// The Responses API uses typed SSE events. The chunk we receive is the
	// JSON data payload. We look for response.output_text.delta events.
	var event struct {
		Type  string `json:"type"`
		Delta string `json:"delta"`
	}

	if err := json.Unmarshal(trimmed, &event); err != nil {
		return "", fmt.Errorf("malformed response: %w", err)
	}

	switch event.Type {
	case "response.output_text.delta":
		return event.Delta, nil
	case "response.completed", "response.done":
		return "", io.EOF
	case "":
		// Some events don't have a type at top level — try delta directly
		if event.Delta != "" {
			return event.Delta, nil
		}
		return "", fmt.Errorf("skip token")
	default:
		// Other event types (response.created, response.in_progress, etc.)
		return "", fmt.Errorf("skip token")
	}
}

// ---------------------------------------------------------------------------
// Function calling
// ---------------------------------------------------------------------------

func (p *OpenAIResponsesProvider) HandleFunctionCalls(body []byte) ([]byte, error) {
	response := string(body)
	functionCalls, err := utils.ExtractFunctionCalls(response)
	if err != nil {
		return nil, fmt.Errorf("error extracting function calls: %w", err)
	}
	if len(functionCalls) == 0 {
		return nil, fmt.Errorf("no function calls found in response")
	}
	return json.Marshal(functionCalls)
}
