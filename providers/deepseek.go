// Package providers implements LLM provider interfaces and implementations.
package providers

import (
	"encoding/json"
	"fmt"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/types"
)

// DeepSeekProvider implements the Provider interface for DeepSeek's API.
// It inherits from OpenAIProvider since DeepSeek uses an OpenAI-compatible API.
type DeepSeekProvider struct {
	OpenAIProvider
}

// NewDeepSeekProvider creates a new DeepSeek provider instance.
// It initializes the provider with the given API key, model, and optional headers.
//
// Parameters:
//   - apiKey: DeepSeek API key for authentication
//   - model: The model to use (e.g., "deepseek-chat")
//   - extraHeaders: Additional HTTP headers for requests
//
// Returns:
//   - A configured DeepSeek Provider instance
func NewDeepSeekProvider(apiKey, model string, extraHeaders map[string]string) Provider {
	provider := &DeepSeekProvider{
		OpenAIProvider: *NewOpenAIProvider(apiKey, model, extraHeaders).(*OpenAIProvider),
	}
	// DeepSeek uses the "system" role, not OpenAI's "developer" role.
	provider.systemRole = "system"
	provider.samplingParams = deepSeekSupportedParams
	provider.samplingScope = "deepseek"
	// Override the endpoint
	return provider
}

// PrepareRequestWithSchema asks for JSON matching schema. DeepSeek's response_format accepts only
// "text" or "json_object" (https://api-docs.deepseek.com/api/create-chat-completion, 2026-11-10), so
// the inherited json_schema request is not sent: the schema goes in the system prompt and the
// response is requested as json_object, which DeepSeek guarantees to be valid JSON but not to match
// the schema. gollm's validator enforces the schema on the response instead.
func (p *DeepSeekProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error) {
	opts, err := withSchemaInstruction(options, schema)
	if err != nil {
		return nil, err
	}
	return p.PrepareRequest(prompt, opts)
}

// PrepareRequestWithMessagesAndSchema is the conversation form of PrepareRequestWithSchema, with the
// same json_object request in place of json_schema.
func (p *DeepSeekProvider) PrepareRequestWithMessagesAndSchema(messages []types.MemoryMessage, options map[string]interface{}, schema interface{}) ([]byte, error) {
	opts, err := withSchemaInstruction(options, schema)
	if err != nil {
		return nil, err
	}
	return p.PrepareRequestWithMessages(messages, opts)
}

// withSchemaInstruction returns a copy of options that requests response_format json_object, with a
// system prompt ending in an instruction to answer in JSON matching schema. json_object mode also
// requires the word "JSON" in the prompt.
func withSchemaInstruction(options map[string]interface{}, schema interface{}) (map[string]interface{}, error) {
	schemaObj, err := normalizeSchema(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to normalize schema: %w", err)
	}
	schemaJSON, err := json.Marshal(schemaObj)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %w", err)
	}
	instruction := fmt.Sprintf("You must respond with a JSON object that strictly adheres to this schema:\n%s\nDo not include any explanatory text, only output valid JSON.", string(schemaJSON))

	opts := make(map[string]interface{}, len(options)+1)
	for k, v := range options {
		opts[k] = v
	}
	if sp, ok := options["system_prompt"].(string); ok && sp != "" {
		instruction = sp + "\n\n" + instruction
	}
	opts["system_prompt"] = instruction
	opts["response_format"] = map[string]interface{}{"type": "json_object"}
	return opts, nil
}

// Name returns "deepseek" as the provider identifier.
// This is used to identify the provider in the system.
func (p *DeepSeekProvider) Name() string {
	return "deepseek"
}

// Endpoint returns the DeepSeek API endpoint URL.
// This is the URL used to make requests to the DeepSeek API.
func (p *DeepSeekProvider) Endpoint() string {
	return "https://api.deepseek.com/chat/completions"
}

// SetDefaultOptions configures standard options from the global configuration.
//
// Forwards only what DeepSeek documents: temperature, max_tokens, top_p. No seed,
// and its penalties are deprecated and explicitly ignored, so they are not sent.
//
// Parameters:
//   - config: The global configuration containing options to set
func (p *DeepSeekProvider) SetDefaultOptions(config *config.Config) {
	applySamplingDefaults(p, config, deepSeekSamplingParams)
	p.logger.Debug("Default options set", "temperature", config.Temperature, "max_tokens", config.MaxTokens)
}
