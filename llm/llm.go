// Package llm provides a unified interface for interacting with various Language Learning Model providers.
// It abstracts away provider-specific implementations and provides a consistent API for text generation,
// prompt management, and error handling.
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/teilomillet/gollm/config"
	"github.com/teilomillet/gollm/providers"
	"github.com/teilomillet/gollm/types"
	"github.com/teilomillet/gollm/utils"
)

// LLM interface defines the methods that our internal language model should implement.
// It provides a unified way to interact with different LLM providers while abstracting
// away provider-specific details.
type LLM interface {
	// Generate produces text based on the given prompt and options.
	// Returns ErrorTypeRequest for request preparation failures,
	// ErrorTypeAPI for provider API errors, or ErrorTypeResponse for response processing issues.
	Generate(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (response string, err error)

	// GenerateWithSchema generates text that conforms to a specific JSON schema.
	// Returns ErrorTypeInvalidInput for schema validation failures,
	// or other error types as per Generate.
	GenerateWithSchema(ctx context.Context, prompt *Prompt, schema interface{}, opts ...GenerateOption) (string, error)

	// GenerateWithUsage produces text and returns response details including token usage.
	// Returns the generated text, response details, and any error encountered.
	GenerateWithUsage(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (string, *types.ResponseDetails, error)

	// GenerateWithSchemaAndUsage generates text conforming to a schema and returns response details.
	// Returns the generated text, response details, and any error encountered.
	GenerateWithSchemaAndUsage(ctx context.Context, prompt *Prompt, schema interface{}, opts ...GenerateOption) (string, *types.ResponseDetails, error)

	// Stream initiates a streaming response from the LLM.
	// Returns ErrorTypeUnsupported if the provider doesn't support streaming.
	Stream(ctx context.Context, prompt *Prompt, opts ...StreamOption) (TokenStream, error)

	// SupportsStreaming checks if the provider supports streaming responses.
	SupportsStreaming() bool

	// SetOption configures a provider-specific option.
	// Returns ErrorTypeInvalidInput if the option is not supported.
	SetOption(key string, value interface{})

	// SetLogLevel adjusts the logging verbosity.
	SetLogLevel(level utils.LogLevel)

	// SetEndpoint updates the API endpoint (primarily for local models).
	// Returns ErrorTypeProvider if the provider doesn't support endpoint configuration.
	SetEndpoint(endpoint string)

	// NewPrompt creates a new prompt instance.
	NewPrompt(input string) *Prompt

	// GetLogger returns the current logger instance.
	GetLogger() utils.Logger

	// SupportsJSONSchema checks if the provider supports JSON schema validation.
	SupportsJSONSchema() bool
}

// LLMImpl implements the LLM interface and manages interactions with specific providers.
// It handles provider communication, error management, and logging.
type LLMImpl struct {
	Provider     providers.Provider     // The underlying LLM provider
	Options      map[string]interface{} // Provider-specific options
	optionsMutex sync.RWMutex           // Mutex to protect concurrent access to Options map
	client       *http.Client           // HTTP client for API requests
	logger       utils.Logger           // Logger for debugging and monitoring
	config       *config.Config         // Configuration settings
	MaxRetries   int                    // Maximum number of retry attempts
	RetryDelay   time.Duration          // Delay between retry attempts

	// usageObserver, if set, is fired once per billed provider round-trip — before schema validation
	// or the retry decision — so callers can record token usage for every attempt, not just the one
	// that ultimately succeeds. Guarded by usageMutex: it is expected to be set once at setup, but
	// generation reads it from whatever goroutines the caller uses, so the access must be safe.
	usageObserver UsageObserver
	usageMutex    sync.RWMutex
}

// GenerateOption is a function type for configuring generation behavior.
type GenerateOption func(*GenerateConfig)

// GenerateConfig holds configuration options for text generation.
type GenerateConfig struct {
	UseJSONSchema bool // Whether to use JSON schema validation
}

// NewLLM creates a new LLM instance with the specified configuration.
// It initializes the appropriate provider and sets up logging and HTTP clients.
//
// Returns:
//   - Configured LLM instance
//   - ErrorTypeProvider if provider initialization fails
//   - ErrorTypeAuthentication if API key validation fails
func NewLLM(cfg *config.Config, logger utils.Logger, registry *providers.ProviderRegistry) (LLM, error) {
	extraHeaders := make(map[string]string)
	if cfg.Provider == "anthropic" && cfg.EnableCaching {
		extraHeaders["anthropic-beta"] = "prompt-caching-2024-07-31"
	}

	// Check if API key is empty (skip for local providers that don't require auth)
	apiKey := cfg.APIKeys[cfg.Provider]
	isLocalProvider := cfg.Provider == "ollama" || cfg.Provider == "lmstudio" || cfg.Provider == "vllm"
	if apiKey == "" && !isLocalProvider {
		return nil, NewLLMError(ErrorTypeAuthentication, "empty API key", nil)
	}

	provider, err := registry.Get(cfg.Provider, apiKey, cfg.Model, extraHeaders)

	if err != nil {
		return nil, err
	}

	provider.SetDefaultOptions(cfg)

	// A caller-supplied client is used verbatim (its own Timeout applies), so a custom
	// RoundTripper can observe every provider request — including response headers, which body
	// parsing cannot see.
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: cfg.Timeout}
	}

	llmClient := &LLMImpl{
		Provider:   provider,
		client:     httpClient,
		logger:     logger,
		config:     cfg,
		MaxRetries: cfg.MaxRetries,
		RetryDelay: cfg.RetryDelay,
		Options:    make(map[string]interface{}),
		// Carried from the config so accounting reaches clients the caller never holds — the
		// per-model and aggregator clients inside MOA, and the per-case clients in assess.
		usageObserver: cfg.UsageObserver,
	}

	return llmClient, nil
}

// SetOption sets a provider-specific option with the given key and value.
// The option is logged at debug level for troubleshooting.
func (l *LLMImpl) SetOption(key string, value interface{}) {
	l.optionsMutex.Lock()
	defer l.optionsMutex.Unlock()

	l.Options[key] = value
	l.logger.Debug("Option set", key, value)
}

// SetUsageObserver registers a hook fired once per billed provider round-trip — before schema
// validation or the retry decision — so a caller can record token usage for every attempt, including
// schema-rejected, unparseable, and retried ones, not just the attempt that ultimately succeeds.
// See UsageObserver for the contract and UsageOutcome for the outcomes reported.
//
// Passing nil removes a previously registered observer. The result is always true: this is the leaf
// implementation, so the observer is genuinely installed. Wrappers forward their inner result
// instead — see UsageObservable.
func (l *LLMImpl) SetUsageObserver(observer UsageObserver) bool {
	l.usageMutex.Lock()
	defer l.usageMutex.Unlock()
	l.usageObserver = observer
	return true
}

// usageObserverFn returns the registered observer, or nil.
func (l *LLMImpl) usageObserverFn() UsageObserver {
	l.usageMutex.RLock()
	defer l.usageMutex.RUnlock()
	return l.usageObserver
}

// reportUsage fires the usage observer for one billed provider round-trip.
//
// Usage is taken from the provider's parsed details when it supplied them, and otherwise recovered
// from the raw response body — that fallback is what makes the unparseable-response and
// no-usage-reported cases recordable instead of silent. The event is delivered even when no usage
// could be determined at all, because an unaccountable billed call is itself worth recording.
func (l *LLMImpl) reportUsage(ctx context.Context, attempt int, outcome UsageOutcome, details *types.ResponseDetails, body []byte) {
	observer := l.usageObserverFn()
	if observer == nil {
		return
	}

	usage := types.TokenUsage{}
	model := ""
	tier := ""
	if details != nil {
		usage = details.TokenUsage
		model = details.Model
		tier = details.ServiceTier
	}
	// Fall back to the raw body only when the provider's parser gave us nothing — one pass,
	// recovering the counts and the tier together. The tier multiplies the price of whatever
	// the counts turn out to be, so a parse failure needs both; a successful parse needs
	// neither, and re-reading the body there would put a second full decode of every response
	// on the generation path for the providers that never report a tier at all.
	if usage.IsZero() && len(body) > 0 {
		recovered, recoveredTier, _ := ExtractUsageAndTier(body)
		if !recovered.IsZero() {
			usage = recovered
		}
		if tier == "" {
			tier = recoveredTier
		}
	}
	if model == "" && l.config != nil {
		model = l.config.Model
	}

	l.deliverUsage(ctx, observer, UsageEvent{
		Provider:    l.Provider.Name(),
		Model:       model,
		Outcome:     outcome,
		Attempt:     attempt,
		Usage:       usage,
		ServiceTier: tier,
		Details:     details,
	})
}

// logUsage emits the debug lines describing a round-trip's token usage. Tolerates nil details, which
// is what providers that report no usage of their own return.
func (l *LLMImpl) logUsage(details *types.ResponseDetails) {
	if details == nil {
		return
	}
	u := details.TokenUsage
	l.logger.Debug("Token usage", "prompt_tokens", u.PromptTokens, "completion_tokens", u.CompletionTokens, "total_tokens", u.TotalTokens)
	if u.CacheReadInputTokens > 0 || u.CacheCreationInputTokens > 0 || u.CachedPromptTokens > 0 {
		l.logger.Debug("Cache usage", "cache_creation_tokens", u.CacheCreationInputTokens, "cache_read_tokens", u.CacheReadInputTokens, "cached_prompt_tokens", u.CachedPromptTokens)
	}
	if u.ReasoningTokens > 0 {
		l.logger.Debug("Reasoning usage", "reasoning_tokens", u.ReasoningTokens)
	}
	if details.ID != "" {
		l.logger.Debug("Response ID", "id", details.ID)
	}
}

// deliverUsage invokes the observer, containing a panicking recorder so it cannot take down the
// generation it was only meant to measure.
func (l *LLMImpl) deliverUsage(ctx context.Context, observer UsageObserver, event UsageEvent) {
	defer func() {
		if r := recover(); r != nil {
			l.logger.Error("Usage observer panicked", "panic", r, "provider", event.Provider, "outcome", string(event.Outcome))
		}
	}()
	observer(ctx, event)
}

// SetEndpoint updates the API endpoint for the provider.
// This is primarily used for local models like Ollama.
func (l *LLMImpl) SetEndpoint(endpoint string) {
	// This is a no-op for non-Ollama providers
	l.logger.Debug("SetEndpoint called on non-Ollama provider", "endpoint", endpoint)
}

// SetLogLevel updates the logging verbosity level.
func (l *LLMImpl) SetLogLevel(level utils.LogLevel) {
	l.logger.Debug("Setting internal LLM log level", "new_level", level)
	l.logger.SetLevel(level)
}

// GetLogger returns the current logger instance.
func (l *LLMImpl) GetLogger() utils.Logger {
	return l.logger
}

// NewPrompt creates a new prompt instance with the given input text.
func (l *LLMImpl) NewPrompt(prompt string) *Prompt {
	return &Prompt{Input: prompt}
}

// SupportsJSONSchema checks if the current provider supports JSON schema validation.
func (l *LLMImpl) SupportsJSONSchema() bool {
	return l.Provider.SupportsJSONSchema()
}

// Generate produces text based on the given prompt and options.
// It handles retries, logging, and error management.
//
// Returns:
//   - Generated text response
//   - ErrorTypeRequest for request preparation failures
//   - ErrorTypeAPI for provider API errors
//   - ErrorTypeResponse for response processing issues
//   - ErrorTypeRateLimit if provider rate limit is exceeded
func (l *LLMImpl) Generate(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (string, error) {
	config := &GenerateConfig{}
	for _, opt := range opts {
		opt(config)
	}
	// Set the system prompt in the LLM's options
	if prompt.SystemPrompt != "" {
		l.SetOption("system_prompt", prompt.SystemPrompt)
	}
	for attempt := 0; attempt <= l.MaxRetries; attempt++ {
		l.logger.Debug("Generating text", "provider", l.Provider.Name(), "prompt", prompt.String(), "system_prompt", prompt.SystemPrompt, "attempt", attempt+1)
		// Pass the entire Prompt struct to attemptGenerate
		result, err := l.attemptGenerate(ctx, prompt, attempt)
		if err == nil {
			return result, nil
		}
		l.logger.Warn("Generation attempt failed", "error", err, "attempt", attempt+1)
		if attempt < l.MaxRetries {
			l.logger.Debug("Retrying", "delay", l.RetryDelay)
			if err := l.wait(ctx); err != nil {
				return "", err
			}
		}
	}
	return "", fmt.Errorf("failed to generate after %d attempts", l.MaxRetries+1)
}

// wait implements a cancellable delay between retry attempts.
// Returns context.Canceled if the context is cancelled during the wait.
func (l *LLMImpl) wait(ctx context.Context) error {
	return l.waitFor(ctx, l.RetryDelay)
}

// waitFor implements a cancellable delay of the given duration.
// Returns the context error if the context is cancelled during the wait.
func (l *LLMImpl) waitFor(ctx context.Context, d time.Duration) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(d):
		return nil
	}
}

// attemptGenerate makes a single attempt to generate text using the provider.
// It handles request preparation, API communication, and response processing.
//
// Returns:
//   - Generated text response
//   - ErrorTypeRequest for request preparation failures
//   - ErrorTypeAPI for provider API errors
//   - ErrorTypeResponse for response processing issues
//   - ErrorTypeRateLimit if provider rate limit is exceeded
//
// attempt is the zero-based retry index, reported to the usage observer so a recorder can
// distinguish a first-try success from the tokens burned on a third paid attempt.
func (l *LLMImpl) attemptGenerate(ctx context.Context, prompt *Prompt, attempt int) (string, error) {
	// Create a new options map that includes both l.Options and prompt-specific options
	options := make(map[string]interface{})

	// Safely read from the Options map
	l.optionsMutex.RLock()
	for k, v := range l.Options {
		options[k] = v
	}
	l.optionsMutex.RUnlock()

	// Add Tools and ToolChoice to options
	if len(prompt.Tools) > 0 {
		options["tools"] = prompt.Tools
	}
	if tc, ok := toolChoiceValue(prompt.ToolChoice); ok {
		options["tool_choice"] = tc
	}

	// Add Images to options for vision-capable models
	if prompt.HasImages() {
		options["images"] = prompt.Images
	}

	var reqBody []byte
	var err error

	// Check if we have structured messages from options (memory system)
	l.optionsMutex.RLock()
	structuredMessages, hasStructuredMessages := l.Options["structured_messages"]
	l.optionsMutex.RUnlock()

	if hasStructuredMessages {
		messages, ok := structuredMessages.([]types.MemoryMessage)
		if ok {
			l.logger.Debug("Using structured messages API", "message_count", len(messages))
			reqBody, err = l.Provider.PrepareRequestWithMessages(messages, options)
		} else {
			l.logger.Warn("Invalid structured_messages format", "type", fmt.Sprintf("%T", structuredMessages))
			reqBody, err = l.Provider.PrepareRequest(prompt.String(), options)
		}
	} else if prompt.hasStructuredMessages() {
		messages := promptMessagesToMemoryMessages(prompt.Messages)
		l.logger.Debug("Using prompt structured messages", "message_count", len(messages))
		reqBody, err = l.Provider.PrepareRequestWithMessages(messages, options)
	} else {
		// Standard request preparation
		reqBody, err = l.Provider.PrepareRequest(prompt.String(), options)
	}

	if err != nil {
		return "", NewLLMError(ErrorTypeRequest, "failed to prepare request", err)
	}

	l.logger.Debug("Full request body", "body", string(reqBody))
	req, err := http.NewRequestWithContext(ctx, "POST", l.Provider.Endpoint(), bytes.NewReader(reqBody))
	if err != nil {
		return "", NewLLMError(ErrorTypeRequest, "failed to create request", err)
	}

	headers := l.Provider.Headers()
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	l.logger.Debug("Request headers", "provider", l.Provider.Name(), "headers", utils.RedactHeaders(headers))

	l.logger.Wire("Full API request", "method", req.Method, "url", req.URL.String(), "headers", utils.RedactHTTPHeaders(req.Header), "body", string(reqBody))
	resp, err := l.client.Do(req)
	if err != nil {
		return "", NewLLMError(ErrorTypeRequest, "failed to send request", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", NewLLMError(ErrorTypeResponse, "failed to read response body", err)
	}

	// Log the full API response
	l.logger.Wire("Full API response", "body", string(body))

	if resp.StatusCode != http.StatusOK {
		l.logger.Warn("API error", "provider", l.Provider.Name(), slog.Int("status", resp.StatusCode), "body", string(body))
		return "", NewLLMError(classifyHTTPStatus(resp.StatusCode), fmt.Sprintf("API error: status code %d: %s", resp.StatusCode, truncateBytes(body, 500)), nil)
	}

	// Parse through the usage-bearing path even though this entrypoint discards the details: the
	// round-trip is billed either way, and Generate is the busiest entrypoint in the library, so
	// skipping the accounting here would leave most spend unrecorded. Every provider's
	// ParseResponseWithUsage returns the same text as ParseResponse.
	result, details, err := l.Provider.ParseResponseWithUsage(body)
	if err != nil {
		// A billed 200 whose content wouldn't parse — recover usage from the raw body.
		l.reportUsage(ctx, attempt, UsageOutcomeParseFail, nil, body)
		return "", NewLLMError(ErrorTypeResponse, "failed to parse response", err)
	}
	l.logUsage(details)
	l.reportUsage(ctx, attempt, UsageOutcomeSuccess, details, body)

	l.logger.Debug("Text generated successfully", "result", result)
	return result, nil
}

// GenerateWithSchema generates text that conforms to a specific JSON schema.
// It handles retries, logging, and error management.
//
// Returns:
//   - Generated text response
//   - ErrorTypeInvalidInput for schema validation failures
//   - Other error types as per Generate
func (l *LLMImpl) GenerateWithSchema(ctx context.Context, prompt *Prompt, schema interface{}, opts ...GenerateOption) (string, error) {
	config := &GenerateConfig{}
	for _, opt := range opts {
		opt(config)
	}

	var result string
	var lastErr error

	for attempt := 0; attempt <= l.MaxRetries; attempt++ {
		l.logger.Debug("Generating text with schema", "provider", l.Provider.Name(), "prompt", prompt.String(), "attempt", attempt+1)

		result, _, lastErr = l.attemptGenerateWithSchema(ctx, prompt, schema, attempt)
		if lastErr == nil {
			return result, nil
		}

		l.logger.Warn("Generation attempt with schema failed", "error", lastErr, "attempt", attempt+1)

		if attempt < l.MaxRetries {
			l.logger.Debug("Retrying", "delay", l.RetryDelay)
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(l.RetryDelay):
				// Continue to next attempt
			}
		}
	}

	return "", fmt.Errorf("failed to generate with schema after %d attempts: %w", l.MaxRetries+1, lastErr)
}

// GenerateWithUsage produces text based on the given prompt and returns token usage information.
// It handles retries, logging, and error management similar to Generate, but also extracts usage data.
//
// The returned details describe only the attempt that succeeded. When a retry occurs, earlier
// attempts were billed too and are not reflected here — register a UsageObserver via
// SetUsageObserver to account for every attempt.
//
// Returns:
//   - Generated text response
//   - Response details (or nil if not available)
//   - ErrorTypeRequest for request preparation failures
//   - ErrorTypeAPI for provider API errors
//   - ErrorTypeResponse for response processing issues
//   - ErrorTypeRateLimit if provider rate limit is exceeded
func (l *LLMImpl) GenerateWithUsage(ctx context.Context, prompt *Prompt, opts ...GenerateOption) (string, *types.ResponseDetails, error) {
	config := &GenerateConfig{}
	for _, opt := range opts {
		opt(config)
	}

	var result string
	var details *types.ResponseDetails
	var lastErr error

	for attempt := 0; attempt <= l.MaxRetries; attempt++ {
		l.logger.Debug("Generating text with usage tracking", "provider", l.Provider.Name(), "prompt", prompt.String(), "attempt", attempt+1)

		result, details, lastErr = l.attemptGenerateWithUsage(ctx, prompt, attempt)
		if lastErr == nil {
			return result, details, nil
		}

		l.logger.Warn("Generation attempt failed", "error", lastErr, "attempt", attempt+1)
		if attempt < l.MaxRetries {
			if err := l.wait(ctx); err != nil {
				return "", nil, err
			}
		}
	}

	return "", nil, fmt.Errorf("failed to generate after %d attempts: %w", l.MaxRetries+1, lastErr)
}

// GenerateWithSchemaAndUsage generates text conforming to a schema and returns response details.
// This combines schema validation with usage tracking.
//
// As with GenerateWithUsage, the returned details cover only the successful attempt. Schema
// rejections are billed and retried; register a UsageObserver to see them.
//
// Returns:
//   - Generated text response
//   - Response details (or nil if not available)
//   - ErrorTypeInvalidInput for schema validation failures
//   - Other error types as per GenerateWithUsage
func (l *LLMImpl) GenerateWithSchemaAndUsage(ctx context.Context, prompt *Prompt, schema interface{}, opts ...GenerateOption) (string, *types.ResponseDetails, error) {
	config := &GenerateConfig{}
	for _, opt := range opts {
		opt(config)
	}

	var result string
	var details *types.ResponseDetails
	var lastErr error

	for attempt := 0; attempt <= l.MaxRetries; attempt++ {
		l.logger.Debug("Generating text with schema and usage tracking", "provider", l.Provider.Name(), "prompt", prompt.String(), "attempt", attempt+1)

		result, details, _, lastErr = l.attemptGenerateWithSchemaAndUsage(ctx, prompt, schema, attempt)
		if lastErr == nil {
			return result, details, nil
		}

		l.logger.Warn("Generation attempt with schema failed", "error", lastErr, "attempt", attempt+1)
		if attempt < l.MaxRetries {
			if err := l.wait(ctx); err != nil {
				return "", nil, err
			}
		}
	}

	return "", nil, fmt.Errorf("failed to generate with schema after %d attempts: %w", l.MaxRetries+1, lastErr)
}

// attemptGenerateWithUsage makes a single attempt to generate text and track usage.
// It's similar to attemptGenerate but extracts response details from the response.
//
// Returns:
//   - Generated text response
//   - Response details (or nil if not available)
//   - Any error encountered during the attempt
//
// attempt is the zero-based retry index, reported to the usage observer.
func (l *LLMImpl) attemptGenerateWithUsage(ctx context.Context, prompt *Prompt, attempt int) (string, *types.ResponseDetails, error) {
	// Create a new options map that includes both l.Options and prompt-specific options
	options := make(map[string]interface{})

	// Safely read from the Options map
	l.optionsMutex.RLock()
	for k, v := range l.Options {
		options[k] = v
	}
	l.optionsMutex.RUnlock()

	// Set system prompt from the prompt object into local options (not l.Options)
	if prompt.SystemPrompt != "" {
		options["system_prompt"] = prompt.SystemPrompt
	}

	// Add Tools and ToolChoice to options
	if len(prompt.Tools) > 0 {
		options["tools"] = prompt.Tools
	}
	if tc, ok := toolChoiceValue(prompt.ToolChoice); ok {
		options["tool_choice"] = tc
	}

	var reqBody []byte
	var err error

	// Check if we have structured messages
	l.optionsMutex.RLock()
	structuredMessages, hasStructuredMessages := l.Options["structured_messages"]
	l.optionsMutex.RUnlock()

	if hasStructuredMessages {
		messages, ok := structuredMessages.([]types.MemoryMessage)
		if ok {
			l.logger.Debug("Using structured messages API", "message_count", len(messages))
			reqBody, err = l.Provider.PrepareRequestWithMessages(messages, options)
		} else {
			l.logger.Warn("Invalid structured_messages format", "type", fmt.Sprintf("%T", structuredMessages))
			reqBody, err = l.Provider.PrepareRequest(prompt.String(), options)
		}
	} else if prompt.hasStructuredMessages() {
		messages := promptMessagesToMemoryMessages(prompt.Messages)
		l.logger.Debug("Using prompt structured messages", "message_count", len(messages))
		reqBody, err = l.Provider.PrepareRequestWithMessages(messages, options)
	} else {
		// Standard request preparation
		reqBody, err = l.Provider.PrepareRequest(prompt.String(), options)
	}

	if err != nil {
		return "", nil, NewLLMError(ErrorTypeRequest, "failed to prepare request", err)
	}

	l.logger.Debug("Full request body", "body", string(reqBody))
	req, err := http.NewRequestWithContext(ctx, "POST", l.Provider.Endpoint(), bytes.NewReader(reqBody))
	if err != nil {
		return "", nil, NewLLMError(ErrorTypeRequest, "failed to create request", err)
	}

	for k, v := range l.Provider.Headers() {
		req.Header.Set(k, v)
	}

	l.logger.Wire("Full API request", "method", req.Method, "url", req.URL.String(), "headers", utils.RedactHTTPHeaders(req.Header), "body", string(reqBody))
	resp, err := l.client.Do(req)
	if err != nil {
		return "", nil, NewLLMError(ErrorTypeRequest, "failed to send request", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", nil, NewLLMError(ErrorTypeResponse, "failed to read response body", err)
	}

	l.logger.Wire("Full API response", "body", string(body))

	if resp.StatusCode != http.StatusOK {
		l.logger.Warn("API error", "provider", l.Provider.Name(), slog.Int("status", resp.StatusCode), "body", string(body))
		return "", nil, NewLLMError(classifyHTTPStatus(resp.StatusCode), fmt.Sprintf("API error: status code %d: %s", resp.StatusCode, truncateBytes(body, 500)), nil)
	}

	// Try to use ParseResponseWithUsage if available
	result, details, err := l.Provider.ParseResponseWithUsage(body)
	if err != nil {
		// The 200 was billed even though its content wouldn't parse (a max_tokens-truncated
		// response with no content blocks is the common case). Recover usage from the raw body
		// so the tokens are recorded before the retry loop pays for another attempt.
		l.reportUsage(ctx, attempt, UsageOutcomeParseFail, nil, body)
		return "", nil, NewLLMError(ErrorTypeResponse, "failed to parse response", err)
	}

	l.logUsage(details)
	l.reportUsage(ctx, attempt, UsageOutcomeSuccess, details, body)
	l.logger.Debug("Text generated successfully", "result", result)
	return result, details, nil
}

// prepareSchemaRequestBody builds the options map from the prompt and prepares
// the request body for schema-based generation. It centralises the logic shared
// by attemptGenerateWithSchema and attemptGenerateWithSchemaAndUsage.
func (l *LLMImpl) prepareSchemaRequestBody(prompt *Prompt, schema interface{}) (reqBody []byte, fullPrompt string, err error) {
	l.optionsMutex.RLock()
	options := make(map[string]interface{})
	for k, v := range l.Options {
		options[k] = v
	}
	l.optionsMutex.RUnlock()

	// Set system prompt from the prompt object into local options (not l.Options)
	if prompt.SystemPrompt != "" {
		options["system_prompt"] = prompt.SystemPrompt
	}

	// Add Tools and ToolChoice to options
	if len(prompt.Tools) > 0 {
		options["tools"] = prompt.Tools
	}
	if tc, ok := toolChoiceValue(prompt.ToolChoice); ok {
		options["tool_choice"] = tc
	}

	// Add Images to options for vision-capable models
	if prompt.HasImages() {
		options["images"] = prompt.Images
	}

	if prompt.hasStructuredMessages() {
		messages := promptMessagesToMemoryMessages(prompt.Messages)
		l.logger.Debug("Using prompt structured messages with schema", "message_count", len(messages))
		reqBody, err = l.Provider.PrepareRequestWithMessagesAndSchema(messages, options, schema)
		fullPrompt = prompt.String()
	} else if l.SupportsJSONSchema() {
		reqBody, err = l.Provider.PrepareRequestWithSchema(prompt.String(), options, schema)
		fullPrompt = prompt.String()
	} else {
		fullPrompt = l.preparePromptWithSchema(prompt.String(), schema)
		reqBody, err = l.Provider.PrepareRequest(fullPrompt, options)
	}

	return reqBody, fullPrompt, err
}

// attemptGenerateWithSchemaAndUsage makes a single attempt to generate text with schema validation and response details.
// It combines schema validation with response details extraction.
//
// Returns:
//   - Generated text response
//   - Response details (or nil if not available)
//   - Full prompt used for generation
//   - Any error encountered during the attempt
//
// attempt is the zero-based retry index, reported to the usage observer.
func (l *LLMImpl) attemptGenerateWithSchemaAndUsage(ctx context.Context, prompt *Prompt, schema interface{}, attempt int) (string, *types.ResponseDetails, string, error) {
	reqBody, fullPrompt, err := l.prepareSchemaRequestBody(prompt, schema)
	if err != nil {
		return "", nil, fullPrompt, NewLLMError(ErrorTypeRequest, "failed to prepare request", err)
	}

	l.logger.Debug("Request body", "provider", l.Provider.Name(), "body", string(reqBody))

	req, err := http.NewRequestWithContext(ctx, "POST", l.Provider.Endpoint(), bytes.NewReader(reqBody))
	if err != nil {
		return "", nil, fullPrompt, NewLLMError(ErrorTypeRequest, "failed to create request", err)
	}

	for k, v := range l.Provider.Headers() {
		req.Header.Set(k, v)
	}

	l.logger.Wire("Full API request", "method", req.Method, "url", req.URL.String(), "headers", utils.RedactHTTPHeaders(req.Header), "body", string(reqBody))
	resp, err := l.client.Do(req)
	if err != nil {
		return "", nil, fullPrompt, NewLLMError(ErrorTypeRequest, "failed to send request", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", nil, fullPrompt, NewLLMError(ErrorTypeResponse, "failed to read response body", err)
	}

	l.logger.Wire("Full API response", "body", string(body))

	if resp.StatusCode != http.StatusOK {
		l.logger.Warn("API error", "provider", l.Provider.Name(), slog.Int("status", resp.StatusCode), "body", string(body))
		return "", nil, fullPrompt, NewLLMError(classifyHTTPStatus(resp.StatusCode), fmt.Sprintf("API error: status code %d: %s", resp.StatusCode, truncateBytes(body, 500)), nil)
	}

	// Try to use ParseResponseWithUsage
	result, details, err := l.Provider.ParseResponseWithUsage(body)
	if err != nil {
		l.reportUsage(ctx, attempt, UsageOutcomeParseFail, nil, body)
		return "", nil, fullPrompt, NewLLMError(ErrorTypeResponse, "failed to parse response", err)
	}

	// Validate the result against the schema
	if err := ValidateAgainstSchema(result, schema); err != nil {
		// The response was billed (usage parsed above) even though it failed schema validation;
		// report it before discarding so recording isn't coupled to the success path — this is the
		// attempt whose tokens the retry loop would otherwise throw away.
		l.reportUsage(ctx, attempt, UsageOutcomeSchemaFail, details, body)
		return "", nil, fullPrompt, NewLLMError(ErrorTypeResponse, "response does not match schema", err)
	}

	l.logUsage(details)
	l.reportUsage(ctx, attempt, UsageOutcomeSuccess, details, body)
	l.logger.Debug("Text generated successfully", "result", result)
	return result, details, fullPrompt, nil
}

// attemptGenerateWithSchema makes a single attempt to generate text using the provider and a JSON schema.
// It handles request preparation, API communication, and response processing.
//
// Returns:
//   - Generated text response
//   - Full prompt used for generation
//   - ErrorTypeInvalidInput for schema validation failures
//   - Other error types as per attemptGenerate
//
// attempt is the zero-based retry index, reported to the usage observer.
func (l *LLMImpl) attemptGenerateWithSchema(ctx context.Context, prompt *Prompt, schema interface{}, attempt int) (string, string, error) {
	reqBody, fullPrompt, err := l.prepareSchemaRequestBody(prompt, schema)
	if err != nil {
		return "", fullPrompt, NewLLMError(ErrorTypeRequest, "failed to prepare request", err)
	}

	l.logger.Debug("Request body", "provider", l.Provider.Name(), "body", string(reqBody))

	req, err := http.NewRequestWithContext(ctx, "POST", l.Provider.Endpoint(), bytes.NewReader(reqBody))
	if err != nil {
		return "", fullPrompt, NewLLMError(ErrorTypeRequest, "failed to create request", err)
	}

	for k, v := range l.Provider.Headers() {
		req.Header.Set(k, v)
	}

	l.logger.Wire("Full API request", "method", req.Method, "url", req.URL.String(), "headers", utils.RedactHTTPHeaders(req.Header), "body", string(reqBody))
	resp, err := l.client.Do(req)
	if err != nil {
		return "", fullPrompt, NewLLMError(ErrorTypeRequest, "failed to send request", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fullPrompt, NewLLMError(ErrorTypeResponse, "failed to read response body", err)
	}

	// Log the full API response
	l.logger.Wire("Full API response", "body", string(body))

	if resp.StatusCode != http.StatusOK {
		l.logger.Warn("API error", "provider", l.Provider.Name(), slog.Int("status", resp.StatusCode), "body", string(body))
		return "", fullPrompt, NewLLMError(classifyHTTPStatus(resp.StatusCode), fmt.Sprintf("API error: status code %d: %s", resp.StatusCode, truncateBytes(body, 500)), nil)
	}

	// Parse through the usage-bearing path even though this entrypoint discards the details, so
	// schema-rejected and unparseable attempts are accounted for here exactly as they are in
	// attemptGenerateWithSchemaAndUsage. Both are billed and both are retried.
	result, details, err := l.Provider.ParseResponseWithUsage(body)
	if err != nil {
		l.reportUsage(ctx, attempt, UsageOutcomeParseFail, nil, body)
		return "", fullPrompt, NewLLMError(ErrorTypeResponse, "failed to parse response", err)
	}

	// Validate the result against the schema
	if err := ValidateAgainstSchema(result, schema); err != nil {
		l.reportUsage(ctx, attempt, UsageOutcomeSchemaFail, details, body)
		return "", fullPrompt, NewLLMError(ErrorTypeResponse, "response does not match schema", err)
	}

	l.logUsage(details)
	l.reportUsage(ctx, attempt, UsageOutcomeSuccess, details, body)
	l.logger.Debug("Text generated successfully", "result", result)
	return result, fullPrompt, nil
}

// preparePromptWithSchema prepares a prompt with a JSON schema for providers that do not support JSON schema validation.
// Returns the original prompt if schema marshaling fails (with a warning log).
func (l *LLMImpl) preparePromptWithSchema(prompt string, schema interface{}) string {
	schemaJSON, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		l.logger.Warn("Failed to marshal schema", "error", err)
		return prompt
	}

	return fmt.Sprintf("%s\n\nPlease provide your response in JSON format according to this schema:\n%s", prompt, string(schemaJSON))
}

// Stream initiates a streaming response from the LLM.
//
// The caller must Close the returned stream. Close releases the underlying HTTP response body, and
// it is the backstop that reports token usage for a stream the caller walks away from: usage is
// reported automatically when the stream reaches its end, errors, or has its context cancelled, but
// a stream simply dropped mid-flight has no other moment at which to report. The provider billed for
// whatever it generated regardless, so a dropped stream without a Close is spend that no
// UsageObserver will ever see. Closing an already-finished stream is safe and does not double-count.
//
//	stream, err := client.Stream(ctx, prompt)
//	if err != nil {
//		return err
//	}
//	defer stream.Close()
func (l *LLMImpl) Stream(ctx context.Context, prompt *Prompt, opts ...StreamOption) (TokenStream, error) {
	if !l.SupportsStreaming() {
		return nil, NewLLMError(ErrorTypeUnsupported, "streaming not supported by provider", nil)
	}

	// Apply stream options
	config := &StreamConfig{
		BufferSize:  100,
		MaxLineSize: DefaultSSEMaxLineSize,
		RetryStrategy: &DefaultRetryStrategy{
			MaxRetries:  l.MaxRetries,
			InitialWait: l.RetryDelay,
			MaxWait:     l.RetryDelay * 10,
		},
	}
	for _, opt := range opts {
		opt(config)
	}

	// Prepare request with streaming enabled
	options := make(map[string]interface{})
	l.optionsMutex.RLock()
	for k, v := range l.Options {
		options[k] = v
	}
	l.optionsMutex.RUnlock()
	options["stream"] = true

	// Carry tool/function definitions, tool choice, and images into the streaming
	// request the same way the non-stream path does (see GenerateWithUsage and
	// attemptGenerate). Without this, providers never receive the tools and can't
	// emit tool-call deltas, can't honor a forced tool_choice, and drop image
	// inputs for vision models.
	if len(prompt.Tools) > 0 {
		options["tools"] = prompt.Tools
	}
	if tc, ok := toolChoiceValue(prompt.ToolChoice); ok {
		options["tool_choice"] = tc
	}
	if prompt.HasImages() {
		options["images"] = prompt.Images
	}

	// Preserve multi-turn + system structure when the provider supports it.
	// PrepareStreamRequest otherwise flattens the whole prompt into a single user
	// turn (via Prompt.String()), losing roles and prior messages. Providers that
	// implement the optional streamMessagesPreparer take the structured path; the
	// rest fall back to the flattened request unchanged.
	type streamMessagesPreparer interface {
		PrepareStreamRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error)
	}

	var body []byte
	var err error
	if smp, ok := l.Provider.(streamMessagesPreparer); ok && prompt.hasStructuredMessages() {
		if prompt.SystemPrompt != "" {
			options["system_prompt"] = prompt.SystemPrompt
		}
		messages := promptMessagesToMemoryMessages(prompt.Messages)
		body, err = smp.PrepareStreamRequestWithMessages(messages, options)
	} else {
		body, err = l.Provider.PrepareStreamRequest(prompt.String(), options)
	}
	if err != nil {
		return nil, NewLLMError(ErrorTypeRequest, "failed to prepare stream request", err)
	}

	// Retry establishment only (no tokens produced yet, so re-issuing is safe).
	// Once data flows, errors are surfaced by Next — chat streams can't resume.
	retry := config.RetryStrategy
	var resp *http.Response
	for {
		req, err := http.NewRequestWithContext(ctx, "POST", l.Provider.Endpoint(), bytes.NewReader(body))
		if err != nil {
			return nil, NewLLMError(ErrorTypeRequest, "failed to create stream request", err)
		}
		for k, v := range l.Provider.Headers() {
			req.Header.Set(k, v)
		}

		l.logger.Wire("Full API request", "method", req.Method, "url", req.URL.String(), "headers", utils.RedactHTTPHeaders(req.Header), "body", string(body))
		resp, err = l.client.Do(req)
		if err == nil && resp.StatusCode == http.StatusOK {
			break
		}

		// Retry only transient failures (transport, 408/429/5xx); fail fast on
		// other 4xx so a permanent error isn't masked as a slow one.
		var streamErr error
		var transient bool
		if err != nil {
			streamErr = NewLLMError(ErrorTypeAPI, "failed to make stream request", err)
			transient = true
		} else {
			code := resp.StatusCode
			// Read the body BEFORE closing so the provider's error JSON (e.g. a
			// 400 carrying context_length_exceeded, or a 429 detail) is
			// diagnosable on the stream path too, then classify like non-streaming.
			errBody, readErr := io.ReadAll(resp.Body)
			resp.Body.Close()
			if readErr != nil {
				errBody = nil
			}
			l.logger.Warn("API error", "provider", l.Provider.Name(), slog.Int("status", code), "body", string(errBody))
			streamErr = NewLLMError(classifyHTTPStatus(code), fmt.Sprintf("API error: status code %d: %s", code, truncateBytes(errBody, 500)), nil)
			transient = code == http.StatusRequestTimeout || code == http.StatusTooManyRequests || code >= 500
		}

		if !transient || retry == nil || !retry.ShouldRetry(streamErr) {
			return nil, streamErr
		}
		l.logger.Warn("Stream establishment failed, retrying", "error", streamErr)
		if werr := l.waitFor(ctx, retry.NextDelay()); werr != nil {
			return nil, werr
		}
	}

	// Create and return stream. The stream reports its accumulated usage when it ends — the
	// request is billed for whatever it generated even if the consumer abandons it mid-flight.
	report := func(outcome UsageOutcome, model, serviceTier string, usage types.TokenUsage) {
		l.reportStreamUsage(ctx, outcome, model, serviceTier, usage)
	}
	return newProviderStream(resp.Body, l.Provider, config, report), nil
}

// reportStreamUsage fires the usage observer for a completed or abandoned stream. Usage is the
// accumulator's total rather than a parsed body, so there is no response detail to attach.
//
// model is what the stream's own chunks said served it, which is the resolved model rather than the
// requested one — the two differ for gateways and moving aliases, and they are priced differently.
// It falls back to the configured model only when the provider named none, or when the stream was
// abandoned before any chunk arrived.
func (l *LLMImpl) reportStreamUsage(ctx context.Context, outcome UsageOutcome, model, serviceTier string, usage types.TokenUsage) {
	observer := l.usageObserverFn()
	if observer == nil {
		return
	}
	if model == "" && l.config != nil {
		model = l.config.Model
	}
	l.deliverUsage(ctx, observer, UsageEvent{
		Provider:    l.Provider.Name(),
		Model:       model,
		Outcome:     outcome,
		Usage:       usage,
		ServiceTier: serviceTier,
	})
}

// SupportsStreaming checks if the provider supports streaming responses.
func (l *LLMImpl) SupportsStreaming() bool {
	return l.Provider.SupportsStreaming()
}

// richStreamParser is the optional capability that lets a provider surface token
// usage and finish/stop reasons during streaming (not just text). Providers that
// don't implement it fall back to the text-only ParseStreamResponse path.
type richStreamParser interface {
	ParseStreamResponseRich(chunk []byte) (types.StreamChunk, error)
}

// providerStream implements TokenStream for a specific provider
type providerStream struct {
	decoder          StreamDecoder
	provider         providers.Provider
	closer           io.Closer // underlying response body; closed by Close()
	config           *StreamConfig
	currentIndex     int
	usageMutex       sync.RWMutex           // guards usage/model/reachedEnd; read from another goroutine
	usage            types.TokenUsage       // running total, merged across chunks (see mergeUsage)
	model            string                 // model the provider says served the stream, when it says
	serviceTier      string                 // tier the stream was served on, when the provider says
	pendingToolCalls []*types.ToolCallDelta // extra parallel tool-call fragments, drained one per Next
	reachedEnd       bool                   // the stream produced its terminal event, so usage is final

	// reportUsage delivers the accumulated total once the stream ends, however it ends. A stream is
	// billed for what it generated even when the consumer walks away mid-flight, so reporting is
	// driven by termination rather than by the consumer reading to completion.
	reportUsage  func(outcome UsageOutcome, model, serviceTier string, usage types.TokenUsage)
	reportedOnce sync.Once
}

func newProviderStream(reader io.ReadCloser, provider providers.Provider, config *StreamConfig, report func(outcome UsageOutcome, model, serviceTier string, usage types.TokenUsage)) *providerStream {
	var decoder StreamDecoder
	if provider.Name() == "ollama" {
		decoder = NewNDJSONDecoder(reader)
	} else {
		decoder = NewSSEDecoderWithLimit(reader, config.MaxLineSize)
	}

	return &providerStream{
		decoder:      decoder,
		provider:     provider,
		closer:       reader,
		config:       config,
		currentIndex: 0,
		reportUsage:  report,
	}
}

// Usage returns the token usage accumulated so far. It is final once the stream has returned io.EOF
// and is a partial (possibly zero) count before that — providers report usage at the end, and some
// only when asked to. Safe to call from another goroutine.
//
// TotalTokens is filled in from the components, because the streaming shapes that break input into
// cached and uncached parts (Anthropic) send no total of their own at any point.
func (s *providerStream) Usage() types.TokenUsage {
	s.usageMutex.RLock()
	defer s.usageMutex.RUnlock()
	usage := s.usage
	usage.TotalTokens = usage.ComputedTotal()
	return usage
}

// ServiceTier returns the tier the provider says it served this stream on, or empty when it said
// nothing. It scales the price of everything Usage reports. Safe to call from another goroutine.
func (s *providerStream) ServiceTier() string {
	s.usageMutex.RLock()
	defer s.usageMutex.RUnlock()
	return s.serviceTier
}

// endOfStream marks the stream as having reached its terminal event, reports the final usage, and
// returns the io.EOF the caller propagates.
func (s *providerStream) endOfStream() error {
	s.usageMutex.Lock()
	s.reachedEnd = true
	s.usageMutex.Unlock()
	s.finish()
	return io.EOF
}

// finish reports the stream's accumulated usage exactly once, whether it ended at its terminal
// event, at a mid-stream error, or at an early Close by the consumer.
func (s *providerStream) finish() {
	s.reportedOnce.Do(func() {
		if s.reportUsage == nil {
			return
		}
		s.usageMutex.RLock()
		model, tier, ended := s.model, s.serviceTier, s.reachedEnd
		s.usageMutex.RUnlock()
		usage := s.Usage()

		outcome := UsageOutcomeStreamAborted
		if ended {
			outcome = UsageOutcomeStream
		}
		s.reportUsage(outcome, model, tier, usage)
	})
}

// mergeUsage overwrites dst per non-zero field. Providers report usage
// cumulatively (Anthropic splits input/output across events), so taking the
// latest non-zero value is correct and avoids double-counting.
func mergeUsage(dst *types.TokenUsage, src types.TokenUsage) {
	if src.PromptTokens > 0 {
		dst.PromptTokens = src.PromptTokens
	}
	if src.CompletionTokens > 0 {
		dst.CompletionTokens = src.CompletionTokens
	}
	if src.TotalTokens > 0 {
		dst.TotalTokens = src.TotalTokens
	}
	if src.CacheCreationInputTokens > 0 {
		dst.CacheCreationInputTokens = src.CacheCreationInputTokens
	}
	if src.CacheReadInputTokens > 0 {
		dst.CacheReadInputTokens = src.CacheReadInputTokens
	}
	if src.CachedPromptTokens > 0 {
		dst.CachedPromptTokens = src.CachedPromptTokens
	}
	if src.ReasoningTokens > 0 {
		dst.ReasoningTokens = src.ReasoningTokens
	}
	if src.CacheCreation5mInputTokens > 0 {
		dst.CacheCreation5mInputTokens = src.CacheCreation5mInputTokens
	}
	if src.CacheCreation1hInputTokens > 0 {
		dst.CacheCreation1hInputTokens = src.CacheCreation1hInputTokens
	}
	if src.CacheWritePromptTokens > 0 {
		dst.CacheWritePromptTokens = src.CacheWritePromptTokens
	}
	if src.AcceptedPredictionTokens > 0 {
		dst.AcceptedPredictionTokens = src.AcceptedPredictionTokens
	}
	if src.RejectedPredictionTokens > 0 {
		dst.RejectedPredictionTokens = src.RejectedPredictionTokens
	}
	if src.AudioPromptTokens > 0 {
		dst.AudioPromptTokens = src.AudioPromptTokens
	}
	if src.AudioCompletionTokens > 0 {
		dst.AudioCompletionTokens = src.AudioCompletionTokens
	}
}

// toolCallToken builds a tool_call_delta StreamToken and advances the index.
func (s *providerStream) toolCallToken(d *types.ToolCallDelta) *StreamToken {
	tok := &StreamToken{Type: "tool_call_delta", Index: s.currentIndex, ToolCallDelta: d}
	s.currentIndex++
	return tok
}

func (s *providerStream) Next(ctx context.Context) (*StreamToken, error) {
	rich, hasRich := s.provider.(richStreamParser)
	for {
		select {
		case <-ctx.Done():
			// Cancellation is the usual way a stream ends early — a request timeout, a
			// disconnected client, a dying parent context — and the provider has already
			// generated and billed whatever arrived before it. Report here rather than
			// leaving it to Close, which the caller is under no obligation to call.
			s.finish()
			return nil, ctx.Err()
		default:
			// Drain buffered parallel tool-call fragments first.
			if len(s.pendingToolCalls) > 0 {
				d := s.pendingToolCalls[0]
				s.pendingToolCalls = s.pendingToolCalls[1:]
				return s.toolCallToken(d), nil
			}

			if !s.decoder.Next() {
				if err := s.decoder.Err(); err != nil {
					// Surfaced to the caller; not retried in place (body is
					// already partially consumed — re-invoke Stream to retry).
					// Whatever the provider generated before the break is billed.
					s.finish()
					return nil, err
				}
				return nil, s.endOfStream()
			}

			event := s.decoder.Event()
			if len(event.Data) == 0 {
				continue
			}

			// Rich path: providers that report usage/finish during streaming.
			if hasRich {
				chunk, err := rich.ParseStreamResponseRich(event.Data)
				if err != nil {
					if errors.Is(err, types.ErrStreamSkip) {
						continue
					}
					if err == io.EOF {
						return nil, s.endOfStream()
					}
					// Genuine parse/API error — surface it instead of swallowing, after
					// recording what the stream had already been billed for.
					s.finish()
					return nil, err
				}
				// Buffer any additional parallel tool-call fragments to emit next.
				if len(chunk.ExtraToolCallDeltas) > 0 {
					s.pendingToolCalls = append(s.pendingToolCalls, chunk.ExtraToolCallDeltas...)
				}
				kind := chunk.Kind
				if kind == "" {
					kind = "text"
				}
				token := &StreamToken{Text: chunk.Text, Type: kind, Index: s.currentIndex}
				if chunk.ToolCallDelta != nil {
					token.ToolCallDelta = chunk.ToolCallDelta
				}
				if chunk.Usage != nil || chunk.Model != "" || chunk.ServiceTier != "" {
					s.usageMutex.Lock()
					if chunk.Usage != nil {
						mergeUsage(&s.usage, *chunk.Usage)
					}
					if chunk.Model != "" {
						s.model = chunk.Model
					}
					if chunk.ServiceTier != "" {
						s.serviceTier = chunk.ServiceTier
					}
					s.usageMutex.Unlock()
				}
				if chunk.Usage != nil || chunk.FinishReason != "" {
					u := s.Usage()
					total := u.TotalTokens
					token.Metadata = map[string]interface{}{
						"prompt_tokens":                  u.PromptTokens,
						"completion_tokens":              u.CompletionTokens,
						"total_tokens":                   total,
						"cache_creation_input_tokens":    u.CacheCreationInputTokens,
						"cache_read_input_tokens":        u.CacheReadInputTokens,
						"cached_prompt_tokens":           u.CachedPromptTokens,
						"reasoning_tokens":               u.ReasoningTokens,
						"cache_creation_5m_input_tokens": u.CacheCreation5mInputTokens,
						"cache_creation_1h_input_tokens": u.CacheCreation1hInputTokens,
						"cache_write_prompt_tokens":      u.CacheWritePromptTokens,
						"accepted_prediction_tokens":     u.AcceptedPredictionTokens,
						"rejected_prediction_tokens":     u.RejectedPredictionTokens,
						"audio_prompt_tokens":            u.AudioPromptTokens,
						"audio_completion_tokens":        u.AudioCompletionTokens,
					}
					if s.ServiceTier() != "" {
						token.Metadata["service_tier"] = s.ServiceTier()
					}
					if chunk.FinishReason != "" {
						token.Metadata["finish_reason"] = chunk.FinishReason
					}
				}
				s.currentIndex++
				return token, nil
			}

			// Text-only fallback for providers without rich parsing.
			token, err := s.provider.ParseStreamResponse(event.Data)
			if err != nil {
				if err.Error() == "skip token" {
					continue
				}
				if err == io.EOF {
					return nil, s.endOfStream()
				}
				continue // Not enough data or malformed
			}

			// Create and return token
			tok := &StreamToken{
				Text:  token,
				Type:  event.Type,
				Index: s.currentIndex,
			}
			s.currentIndex++
			return tok, nil
		}
	}
}

// Close releases the underlying response body and reports usage if the stream never reached its
// terminal event — a consumer that stops early still pays for what was generated. Reporting is
// once-only, so closing a fully consumed stream does not double-count it.
func (s *providerStream) Close() error {
	s.finish()
	if s.closer != nil {
		return s.closer.Close()
	}
	return nil
}
