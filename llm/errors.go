package llm

import (
	"fmt"
	"net/http"
	"unicode/utf8"

	"github.com/teilomillet/gollm/utils"
)

// ErrorType represents the category of an LLM error.
// It helps classify errors for appropriate handling and logging.
type ErrorType int

const (
	// ErrorTypeUnknown represents an unclassified error
	ErrorTypeUnknown ErrorType = iota

	// ErrorTypeProvider indicates an error from the LLM provider
	ErrorTypeProvider

	// ErrorTypeRequest indicates an error in preparing or sending the request
	ErrorTypeRequest

	// ErrorTypeResponse indicates an error in processing the response
	ErrorTypeResponse

	// ErrorTypeAPI indicates an error returned by the provider's API
	ErrorTypeAPI

	// ErrorTypeRateLimit indicates the provider's rate limit has been exceeded
	ErrorTypeRateLimit

	// ErrorTypeAuthentication indicates an authentication or authorization failure
	ErrorTypeAuthentication

	// ErrorTypeInvalidInput indicates invalid input parameters or prompt
	ErrorTypeInvalidInput

	// ErrorTypeUnsupported indicates a requested feature is not supported
	ErrorTypeUnsupported
)

// LLMError represents a structured error in the LLM package.
// It implements the error interface and provides additional context
// about the error type and underlying cause.
type LLMError struct {
	Type    ErrorType // The category of the error
	Message string    // A human-readable error message
	Err     error     // The underlying error, if any
}

// LoggableFields returns a slice of interface{} containing error information
// in a format suitable for structured logging.
func (e *LLMError) LoggableFields() []interface{} {
	return []interface{}{
		"error_type", e.TypeString(),
		"message", e.Message,
		"error", e.Err,
	}
}

// Error implements the error interface.
// It returns a formatted string containing the error type, message,
// and underlying error (if present).
func (e *LLMError) Error() string {
	if e.Err != nil {
		return fmt.Sprintf("%s (%s): %v", e.TypeString(), e.Message, e.Err)
	}
	return fmt.Sprintf("%s: %s", e.TypeString(), e.Message)
}

// Unwrap returns the underlying error.
// This implements the Go 1.13+ error unwrapping interface.
func (e *LLMError) Unwrap() error {
	return e.Err
}

// TypeString returns a string representation of the error type.
// This is used for logging and error messages.
func (e *LLMError) TypeString() string {
	switch e.Type {
	case ErrorTypeProvider:
		return "ProviderError"
	case ErrorTypeRequest:
		return "RequestError"
	case ErrorTypeResponse:
		return "ResponseError"
	case ErrorTypeAPI:
		return "APIError"
	case ErrorTypeRateLimit:
		return "RateLimitError"
	case ErrorTypeAuthentication:
		return "AuthenticationError"
	case ErrorTypeInvalidInput:
		return "InvalidInputError"
	case ErrorTypeUnsupported:
		return "UnsupportedError"
	default:
		return "UnknownError"
	}
}

// NewLLMError creates a new LLMError with the specified type, message,
// and underlying error.
//
// Parameters:
//   - errType: The category of the error
//   - message: A human-readable error message
//   - err: The underlying error, if any
//
// Returns:
//   - A new LLMError instance
func NewLLMError(errType ErrorType, message string, err error) *LLMError {
	return &LLMError{
		Type:    errType,
		Message: message,
		Err:     err,
	}
}

// classifyHTTPStatus maps a non-200 HTTP status code to the most specific
// ErrorType available, so callers (and the backend) can distinguish retryable
// rate limits, non-retryable auth failures, and deterministic bad requests
// from generic API errors.
//
// Only the statuses below are intentionally specialized:
//   - 429                     -> RateLimit      (retryable with backoff)
//   - 401 / 403               -> Authentication (non-retryable; user config)
//   - 400 / 413 / 422         -> InvalidInput   (deterministic; e.g. context-length,
//     oversized payload, or unprocessable params)
//
// Everything else (incl. 408, 409, and all 5xx) falls back to ErrorTypeAPI.
// Note that 408 and 5xx are still retried on the stream path via a separate
// status check, independent of this type.
func classifyHTTPStatus(statusCode int) ErrorType {
	switch statusCode {
	case http.StatusTooManyRequests: // 429
		return ErrorTypeRateLimit
	case http.StatusUnauthorized, http.StatusForbidden: // 401 / 403
		return ErrorTypeAuthentication
	case http.StatusBadRequest, // 400
		http.StatusRequestEntityTooLarge, // 413 (payload/context too large)
		http.StatusUnprocessableEntity:   // 422 (invalid params on some providers)
		return ErrorTypeInvalidInput
	default:
		return ErrorTypeAPI
	}
}

// truncateBytes caps b at max runes (appending an ellipsis when truncated) and
// returns it as a string, so large provider error payloads can be carried in an
// error/log line without dumping the full body. It walks to a UTF-8 rune
// boundary and converts only the retained prefix — no []rune copy and no
// full-body string conversion — and the result is always valid UTF-8 even when
// the body contains multibyte characters.
func truncateBytes(b []byte, max int) string {
	// Fast path: byte length ≤ max implies rune count ≤ max, so no truncation.
	if len(b) <= max {
		return string(b)
	}
	i, n := 0, 0
	for i < len(b) && n < max {
		_, size := utf8.DecodeRune(b[i:])
		i += size
		n++
	}
	if i >= len(b) {
		return string(b)
	}
	return string(b[:i]) + "…"
}

// HandleError processes an error based on its severity.
// It logs the error appropriately and can optionally terminate the program
// if the error is considered fatal.
//
// Parameters:
//   - err: The error to handle
//   - fatal: If true, the program will panic after logging
//   - logger: The logger to use for error reporting
func HandleError(err error, fatal bool, logger utils.Logger) {
	if err == nil {
		return
	}

	if llmErr, ok := err.(*LLMError); ok {
		logger.Error(llmErr.Message, "error_type", llmErr.TypeString(), "error", llmErr.Err)
	} else {
		logger.Error("An error occurred", "error", err)
	}

	if fatal {
		// Consider using os.Exit(1) here or returning an error to let the caller decide
		panic(err)
	}
}
