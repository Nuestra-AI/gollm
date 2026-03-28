package utils

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// LogLevel
// ---------------------------------------------------------------------------

func TestLogLevelString(t *testing.T) {
	tests := []struct {
		level    LogLevel
		expected string
	}{
		{LogLevelOff, "OFF"},
		{LogLevelError, "ERROR"},
		{LogLevelWarn, "WARN"},
		{LogLevelInfo, "INFO"},
		{LogLevelWire, "WIRE"},
		{LogLevelDebug, "DEBUG"},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.expected, tt.level.String())
	}
}

func TestLogLevelUnmarshalText(t *testing.T) {
	tests := []struct {
		input    string
		expected LogLevel
	}{
		{"OFF", LogLevelOff},
		{"off", LogLevelOff},
		{"ERROR", LogLevelError},
		{"error", LogLevelError},
		{"WARN", LogLevelWarn},
		{"warn", LogLevelWarn},
		{"INFO", LogLevelInfo},
		{"info", LogLevelInfo},
		{"WIRE", LogLevelWire},
		{"wire", LogLevelWire},
		{"DEBUG", LogLevelDebug},
		{"debug", LogLevelDebug},
	}
	for _, tt := range tests {
		var level LogLevel
		err := level.UnmarshalText([]byte(tt.input))
		require.NoError(t, err, "input: %s", tt.input)
		assert.Equal(t, tt.expected, level, "input: %s", tt.input)
	}
}

func TestLogLevelUnmarshalTextInvalid(t *testing.T) {
	var level LogLevel
	err := level.UnmarshalText([]byte("TRACE"))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid log level")
}

// ---------------------------------------------------------------------------
// Sensitive header detection
// ---------------------------------------------------------------------------

func TestIsSensitiveHeader(t *testing.T) {
	sensitive := []string{
		"Authorization", "authorization", "AUTHORIZATION",
		"X-Api-Key", "x-api-key", "X-API-KEY",
		"Api-Key", "api-key", "API-KEY",
	}
	for _, h := range sensitive {
		assert.True(t, isSensitiveHeader(h), "expected %q to be sensitive", h)
	}

	notSensitive := []string{
		"Content-Type", "Accept", "X-Request-Id", "User-Agent",
	}
	for _, h := range notSensitive {
		assert.False(t, isSensitiveHeader(h), "expected %q to NOT be sensitive", h)
	}
}

// ---------------------------------------------------------------------------
// Redaction
// ---------------------------------------------------------------------------

func TestRedactValueLongString(t *testing.T) {
	// Value longer than 8 chars: keep last 8, prefix with "..."
	result := redactValue("sk-1234567890abcdef")
	assert.Equal(t, "...90abcdef", result)
}

func TestRedactValueShortString(t *testing.T) {
	// Value 8 chars or less: replace entirely with "..."
	assert.Equal(t, "...", redactValue("short"))
	assert.Equal(t, "...", redactValue("12345678"))
}

func TestRedactValueEmptyString(t *testing.T) {
	assert.Equal(t, "...", redactValue(""))
}

func TestRedactHeaders(t *testing.T) {
	headers := map[string]string{
		"Authorization": "Bearer sk-1234567890abcdef",
		"Content-Type":  "application/json",
		"x-api-key":     "key-abcdefghijklmnop",
		"Api-Key":       "azure-key-1234567890",
	}

	redacted := RedactHeaders(headers)

	// Non-sensitive headers should be unchanged
	assert.Equal(t, "application/json", redacted["Content-Type"])

	// Sensitive headers should be redacted
	assert.Equal(t, "...90abcdef", redacted["Authorization"])
	assert.Equal(t, "...ijklmnop", redacted["x-api-key"])
	assert.Equal(t, "...34567890", redacted["Api-Key"])

	// Original should be unchanged
	assert.Equal(t, "Bearer sk-1234567890abcdef", headers["Authorization"])
}

func TestRedactHeadersEmpty(t *testing.T) {
	redacted := RedactHeaders(map[string]string{})
	assert.Empty(t, redacted)
}

func TestRedactHTTPHeaders(t *testing.T) {
	headers := http.Header{
		"Authorization": {"Bearer sk-1234567890abcdef"},
		"Content-Type":  {"application/json"},
		"X-Api-Key":     {"key-abcdefghijklmnop", "key-second-value-here"},
	}

	redacted := RedactHTTPHeaders(headers)

	// Non-sensitive headers copied verbatim
	assert.Equal(t, []string{"application/json"}, redacted["Content-Type"])

	// Sensitive headers redacted
	assert.Equal(t, []string{"...90abcdef"}, redacted["Authorization"])
	assert.Len(t, redacted["X-Api-Key"], 2)
	assert.Equal(t, "...ijklmnop", redacted["X-Api-Key"][0])
	assert.Equal(t, "...lue-here", redacted["X-Api-Key"][1])

	// Original should be unchanged
	assert.Equal(t, "Bearer sk-1234567890abcdef", headers.Get("Authorization"))
}

func TestRedactHTTPHeadersEmpty(t *testing.T) {
	redacted := RedactHTTPHeaders(http.Header{})
	assert.Empty(t, redacted)
}

// ---------------------------------------------------------------------------
// NopLogger
// ---------------------------------------------------------------------------

func TestNopLoggerImplementsInterface(t *testing.T) {
	var _ Logger = NewNopLogger()
}

func TestNopLoggerDoesNotPanic(t *testing.T) {
	l := NewNopLogger()
	// All methods should be no-ops without panicking
	l.Debug("msg", "key", "value")
	l.Wire("msg", "key", "value")
	l.Info("msg", "key", "value")
	l.Warn("msg", "key", "value")
	l.Error("msg", "key", "value")
	l.SetLevel(LogLevelDebug)
}

// ---------------------------------------------------------------------------
// DefaultLogger
// ---------------------------------------------------------------------------

func TestNewLoggerSetsLevel(t *testing.T) {
	l := NewLogger(LogLevelWarn)
	dl, ok := l.(*DefaultLogger)
	require.True(t, ok)
	assert.Equal(t, LogLevelWarn, dl.level)
}

func TestDefaultLoggerSetLevel(t *testing.T) {
	l := NewLogger(LogLevelInfo)
	l.SetLevel(LogLevelDebug)
	dl := l.(*DefaultLogger)
	assert.Equal(t, LogLevelDebug, dl.level)
}
