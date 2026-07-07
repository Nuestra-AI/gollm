package llm

import (
	"net/http"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/require"
)

// TestClassifyHTTPStatus verifies that non-200 statuses map to the most
// specific ErrorType, so the backend can distinguish retryable rate limits,
// non-retryable auth failures, and deterministic bad requests via the stable
// TypeString() prefix.
func TestClassifyHTTPStatus(t *testing.T) {
	cases := []struct {
		status     int
		wantType   ErrorType
		wantString string
	}{
		{http.StatusTooManyRequests, ErrorTypeRateLimit, "RateLimitError"},
		{http.StatusUnauthorized, ErrorTypeAuthentication, "AuthenticationError"},
		{http.StatusForbidden, ErrorTypeAuthentication, "AuthenticationError"},
		{http.StatusBadRequest, ErrorTypeInvalidInput, "InvalidInputError"},
		{http.StatusRequestEntityTooLarge, ErrorTypeInvalidInput, "InvalidInputError"},
		{http.StatusUnprocessableEntity, ErrorTypeInvalidInput, "InvalidInputError"},
		{http.StatusInternalServerError, ErrorTypeAPI, "APIError"},
		{http.StatusBadGateway, ErrorTypeAPI, "APIError"},
		{http.StatusNotFound, ErrorTypeAPI, "APIError"},
		{http.StatusRequestTimeout, ErrorTypeAPI, "APIError"}, // 408 stays API; transient handled separately
		{http.StatusConflict, ErrorTypeAPI, "APIError"},       // 409 stays API
	}

	for _, tc := range cases {
		t.Run(http.StatusText(tc.status), func(t *testing.T) {
			got := classifyHTTPStatus(tc.status)
			require.Equal(t, tc.wantType, got)

			err := NewLLMError(got, "API error: status code", nil)
			require.Equal(t, tc.wantString, err.TypeString())
		})
	}
}

// TestTruncateBytes verifies bodies are capped so large provider payloads don't
// get dumped into the error/log line, while short bodies pass through unchanged.
func TestTruncateBytes(t *testing.T) {
	require.Equal(t, "short", truncateBytes([]byte("short"), 500))

	// Exactly max runes: no truncation, no ellipsis.
	require.Equal(t, strings.Repeat("a", 500), truncateBytes([]byte(strings.Repeat("a", 500)), 500))

	long := make([]byte, 600)
	for i := range long {
		long[i] = 'a'
	}
	got := truncateBytes(long, 500)
	require.Len(t, []rune(got), 501) // 500 chars + ellipsis rune
	require.Contains(t, got, "…")

	// Multibyte bodies must be truncated on a rune boundary so the result stays
	// valid UTF-8 (each 'é' is 2 bytes; a byte-slice at 500 would split one).
	multibyte := []byte(strings.Repeat("é", 600))
	gotMB := truncateBytes(multibyte, 500)
	require.True(t, utf8.ValidString(gotMB), "truncated string must be valid UTF-8")
	require.Len(t, []rune(gotMB), 501) // 500 runes + ellipsis
	require.Equal(t, strings.Repeat("é", 500)+"…", gotMB)
}
