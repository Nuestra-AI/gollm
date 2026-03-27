package utils

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
)

type LogLevel int

const (
	LogLevelOff LogLevel = iota
	LogLevelError
	LogLevelWarn
	LogLevelInfo
	LogLevelWire // level that includes wire-level request/response logging
	LogLevelDebug
)

type Logger interface {
	Debug(msg string, keysAndValues ...interface{})
	Wire(msg string, keysAndValues ...interface{})
	Info(msg string, keysAndValues ...interface{})
	Warn(msg string, keysAndValues ...interface{})
	Error(msg string, keysAndValues ...interface{})
	SetLevel(level LogLevel)
}

type DefaultLogger struct {
	logger *log.Logger
	level  LogLevel
}

func NewLogger(level LogLevel) Logger {
	return &DefaultLogger{
		logger: log.New(os.Stderr, "", log.LstdFlags),
		level:  level,
	}
}

func (l *DefaultLogger) SetLevel(level LogLevel) {
	l.level = level
}

func (l *DefaultLogger) log(level LogLevel, msg string, keysAndValues ...interface{}) {
	if level <= l.level {
		l.logger.Printf("%s: %s %v", level, msg, keysAndValues)
	}
}

func (l *DefaultLogger) Debug(msg string, keysAndValues ...interface{}) {
	l.log(LogLevelDebug, msg, keysAndValues...)
}

func (l *DefaultLogger) Wire(msg string, keysAndValues ...interface{}) {
	l.log(LogLevelWire, msg, keysAndValues...)
}

func (l *DefaultLogger) Info(msg string, keysAndValues ...interface{}) {
	l.log(LogLevelInfo, msg, keysAndValues...)
}

func (l *DefaultLogger) Warn(msg string, keysAndValues ...interface{}) {
	l.log(LogLevelWarn, msg, keysAndValues...)
}

func (l *DefaultLogger) Error(msg string, keysAndValues ...interface{}) {
	l.log(LogLevelError, msg, keysAndValues...)
}

func (l LogLevel) String() string {
	return [...]string{"OFF", "ERROR", "WARN", "INFO", "WIRE", "DEBUG"}[l]
}

func (l *LogLevel) UnmarshalText(text []byte) error {
	switch strings.ToUpper(string(text)) {
	case "OFF":
		*l = LogLevelOff
	case "ERROR":
		*l = LogLevelError
	case "WARN":
		*l = LogLevelWarn
	case "INFO":
		*l = LogLevelInfo
	case "WIRE":
		*l = LogLevelWire
	case "DEBUG":
		*l = LogLevelDebug
	default:
		return fmt.Errorf("invalid log level: %s", string(text))
	}
	return nil
}

// sensitiveHeaders lists HTTP header keys whose values contain credentials.
var sensitiveHeaders = []string{
	"authorization",
	"x-api-key",
	"api-key",
}

// isSensitiveHeader returns true if the header key is known to carry credentials.
func isSensitiveHeader(key string) bool {
	lower := strings.ToLower(key)
	for _, h := range sensitiveHeaders {
		if lower == h {
			return true
		}
	}
	return false
}

// redactValue keeps the last 8 characters (or fewer if the value is short)
// and replaces the rest with "...".
func redactValue(value string) string {
	const tailLen = 8
	if len(value) <= tailLen {
		return "..."
	}
	return "..." + value[len(value)-tailLen:]
}

// RedactHeaders returns a copy of headers with sensitive values partially redacted.
// The last 8 characters of each sensitive value are preserved for identification.
func RedactHeaders(headers map[string]string) map[string]string {
	redacted := make(map[string]string, len(headers))
	for k, v := range headers {
		if isSensitiveHeader(k) {
			redacted[k] = redactValue(v)
		} else {
			redacted[k] = v
		}
	}
	return redacted
}

// RedactHTTPHeaders returns a copy of http.Header with sensitive values partially redacted.
func RedactHTTPHeaders(headers http.Header) http.Header {
	redacted := make(http.Header, len(headers))
	for k, vals := range headers {
		if isSensitiveHeader(k) {
			redactedVals := make([]string, len(vals))
			for i, v := range vals {
				redactedVals[i] = redactValue(v)
			}
			redacted[k] = redactedVals
		} else {
			redacted[k] = append([]string(nil), vals...)
		}
	}
	return redacted
}

// NopLogger is a logger that discards all output.
// Use this when you want to completely disable logging.
type NopLogger struct{}

// NewNopLogger creates a new no-op logger that discards all output.
func NewNopLogger() Logger {
	return &NopLogger{}
}

func (l *NopLogger) Debug(msg string, keysAndValues ...interface{}) {}
func (l *NopLogger) Wire(msg string, keysAndValues ...interface{})  {}
func (l *NopLogger) Info(msg string, keysAndValues ...interface{})  {}
func (l *NopLogger) Warn(msg string, keysAndValues ...interface{})  {}
func (l *NopLogger) Error(msg string, keysAndValues ...interface{}) {}
func (l *NopLogger) SetLevel(level LogLevel)                        {}
