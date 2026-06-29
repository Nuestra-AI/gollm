package llm

import (
	"bufio"
	"bytes"
	"context"
	"io"
	"time"

	"github.com/teilomillet/gollm/types"
)

// StreamToken represents a single token from the streaming response.
type StreamToken struct {
	// Text is the actual token text
	Text string

	// Type indicates the type of token (e.g., "text", "usage", "finish",
	// "tool_call_delta", "error")
	Type string

	// Index is the position of this token in the sequence
	Index int

	// Metadata contains provider-specific metadata
	Metadata map[string]interface{}

	// ToolCallDelta carries an incremental tool-call fragment when
	// Type == "tool_call_delta"; nil otherwise.
	ToolCallDelta *types.ToolCallDelta
}

// TokenStream represents a stream of tokens from the LLM.
// It follows Go's io.ReadCloser pattern but with token-level granularity.
type TokenStream interface {
	// Next returns the next token in the stream.
	// When the stream is finished, it returns io.EOF.
	Next(context.Context) (*StreamToken, error)

	// Close releases any resources associated with the stream.
	io.Closer
}

// StreamOption is a function type for configuring streaming behavior.
type StreamOption func(*StreamConfig)

// StreamConfig holds configuration options for streaming.
type StreamConfig struct {
	// BufferSize is the size of the token buffer
	BufferSize int

	// RetryStrategy governs retries while *establishing* the stream (connection
	// errors and non-200 responses, before any token is produced). It does not
	// apply once the stream is yielding data: chat streams have no resumption
	// point, so mid-stream interruptions are surfaced to the caller instead.
	RetryStrategy RetryStrategy

	// MaxLineSize caps a single SSE line (defaults to DefaultSSEMaxLineSize when
	// zero). Raise it via WithMaxLineSize for streams with >1 MB lines; per-stream,
	// so it's race-free.
	MaxLineSize int
}

// WithMaxLineSize sets the per-stream SSE line cap (see StreamConfig.MaxLineSize).
func WithMaxLineSize(n int) StreamOption {
	return func(c *StreamConfig) { c.MaxLineSize = n }
}

// RetryStrategy governs retries while establishing a stream (see
// StreamConfig.RetryStrategy). It is not consulted for mid-stream interruptions.
type RetryStrategy interface {
	// ShouldRetry determines if a retry should be attempted.
	ShouldRetry(error) bool

	// NextDelay returns the delay before the next retry.
	NextDelay() time.Duration

	// Reset resets the retry state.
	Reset()
}

// DefaultRetryStrategy implements a simple exponential backoff strategy.
type DefaultRetryStrategy struct {
	MaxRetries  int
	InitialWait time.Duration
	MaxWait     time.Duration
	attempts    int
}

func (s *DefaultRetryStrategy) ShouldRetry(err error) bool {
	return s.attempts < s.MaxRetries
}

func (s *DefaultRetryStrategy) NextDelay() time.Duration {
	s.attempts++
	delay := s.InitialWait * time.Duration(1<<uint(s.attempts-1))
	if delay > s.MaxWait {
		delay = s.MaxWait
	}
	return delay
}

func (s *DefaultRetryStrategy) Reset() {
	s.attempts = 0
}

// StreamDecoder interface for different streaming formats
type StreamDecoder interface {
	Next() bool
	Event() Event
	Err() error
}

// DefaultSSEMaxLineSize is the default per-line cap for the SSE decoder. It is
// raised well above bufio.Scanner's 64KB default because a single SSE data line
// can carry a large delta — notably streamed tool-call arguments — and would
// otherwise fail mid-stream with bufio.ErrTooLong.
const DefaultSSEMaxLineSize = 1024 * 1024

// SSEDecoder handles Server-Sent Events (SSE) streaming
type SSEDecoder struct {
	reader  *bufio.Scanner
	current Event
	err     error
}

type Event struct {
	Type string
	Data []byte
}

func NewSSEDecoder(reader io.Reader) *SSEDecoder {
	return NewSSEDecoderWithLimit(reader, DefaultSSEMaxLineSize)
}

// NewSSEDecoderWithLimit caps a single SSE line at maxLineSize bytes (default
// when <= 0), raised above bufio's 64KB default so large deltas (streamed
// tool-call arguments) don't fail with bufio.ErrTooLong.
func NewSSEDecoderWithLimit(reader io.Reader, maxLineSize int) *SSEDecoder {
	if maxLineSize <= 0 {
		maxLineSize = DefaultSSEMaxLineSize
	}
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), maxLineSize)
	return &SSEDecoder{
		reader: scanner,
	}
}

func (d *SSEDecoder) Next() bool {
	if d.err != nil {
		return false
	}

	event := ""
	data := bytes.NewBuffer(nil)

	for d.reader.Scan() {
		line := d.reader.Bytes()

		// Dispatch event on empty line
		if len(line) == 0 {
			d.current = Event{
				Type: event,
				Data: data.Bytes(),
			}
			return true
		}

		// Split "event: value" into parts
		name, value, _ := bytes.Cut(line, []byte(":"))

		// Remove optional space after colon
		if len(value) > 0 && value[0] == ' ' {
			value = value[1:]
		}

		switch string(name) {
		case "":
			continue // Skip comments
		case "event":
			event = string(value)
		case "data":
			data.Write(value)
			data.WriteRune('\n')
		}
	}

	return false
}

func (d *SSEDecoder) Event() Event {
	return d.current
}

func (d *SSEDecoder) Err() error {
	return d.err
}

// NDJSONDecoder handles Newline Delimited JSON streaming
type NDJSONDecoder struct {
	scanner *bufio.Scanner
	current Event
	err     error
}

func NewNDJSONDecoder(reader io.Reader) *NDJSONDecoder {
	return &NDJSONDecoder{
		scanner: bufio.NewScanner(reader),
	}
}

func (d *NDJSONDecoder) Next() bool {
	if d.err != nil {
		return false
	}

	// Use loop instead of recursion to avoid stack overflow on many empty lines
	for d.scanner.Scan() {
		line := d.scanner.Bytes()
		if len(line) == 0 {
			continue // Skip empty lines
		}

		d.current = Event{
			Type: "text",
			Data: line,
		}
		return true
	}

	d.err = d.scanner.Err()
	return false
}

func (d *NDJSONDecoder) Event() Event {
	return d.current
}

func (d *NDJSONDecoder) Err() error {
	return d.err
}
