package types

import "errors"

// ErrStreamSkip is a sentinel a stream parser returns for a chunk with no token
// to emit (keep-alives, role-only deltas, empty data). Consumers should detect it
// with errors.Is and continue; any other error is theirs to surface rather than
// swallow, so malformed payloads and in-band API errors don't go unnoticed.
var ErrStreamSkip = errors.New("skip token")

// StreamChunk is a provider-normalized unit emitted by the optional
// ParseStreamResponseRich path. It extends the plain text token returned by
// ParseStreamResponse with token usage and a finish/stop reason, bringing the
// streaming path to parity with the non-stream ParseResponseWithUsage path.
//
// A single chunk may carry more than one signal — notably Anthropic's final
// message_delta event reports both the stop reason and the output-token count —
// so consumers should read every populated field rather than switch on Kind
// alone.
type StreamChunk struct {
	Text          string         // incremental text (empty for usage/finish-only chunks)
	Kind          string         // primary signal: "text" | "usage" | "finish" | "tool_call_delta"
	FinishReason  string         // provider stop/finish reason (set on finish chunks)
	Usage         *TokenUsage    // token usage, when the provider reports it mid/end of stream
	ToolCallDelta *ToolCallDelta // incremental tool-call fragment (set when Kind == "tool_call_delta")
	// ExtraToolCallDeltas carries additional fragments when one chunk opens
	// multiple parallel calls; the loop emits them as subsequent tokens.
	ExtraToolCallDeltas []*ToolCallDelta
}

// ToolCallDelta is one incremental fragment of a streamed tool/function call.
// Providers emit a tool call across many chunks: an opening fragment carries the
// ID and Name; subsequent fragments carry partial-JSON ArgsFragment pieces that
// the consumer concatenates per Index to reconstruct the full arguments. The
// rich parser stays stateless per-chunk; assembly (keyed by Index) is the
// consumer's job.
type ToolCallDelta struct {
	Index        int    // which tool call this fragment belongs to (provider-assigned slot)
	ID           string // tool-call id; set on the opening fragment, empty thereafter
	Name         string // function name; set on the opening fragment, empty thereafter
	ArgsFragment string // partial JSON of the arguments to append for this Index
}
