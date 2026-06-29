# Streaming

This fork's streaming path reaches parity with the blocking `GenerateWithUsage`
path: it preserves conversation structure, reports token usage and the finish
reason, runs the request with tools, and surfaces tool/function calls as
incremental deltas — all while keeping the `Provider` interface backward
compatible. Providers opt in via small optional capability interfaces; providers
that don't implement them fall back to plain text streaming unchanged.

## Quick start

```go
stream, err := llm.Stream(ctx, prompt) // prompt may carry SystemPrompt, Messages, Tools
if err != nil {
    return err
}
defer stream.Close() // closes the underlying HTTP body — always call it

for {
    tok, err := stream.Next(ctx)
    if err == io.EOF {
        break
    }
    if err != nil {
        return err
    }
    switch tok.Type {
    case "text":
        fmt.Print(tok.Text)
    case "usage", "finish":
        // tok.Metadata carries usage ints and "finish_reason" (see below)
    case "tool_call_delta":
        // tok.ToolCallDelta is one fragment of a tool call (assemble per Index)
    }
}
```

## What a token carries

`llm.StreamToken` (in `llm/stream.go`):

| Field | Meaning |
|---|---|
| `Text` | incremental text (set when `Type == "text"`) |
| `Type` | `"text"` \| `"usage"` \| `"finish"` \| `"tool_call_delta"` |
| `Metadata` | populated on usage/finish tokens: `prompt_tokens`, `completion_tokens`, `total_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens` (all `int`), and `finish_reason` (`string`) |
| `ToolCallDelta` | `*types.ToolCallDelta`, set when `Type == "tool_call_delta"` |

## Capabilities and how they work

### 1. Structured-message streaming

`Stream` preserves the system prompt and multi-turn roles instead of flattening
the prompt into a single user message. When the provider implements

```go
PrepareStreamRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error)
```

and the prompt has structured messages, `Stream` takes that path (injecting
`prompt.SystemPrompt` into `options["system_prompt"]`). Otherwise it falls back
to the flattened `PrepareStreamRequest(prompt.String(), options)`. Implemented
for the OpenAI provider (inherited by Google via embedding), Anthropic, the
OpenAI Responses API, and OpenRouter.

**System / developer prompt routing.** On the structured path each provider
places the system prompt in its *native* field (so it is never just prepended to
user text):

| Provider | System prompt lands in |
|---|---|
| OpenAI (Chat Completions) | a leading message with role `developer` |
| Google (OpenAI-compatible) | a leading message with role `system`, which Gemini's compat layer maps to its native `system_instruction` |
| DeepSeek (OpenAI-compatible) | a leading message with role `system` (DeepSeek doesn't recognize `developer`) |
| Anthropic | top-level `system` field (with `cache_control` when caching) |
| OpenAI Responses | top-level `instructions` field |

The role each provider uses is set by `OpenAIProvider.systemRole` (default
`developer`; overridden to `system` by the Google and DeepSeek constructors).

This is intentionally different from the **flattened** path: when the prompt has
no structured messages, `PrepareStreamRequest` collapses everything (including
the system prompt, via `Prompt.String()`'s `"System: …"` prefix) into a single
user turn. Use structured messages whenever the system prompt must be honored as
a first-class system/developer instruction.

### 2. Tools in the request

`Stream` carries `prompt.Tools` into `options["tools"]`, plus
`prompt.ToolChoice` into `options["tool_choice"]` and `prompt.Images` into
`options["images"]`, mirroring the blocking path. Without these, the provider
never receives the tool definitions (so it can't emit tool calls), can't honor a
forced `tool_choice`, and drops image inputs for vision models. `ToolChoice` is a
`map[string]interface{}{"type": <strategy>}`; it is normalized to the bare
strategy string (`"auto"`/`"required"`/`"any"`/`"none"`) that providers expect —
OpenAI uses it directly, Anthropic re-wraps it as `{"type": …}`. Tool-calling over
a stream therefore "just works" for any provider whose
`PrepareRequestWithMessages` already serializes `options["tools"]`.

### 3. Usage + finish reason (rich parsing)

The optional capability:

```go
ParseStreamResponseRich(chunk []byte) (types.StreamChunk, error)
```

returns a provider-normalized `types.StreamChunk` (`types/stream.go`) instead of
plain text. `Next` prefers it when present and maps it to a `StreamToken`; when a
provider doesn't implement it, `Next` falls back to the text-only
`ParseStreamResponse`. `StreamChunk`:

```go
type StreamChunk struct {
    Text          string         // incremental text
    Kind          string         // "text" | "usage" | "finish" | "tool_call_delta"
    FinishReason  string         // provider stop/finish reason
    Usage         *TokenUsage    // token usage, when reported mid/end of stream
    ToolCallDelta *ToolCallDelta // incremental tool-call fragment
}
```

A single chunk may carry more than one signal (e.g. Anthropic's final
`message_delta` reports both stop reason and output tokens), so consumers should
read every populated field rather than switch on `Kind` alone.

**Reading the final tokens.** Usage can arrive split across the stream (Anthropic
reports input tokens on `message_start` and output tokens on the final
`message_delta`; OpenAI sends a `finish` token then a separate trailing `usage`
token). `Stream` **accumulates this for you**: it merges each chunk's usage
(latest-non-zero per field, matching Anthropic's cumulative semantics) and emits
the running total on every `usage`/`finish` token, synthesizing `total_tokens`
from prompt+completion when the provider doesn't report it. So the metadata on the
last `usage`/`finish` token you receive carries the complete totals — no manual
summing required. All surfaces emit the stop/finish reason on a `Kind == "finish"`
token, so switching on `finish` is sufficient to detect completion.

### 4. Tool-call deltas

A streamed tool call arrives across many chunks. The rich parser stays stateless
per-chunk and emits fragments; the consumer assembles them. A fragment is:

```go
type ToolCallDelta struct {
    Index        int    // which call this fragment belongs to (provider slot)
    ID           string // call id; set on the opening fragment, empty after
    Name         string // function name; set on the opening fragment, empty after
    ArgsFragment string // partial JSON of the arguments to append for this Index
}
```

Assembly: keep a map keyed by `Index`; set `ID`/`Name` when non-empty, append
each `ArgsFragment`. When the stream reaches its finish/usage chunk, the
concatenated `ArgsFragment` pieces per `Index` form the full arguments JSON.

## Per-provider wire handling

All five surfaces support structured streaming + usage + finish + tool-call
deltas. Coverage is verified in `providers/stream_rich_test.go` and
`providers/stream_messages_test.go`.

| Provider | Text | Usage | Finish | Tool-call deltas |
|---|---|---|---|---|
| OpenAI (Chat Completions) | `delta.content` | trailing `choices:[] + usage` (via `stream_options.include_usage`) | `finish_reason` (does **not** end the stream — usage chunk follows) | `delta.tool_calls[]` (index, id, name, partial `arguments`) |
| Google (OpenAI-compatible) | inherited from OpenAI via embedding | inherited | inherited | inherited |
| Anthropic | `content_block_delta` / `text_delta` | `message_start` (input) + `message_delta` (output) | `message_delta.stop_reason` | `content_block_start[tool_use]` (id+name) then `content_block_delta` / `input_json_delta` (partial JSON), keyed by event `index` |
| OpenAI Responses API | `response.output_text.delta` | terminal event `response.completed` / `.incomplete` (carries `response.usage`; no `stream_options`) | `.completed`/`.incomplete` → `finish` token (`response.status`); `.failed` and top-level `error` events → surfaced as an error | `response.output_item.added[function_call]` (id+name) then `response.function_call_arguments.delta`, keyed by `output_index` |
| OpenRouter | `delta.content` | trailing `usage` chunk (via `usage:{include:true}`) | `finish_reason` (does **not** end the stream) | `delta.tool_calls[]` (index, id, name, partial `arguments`) |

OpenRouter is a standalone (non-embedding) provider; it implements
`PrepareStreamRequestWithMessages` (prepending the system prompt as a leading
`system` message, since its message builder doesn't inject one) and
`ParseStreamResponseRich` directly. It honors the same `stream_usage` opt-out,
mapping it to OpenRouter's native `usage:{include:true}` request flag.

**Note on OpenAI/Google usage:** the OpenAI provider sets
`stream_options.include_usage` so the stream ends with a meterable usage chunk.
Google rides this through its OpenAI-compatible endpoint (inherited via
embedding); confirm against a live Google key that the trailing usage chunk
arrives. If an OpenAI-compatible gateway **rejects** the `stream_options`
parameter (some self-hosted/proxy endpoints do), disable it per call or via
defaults:

```go
llm.SetOption("stream_usage", false) // omit stream_options.include_usage
```

The `stream_usage` control key is consumed before the request is built, so it
never leaks into the wire payload. With it off you lose the trailing usage chunk
and must count tokens locally.

## Resource safety

- `providerStream.Close()` closes the underlying HTTP response body. Always
  `defer stream.Close()` — skipping it leaks a connection per stream.
- The SSE decoder's line buffer is raised to `DefaultSSEMaxLineSize` (1 MB) so
  large SSE lines (e.g. streamed tool-call arguments) don't fail mid-stream. If
  you stream even larger single lines, pass `llm.WithMaxLineSize(n)` to `Stream`
  — it's a per-stream `StreamConfig` field, so it's race-free (no process-global).
- `Next` surfaces transport/decoder errors **and** genuine parser/in-band API
  errors (e.g. an OpenRouter `{"error":…}` chunk) directly to the caller. Parsers
  signal a non-emitting chunk with the `types.ErrStreamSkip` sentinel, which the
  loop skips; any other error ends the stream with that error rather than a silent
  EOF. `Next` does **not** retry in place (the HTTP body is already partially
  consumed); to retry, close the stream and re-invoke `Stream`.
- `StreamConfig.RetryStrategy` governs retries while **establishing** the stream
  only — *before* any token is produced. Retries are limited to transient failures
  (transport errors and 408/429/5xx); non-transient 4xx (401/400/403/404) fail
  fast. This is idempotent (no output yet) and mirrors the blocking `Generate`
  retry loop, bounded by the LLM's `MaxRetries` / `RetryDelay`. Chat streams have
  no resumption point, so it is deliberately not consulted for mid-stream
  interruptions (see the previous bullet).

## Adding a provider

Plain text streaming requires only `PrepareStreamRequest` + `ParseStreamResponse`
(the existing `Provider` interface). To opt into the richer behaviour, implement
any of these optional methods — `Stream`/`Next` detect them by type assertion:

- `PrepareStreamRequestWithMessages(...)` — structured messages + system prompt.
- `ParseStreamResponseRich(...)` — usage, finish reason, and tool-call deltas.

No change to the `Provider` interface is required, so existing providers keep
working untouched.
