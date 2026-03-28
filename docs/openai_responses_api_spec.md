# Spec: OpenAI Responses API Support

**Date:** 2026-03-28
**Status:** Draft
**Scope:** Add support for OpenAI's Responses API (`/v1/responses`) alongside the existing Chat Completions API (`/v1/chat/completions`).

---

## Background

OpenAI's Responses API is their recommended API for new projects. It differs from Chat Completions in three fundamental ways:

1. **Built-in tools** — `web_search`, `file_search`, `code_interpreter`, `computer_use_preview`, and remote MCP servers are first-class tool types (not just function calling).
2. **Server-side conversation state** — `previous_response_id` lets the server manage history instead of the client resending all messages.
3. **Different request/response shapes** — `input` + `instructions` instead of `messages[]`; `output[]` items instead of `choices[]`.

### Current State

- All OpenAI requests go to `/v1/chat/completions`.
- Response parsing already handles the Responses API `output[]` format for web search results (`openai.go:601-716`, `parseWebSearchResponse`).
- `web_search` tools are currently **filtered out** in `PrepareRequest()` because Chat Completions doesn't support them.
- The `Provider` interface has 4 request preparation methods, all targeting Chat Completions shape.

### Fork Maintenance Constraint

This repo is a fork of `teilomillet/gollm`. Upstream actively modifies `providers/openai.go` (20+ commits touching it, 1389 lines). **Any design that injects branching logic into existing `openai.go` methods will create recurring merge conflicts.** The architecture must keep our changes in separate files that upstream doesn't touch.

---

## Design: Separate Provider via Composition

Instead of adding `if apiMode == "responses"` branches inside `OpenAIProvider`, we register a **separate `OpenAIResponsesProvider`** in its own file. This keeps `openai.go` untouched (or nearly so) and isolates all Responses API logic into files that upstream will never conflict with.

### File Layout

```
providers/
  openai.go                    ← UPSTREAM-OWNED — zero or minimal edits
  openai_responses.go          ← NEW — all Responses API request/response logic
  openai_responses_test.go     ← NEW — isolated tests
  openai_shared.go             ← NEW — helpers extracted from openai.go (option merging, model quirks)
  provider.go                  ← one line added to registry
```

### Provider Structure

```go
// openai_responses.go

type OpenAIResponsesProvider struct {
    apiKey       string
    model        string
    extraHeaders map[string]string
    options      map[string]interface{}
    logger       utils.Logger
}

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
```

This implements the full `Provider` interface independently. It reuses shared helpers (from `openai_shared.go`) but does **not** embed or wrap `OpenAIProvider` — embedding creates a fragile coupling where upstream struct changes silently break delegation.

### Shared Helpers (extracted to `openai_shared.go`)

The current `openai.go` has the same option-merging block copy-pasted across `PrepareRequest`, `PrepareRequestWithSchema`, `PrepareStreamRequest`, etc. (~50 lines × 4 methods). Extract into package-level functions:

```go
// openai_shared.go

// mergeOpenAIOptions combines provider defaults with per-request options,
// handling max_tokens ↔ max_completion_tokens conversion, temperature
// filtering for o3/gpt-5, reasoning_effort gating, etc.
func mergeOpenAIOptions(model string, providerOpts, requestOpts map[string]interface{}, excludeKeys []string) map[string]interface{}

// Re-export existing unexported helpers that both providers need:
// - modelNeedsNoTemperature(model) bool     ← already exists
// - modelNeedsNoToolChoice(model) bool      ← already exists
// - normalizeSchema(schema) (interface{}, error)  ← already exists in provider.go
// - cleanSchemaForOpenAI(schema) interface{}      ← already exists
// - ConvertImagesToOpenAIContent(images)           ← already exists (exported)
```

**Upstream merge impact:** Extracting these helpers is a refactor of `openai.go` that *removes* duplication. It's the kind of change that's easy to resolve if upstream also changes the same blocks — the conflict resolution is "keep the extracted version." This is the **only** edit to `openai.go`: replace inline option-merging with calls to the shared helper. This could also be deferred to a separate upstream-friendly PR.

### Registry Registration

One line in `provider.go`:

```go
// In NewProviderRegistry(), add to knownProviders:
"openai-responses": NewOpenAIResponsesProvider,
```

And the matching config:

```go
"openai-responses": {
    Name:              "openai-responses",
    Type:              TypeOpenAI,
    Endpoint:          "https://api.openai.com/v1/responses",
    AuthHeader:        "Authorization",
    AuthPrefix:        "Bearer ",
    RequiredHeaders:   map[string]string{"Content-Type": "application/json"},
    SupportsSchema:    true,
    SupportsStreaming: true,
},
```

### User-Facing Selection

Two ways to activate:

```go
// Explicit provider name
llm, _ := gollm.NewLLM(
    gollm.SetProvider("openai-responses"),
    gollm.SetModel("gpt-4o"),
)

// Or via convenience option (syntactic sugar — rewrites provider name internally)
llm, _ := gollm.NewLLM(
    gollm.SetProvider("openai"),
    gollm.SetModel("gpt-4o"),
    gollm.WithOpenAIResponsesAPI(),  // sets cfg.Provider = "openai-responses"
)
```

`WithOpenAIResponsesAPI()` is a one-line config option in `config/config.go` that swaps the provider name. No magic, no auto-upgrade — explicit opt-in only.

---

## Merge Conflict Analysis

| File | Upstream touches? | Our changes | Conflict risk |
|---|---|---|---|
| `providers/openai.go` | **Frequently** | Extract option-merging to shared helper (optional, deferrable) | Low if deferred; medium if done |
| `providers/openai_responses.go` | **Never** (new file) | All Responses API logic | **None** |
| `providers/openai_shared.go` | **Never** (new file) | Shared helpers | **None** |
| `providers/openai_responses_test.go` | **Never** (new file) | Tests | **None** |
| `providers/provider.go` | Occasionally (new providers) | 2 lines: registry entry + config | **Low** — additive, rarely conflicts |
| `config/config.go` | Occasionally | Add `WithOpenAIResponsesAPI()` option | **Low** — additive |
| `gollm.go` | Rarely | Wire convenience option | **Low** |
| `llm/llm.go` | Occasionally | No changes in Phase 1 | **None** |
| `utils/shared_types.go` | Occasionally | Extend `Tool` struct | **Low** |
| `types/` | Rarely | New types in new files | **None** |

**Net result:** 80%+ of our code lives in files upstream doesn't know about.

---

## Phase 1: Core Request/Response (MVP)

### 1.1 Request Building

`OpenAIResponsesProvider.PrepareRequest()` builds the `/v1/responses` payload:

**Chat Completions shape (current `openai.go` — unchanged):**
```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "temperature": 0.7,
  "max_completion_tokens": 4096,
  "response_format": {"type": "json_schema", "json_schema": {...}}
}
```

**Responses API shape (new `openai_responses.go`):**
```json
{
  "model": "gpt-4o",
  "input": "..." ,
  "instructions": "...",
  "temperature": 0.7,
  "max_output_tokens": 4096,
  "text": {"format": {"type": "json_schema", "name": "structured_response", "schema": {...}}}
}
```

For multi-turn (`PrepareRequestWithMessages`), `input` becomes an item list:
```json
{
  "model": "gpt-4o",
  "input": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "Follow up question"}
  ],
  "instructions": "..."
}
```

**Key parameter mappings:**

| Chat Completions | Responses API | Notes |
|---|---|---|
| `messages[]` | `input` (string or item list) | String for simple prompts, item list for multi-turn |
| `messages[role=system]` | `instructions` | Top-level field, not in input array |
| `messages[role=developer]` | `instructions` | Same — maps to `instructions` |
| `max_tokens` / `max_completion_tokens` | `max_output_tokens` | Name change |
| `response_format` | `text.format` | Nested under `text` object |
| `tools[]` (function only) | `tools[]` (function + built-in) | Built-in tools pass through directly |
| `tool_choice` | `tool_choice` | Same semantics |
| N/A | `previous_response_id` | Phase 2 |
| N/A | `store` | Phase 2 |

### 1.2 Provider Methods to Implement

The `OpenAIResponsesProvider` implements all `Provider` interface methods:

```go
// Identity & config
func (p *OpenAIResponsesProvider) Name() string              // "openai-responses"
func (p *OpenAIResponsesProvider) Endpoint() string           // "https://api.openai.com/v1/responses"
func (p *OpenAIResponsesProvider) Headers() map[string]string // Same auth as OpenAI
func (p *OpenAIResponsesProvider) SupportsJSONSchema() bool   // true
func (p *OpenAIResponsesProvider) SupportsStreaming() bool    // true

// Request preparation — Responses API format
func (p *OpenAIResponsesProvider) PrepareRequest(prompt string, options map[string]interface{}) ([]byte, error)
func (p *OpenAIResponsesProvider) PrepareRequestWithSchema(prompt string, options map[string]interface{}, schema interface{}) ([]byte, error)
func (p *OpenAIResponsesProvider) PrepareRequestWithMessages(messages []types.MemoryMessage, options map[string]interface{}) ([]byte, error)
func (p *OpenAIResponsesProvider) PrepareRequestWithMessagesAndSchema(messages []types.MemoryMessage, options map[string]interface{}, schema interface{}) ([]byte, error)

// Response parsing — handles output[] format
func (p *OpenAIResponsesProvider) ParseResponse(body []byte) (string, error)
func (p *OpenAIResponsesProvider) ParseResponseWithUsage(body []byte) (string, *types.ResponseDetails, error)

// Streaming — Responses API SSE events
func (p *OpenAIResponsesProvider) PrepareStreamRequest(prompt string, options map[string]interface{}) ([]byte, error)
func (p *OpenAIResponsesProvider) ParseStreamResponse(chunk []byte) (string, error)

// Options & misc
func (p *OpenAIResponsesProvider) SetOption(key string, value interface{})
func (p *OpenAIResponsesProvider) SetDefaultOptions(config *config.Config)
func (p *OpenAIResponsesProvider) SetExtraHeaders(extraHeaders map[string]string)
func (p *OpenAIResponsesProvider) SetLogger(logger utils.Logger)
func (p *OpenAIResponsesProvider) HandleFunctionCalls(body []byte) ([]byte, error)
```

### 1.3 Response Parsing

The Responses API returns `output[]` items instead of `choices[]`. Each item is type-tagged:

```json
{
  "id": "resp_abc123",
  "status": "completed",
  "model": "gpt-4o",
  "output": [
    {
      "type": "message",
      "role": "assistant",
      "content": [{"type": "output_text", "text": "The answer is..."}]
    }
  ],
  "usage": {
    "input_tokens": 50,
    "output_tokens": 100,
    "total_tokens": 150
  }
}
```

Other output item types to handle:

| Output type | Content | Action |
|---|---|---|
| `message` | `content[].text` | Extract text, annotations, citations |
| `function_call` | `name`, `arguments`, `call_id` | Format as function call string (same as today) |
| `web_search_call` | `action.query`, `action.sources` | Delegate to existing `parseWebSearchResponse` |
| `file_search_call` | `results[]` | Store in metadata |
| `reasoning` | `summary[]` | Store in metadata (or discard) |

The existing `parseWebSearchResponse` helper can be reused directly — it already lives in the `providers` package and handles the `output[]` shape.

**Usage field mapping:**

| Chat Completions | Responses API |
|---|---|
| `usage.prompt_tokens` | `usage.input_tokens` |
| `usage.completion_tokens` | `usage.output_tokens` |
| `usage.total_tokens` | `usage.total_tokens` |

### 1.4 Built-in Tool Support

In the Responses API, built-in tools are first-class — they go in `tools[]` alongside function tools but with different shapes:

```go
// Function tool (wrapped, same concept as Chat Completions)
{"type": "function", "name": "get_weather", "description": "...", "parameters": {...}}

// Built-in tools (passed directly, no function wrapper)
{"type": "web_search"}
{"type": "web_search", "search_context_size": "medium", "user_location": {"type": "approximate", "city": "NYC"}}
{"type": "file_search", "vector_store_ids": ["vs_123"]}
{"type": "code_interpreter", "container": {"type": "auto"}}
{"type": "computer_use_preview", "display_width": 1024, "display_height": 768, "environment": "browser"}
{"type": "mcp", "server_label": "my-server", "server_url": "https://...", "headers": {"Authorization": "Bearer ..."}}
```

Extend `utils.Tool` to carry built-in tool parameters:

```go
// In utils/shared_types.go — extend existing Tool struct
type Tool struct {
    Type     string   `json:"type"`
    Function Function `json:"function,omitempty"`

    // Existing web search fields (already present)
    Filters           interface{} `json:"filters,omitempty"`
    UserLocation      interface{} `json:"user_location,omitempty"`
    ExternalWebAccess interface{} `json:"external_web_access,omitempty"`
    // ...existing fields...

    // NEW: Responses API built-in tool parameters
    SearchContextSize string      `json:"search_context_size,omitempty"` // web_search: "low", "medium", "high"
    VectorStoreIDs    []string    `json:"vector_store_ids,omitempty"`    // file_search
    Container         interface{} `json:"container,omitempty"`           // code_interpreter
    DisplayWidth      int         `json:"display_width,omitempty"`       // computer_use
    DisplayHeight     int         `json:"display_height,omitempty"`      // computer_use
    Environment       string      `json:"environment,omitempty"`         // computer_use
    ServerLabel       string      `json:"server_label,omitempty"`        // mcp
    ServerURL         string      `json:"server_url,omitempty"`          // mcp
    ServerHeaders     map[string]string `json:"server_headers,omitempty"` // mcp
}
```

The `OpenAIResponsesProvider` formats these into the correct JSON shape. `OpenAIProvider` (Chat Completions) continues to ignore/filter non-function tools — no change needed there.

### 1.5 Files Changed (Phase 1)

| File | Change | Upstream conflict risk |
|---|---|---|
| `providers/openai_responses.go` | **NEW** — all Responses API logic (~500-700 lines) | None |
| `providers/openai_shared.go` | **NEW** — extracted option-merging, model quirk helpers | None |
| `providers/openai_responses_test.go` | **NEW** — unit tests | None |
| `providers/provider.go` | Add `"openai-responses"` to registry + config (2 blocks) | Low |
| `config/config.go` | Add `WithOpenAIResponsesAPI()` config option | Low |
| `gollm.go` | Export `WithOpenAIResponsesAPI` convenience alias | Low |
| `utils/shared_types.go` | Add built-in tool fields to `Tool` struct | Low |
| `types/responses.go` | **NEW** — Responses API-specific types | None |
| `providers/openai.go` | **Optional/deferrable** — replace inline option-merging with shared helper calls | Medium if done, none if deferred |

### 1.6 Deferring the `openai.go` Refactor

The option-merging extraction from `openai.go` → `openai_shared.go` is **nice-to-have** but not required for Phase 1. Without it, `OpenAIResponsesProvider` duplicates the merging logic (copying it into the new file). This is acceptable because:

- The duplication is isolated to one new file we fully own
- It avoids touching `openai.go` entirely in Phase 1
- It can be done later as a standalone cleanup PR (potentially upstream-contributed)

**Recommendation:** Defer the `openai.go` refactor. Copy the option-merging logic into `openai_shared.go` as standalone functions. Later, update `openai.go` to call those same functions — at which point the duplication is resolved.

---

## Phase 2: Server-Side Conversation State

### 2.1 `previous_response_id` Support

Add option plumbing so callers can pass or receive response IDs:

```go
// Set via option on the prompt
prompt := gollm.NewPrompt("follow up question",
    gollm.WithPreviousResponseID("resp_abc123"),
)

// Or via provider option
llm.SetOption("previous_response_id", "resp_abc123")
```

The response ID comes back in `ResponseDetails.ID` (already parsed from the `id` field).

### 2.2 Memory Integration

When using `openai-responses` provider with `store == true`:
- `PrepareRequestWithMessages` can send `previous_response_id` instead of the full message history.
- The `Memory` system stores response IDs alongside messages.
- Fallback: if server-side state is lost (404), resend full history.

New `Memory` methods:
```go
func (m *ConversationMemory) LastResponseID() string
func (m *ConversationMemory) SetLastResponseID(id string)
```

### 2.3 `store` Parameter

```go
gollm.WithStore(true) // persist conversation server-side
```

Defaults to `false` to match current client-side-only behavior.

### 2.4 Files Changed (Phase 2)

| File | Change | Upstream conflict risk |
|---|---|---|
| `providers/openai_responses.go` | Include `previous_response_id` and `store` in requests | None |
| `llm/memory.go` | Add `LastResponseID` / `SetLastResponseID` | Low |
| `llm/llm.go` | Pass response ID from `ResponseDetails` back to memory after each call | Low |
| `config/config.go` | Add `Store` config option | Low |
| `gollm.go` | Wire `WithStore`, `WithPreviousResponseID` | Low |

---

## Phase 3: Streaming

### 3.1 Different SSE Event Format

Chat Completions streaming: `data: {"choices": [{"delta": {"content": "token"}}]}`

Responses API streaming uses typed events:
```
event: response.created
data: {"id": "resp_...", "status": "in_progress", ...}

event: response.output_text.delta
data: {"delta": "token text"}

event: response.completed
data: {"id": "resp_...", "status": "completed", "output": [...], "usage": {...}}
```

### 3.2 Implementation

Since `OpenAIResponsesProvider` is a separate type, it has its own `PrepareStreamRequest` and `ParseStreamResponse` methods — no branching needed in the existing OpenAI streaming code.

- `PrepareStreamRequest` adds `"stream": true` to the Responses API payload.
- `ParseStreamResponse` handles event-typed SSE chunks, extracting text from `response.output_text.delta` events and signaling completion on `response.completed`.

**Potential issue:** The `llm.go` streaming loop (`streamGenerate`) may make assumptions about chunk format. If so, the `Provider` interface's `ParseStreamResponse` should abstract those differences — which it already does by returning `(string, error)`. As long as `ParseStreamResponse` returns text deltas and `io.EOF` on completion, `llm.go` shouldn't need changes.

### 3.3 Files Changed (Phase 3)

| File | Change | Upstream conflict risk |
|---|---|---|
| `providers/openai_responses.go` | Implement `PrepareStreamRequest`, `ParseStreamResponse` | None |
| `llm/llm.go` | Likely no changes (verify SSE parsing handles event-typed format) | None |

---

## Phase 4: Advanced Built-in Tools

### 4.1 `computer_use`

Requires multi-turn loop: model returns `computer_call` output → client executes action → sends screenshot back as `computer_call_output`. This needs a callback/handler pattern:

```go
type ComputerUseHandler interface {
    ExecuteAction(action ComputerAction) (screenshot []byte, err error)
}

gollm.WithComputerUseHandler(handler)
```

The LLM engine would loop internally until the model stops issuing computer calls.

### 4.2 Remote MCP Servers

```json
{"type": "mcp", "server_label": "my-server", "server_url": "https://...", "headers": {...}}
```

Passthrough in `tools[]` — no special client-side handling needed beyond formatting.

### 4.3 `code_interpreter`

Returns `code_interpreter_call` output items with `code` and `results[]`. Parse and surface in `ResponseDetails.Metadata`.

### 4.4 Files Changed (Phase 4)

| File | Change | Upstream conflict risk |
|---|---|---|
| `providers/openai_responses.go` | Format MCP, computer_use, code_interpreter tools; parse their output item types | None |
| `llm/llm.go` | Add computer-use loop in `attemptGenerate` | Medium |
| `types/responses.go` | Add types: `ComputerAction`, `CodeInterpreterResult`, `MCPToolConfig` | None |
| `gollm.go` | New options: `WithComputerUseHandler`, `WithMCPServer` | Low |

---

## What Does NOT Change

- **`Provider` interface** — no new interface methods. `OpenAIResponsesProvider` implements the existing interface.
- **`providers/openai.go`** — zero edits in Phase 1 (option-merging refactor is deferred). Chat Completions provider is untouched.
- **Other providers** — Anthropic, Groq, Ollama, etc. are unaffected.
- **Default behavior** — `SetProvider("openai")` still uses Chat Completions. Responses API requires explicit `SetProvider("openai-responses")` or `WithOpenAIResponsesAPI()`.
- **`GenericProvider`** with `TypeOpenAI` — continues to use Chat Completions. Responses API is OpenAI-specific.

---

## Testing Strategy

### Unit Tests (in `openai_responses_test.go`)
- `TestResponsesPrepareRequest` — verify request shape for simple prompt.
- `TestResponsesPrepareRequestWithSchema` — verify `text.format` structure.
- `TestResponsesPrepareRequestWithMessages` — verify `input` item list format.
- `TestResponsesParseResponse` — verify parsing of `output[]` with message items.
- `TestResponsesParseFunctionCall` — verify parsing of function_call output items.
- `TestResponsesParseWebSearch` — verify reuse of `parseWebSearchResponse`.
- `TestResponsesEndpoint` — verify `/v1/responses` URL.
- `TestResponsesOptionMerging` — verify `max_output_tokens` naming, temperature/reasoning_effort handling.
- `TestResponsesBuiltinTools` — verify web_search, file_search tools are passed through.

### Integration Tests
- Round-trip with live OpenAI API using `openai-responses` provider (gated behind `OPENAI_API_KEY` env var).
- `previous_response_id` chaining across two calls (Phase 2).
- Web search with Responses API.
- Streaming with Responses API events (Phase 3).

### Backward Compatibility
- All existing tests pass unchanged — `openai.go` is not modified.
- `examples/web_search/` continues to work with Chat Completions provider.

---

## Migration Path for Users

```go
// Before (Chat Completions — still works, no changes needed)
llm, _ := gollm.NewLLM(
    gollm.SetProvider("openai"),
    gollm.SetModel("gpt-4o"),
)

// After (opt in to Responses API — option A: explicit provider)
llm, _ := gollm.NewLLM(
    gollm.SetProvider("openai-responses"),
    gollm.SetModel("gpt-4o"),
)

// After (opt in to Responses API — option B: convenience option)
llm, _ := gollm.NewLLM(
    gollm.SetProvider("openai"),
    gollm.SetModel("gpt-4o"),
    gollm.WithOpenAIResponsesAPI(),
)

// Using built-in tools
prompt := gollm.NewPrompt("What is the weather in NYC?",
    gollm.WithTools([]utils.Tool{
        {Type: "web_search", SearchContextSize: "medium"},
    }),
)

// Using server-side conversation state (Phase 2)
llm, _ := gollm.NewLLM(
    gollm.SetProvider("openai-responses"),
    gollm.SetModel("gpt-4o"),
    gollm.WithStore(true),
)
resp1, _ := llm.Generate(ctx, gollm.NewPrompt("Hello"))
// subsequent calls automatically chain via previous_response_id
resp2, _ := llm.Generate(ctx, gollm.NewPrompt("Follow up"))
```

---

## Open Questions

1. **Azure OpenAI** — Does the Azure OpenAI endpoint support the Responses API? If not, there should be no `azure-openai-responses` provider. Needs investigation.

2. **Response ID persistence** — Should response IDs survive process restarts? If so, the memory system needs serialization support for response IDs. Current memory is in-process only.

3. **`store` default** — OpenAI's default for `store` is `true` in their API. Should gollm default to `true` (match OpenAI) or `false` (minimize side effects)? Recommendation: `false`.

4. **Shared helper timing** — When to do the `openai.go` option-merging refactor? Options: (a) never — accept duplication, (b) Phase 1 — do it alongside the new provider, (c) separate PR first — upstream-friendly cleanup that stands alone. Recommendation: (c), but don't block Phase 1 on it.

5. **`parseWebSearchResponse` reuse** — This helper currently lives as a method on `*OpenAIProvider`. To call it from `OpenAIResponsesProvider`, either: (a) extract to a package-level function, or (b) duplicate it. Recommendation: (a) — it's a small, safe refactor that changes a method receiver to a function parameter.
