# gollm — Claude Code Rules

## What This Is
Unified Go interface for LLM providers — simplifies multi-provider integration with flexible prompt management and common task helpers. Used across the magicform suite as the shared LLM client.

## Stack & Conventions
- **Go library** (module `github.com/teilomillet/gollm`). Public API in `gollm.go`, provider adapters and utilities under the root packages. `examples/` shows intended usage.
- Tests: `go test ./...`. No build step — consumed as a Go module.
- Public API is a dependency surface for every downstream service — breaking changes require coordination across the suite. Prefer additive changes.
- This repo is a nuestra-ai fork tracking an upstream; be conservative about restructuring that would conflict with upstream merges.

## Control Plane
Directives, skills, agent definitions, and the broader context tree live in the **agent-platform umbrella repo** at `../agent-platform/`. For cross-repo work or any `/directive` pipeline usage, open Claude from there:

```bash
cd ../agent-platform && claude
```

## Git Operations
NEVER perform git operations without explicit user approval.
