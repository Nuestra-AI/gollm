# Contributing — Nuestra AI Fork

This file is specific to `Nuestra-AI/gollm` and has no upstream counterpart.
Upstream's own guidance lives in [`CONTRIBUTING.md`](CONTRIBUTING.md), which this
fork leaves untouched so it can never conflict on merge. When the two disagree
about fork workflow, this file wins; for library style and design questions,
upstream's still applies.

## What this fork is

`gollm` is the shared LLM client across the magicform suite. This fork is a fork
of [`teilomillet/gollm`](https://github.com/teilomillet/gollm) carrying changes we
need ahead of upstream — streaming parity, usage accounting, reasoning-effort
support, and provider fixes.

## Branch layout

| Branch | Meaning |
| --- | --- |
| `main` | Our fork. All work lands here. |
| `public-main` | Mirror of upstream `main` at the last baseline refresh. Never commit to it. |

`public-main` is published on `origin`, so every clone can read the fork delta
**without configuring an `upstream` remote**.

## Seeing the fork delta

```bash
git log public-main..main            # commits this fork adds on top of upstream
git diff public-main main            # full diff against upstream
git diff public-main main -- <path>  # diff a single file or directory
```

Useful variants:

```bash
git rev-list --left-right --count public-main...main   # behind / ahead counts
git diff --stat public-main main                       # per-file change summary
```

On a fresh clone, materialize the local branch once:

```bash
git branch public-main origin/public-main
```

## Making changes

1. Branch from `main`.
2. Keep changes additive. The public API is a dependency surface for every
   downstream magicform service — breaking changes require coordination across
   the suite.
3. Run `go test ./...` before opening a PR. There is no build step; the module is
   consumed directly.
4. Open the PR against `Nuestra-AI/gollm` `main`.

`main` is protected. Landing a change requires:

- a pull request — direct pushes are rejected;
- **1 approving review** (stale approvals are dismissed when you push again);
- the **`Analyze (go)`** check passing.

Note that `Analyze (go)` is CodeQL. There is no test workflow, so `go test ./...`
is **not** enforced by CI — run it yourself before pushing.

Org admins bypass all of the above, which is how the upstream-baseline refresh
below is possible.

### Conventions that reduce upstream merge pain

Be conservative about restructuring. Every file we reorganize is a file that
conflicts on the next upstream merge.

- Prefer new files over edits to existing upstream files where it is reasonable.
- Prefer appending to existing files over rewriting or reordering them.
- Keep fork-only documentation in fork-only filenames — the `*.nuestra.md`
  suffix, or a new file under `docs/`. Do not edit upstream's `README.md`,
  `CONTRIBUTING.md`, or `AGENTS.md` unless the change is genuinely required.
- Match the surrounding code's naming, comment density, and idiom.

## Push safety

Local git config in this repo prevents accidental pushes to upstream:

- `remote.upstream.pushurl=DISABLED_use_origin` — `git push upstream` fails fast.
- `remote.pushDefault=origin`, `push.default=simple` — a branch tracking upstream
  still pushes to `origin`.
- `gh repo set-default Nuestra-AI/gollm` — `gh pr create` and `gh issue` target
  the fork.

`git config` and `gh` are independent: the git settings do not affect where
`gh pr create` opens a PR, and the `gh` default does nothing for `git push`.
Both are needed.

These are `--local`, so a fresh clone does **not** inherit them. If you clone this
repo and add an `upstream` remote, re-apply them:

```bash
git remote set-url --push upstream DISABLED_use_origin
git config remote.pushDefault origin
git config push.default simple
gh repo set-default Nuestra-AI/gollm
```

### Branch protection on GitHub

Server-side protection backs up the local config. The two branches are protected
differently, because they need opposite things:

| Branch | Protection | Effect |
| --- | --- | --- |
| `main` | Ruleset `protect-main` | PR + 1 approval + `Analyze (go)`; no force-push; no deletion. |
| `public-main` | Classic, `lock_branch` | Read-only: no commits at all. Force-push **allowed**; no deletion. |

`public-main` allows force-pushes and forbids commits — the exact inverse of
`main`. That is deliberate: the mirror's integrity is threatened by someone
committing their own work to it, not by force-updates, and the refresh above
*is* a force-update.

Admins bypass both. The protection is a guardrail against accident, not a
guarantee — nothing stops an admin from committing directly to `public-main`.
The invariant "`public-main` equals upstream `main`" is upheld by convention
here, not enforced by CI.

## Refreshing the upstream baseline

`public-main` is pinned — it does not advance on its own. Merge or rebase upstream
into `main` **first**, then move the mirror:

```bash
git remote add upstream https://github.com/teilomillet/gollm.git   # first time only
git fetch upstream
git branch -f public-main upstream/main
git push --force origin public-main
```

`git branch -f` moves the local pointer; the force-push updates the shared mirror
so the delta commands stay accurate for everyone else.

`--force` is required. `public-main` is a locked, read-only branch (see below), so
an ordinary `git push` is rejected. The force-push is also genuinely necessary
whenever upstream rebases or amends its own `main` — the mirror then has to move
non-linearly to keep matching, which a fast-forward push cannot do.

Moving `public-main` without integrating the upstream changes into `main` makes
`git log public-main..main` understate the delta and makes `main` read as behind
upstream.

The `upstream` remote is optional — it is only needed to *refresh* the baseline,
never to read the delta.

## Contributing back to upstream

If a change is genuinely general-purpose, send it upstream rather than carrying
it here forever. Open that PR from a branch pushed to a personal fork of
`teilomillet/gollm`, not from this repository — pushes to upstream are disabled
here by design.

## Git operations

Do not perform git operations — commits, pushes, branch changes — without
explicit approval from the repository owner. This applies to automated agents
working in this repo as well; see [`CLAUDE.md`](CLAUDE.md).
