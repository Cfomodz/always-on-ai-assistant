# Plan: Unified Always-On Voice Assistant

## Phase 1: Unified Entry Point + Intent Router

**Goal:** Single process that defaults to conversation, switches to command execution when needed.

### 1a. Create `modules/router.py`

A lightweight DeepSeek classification call that takes the user's utterance and returns `"command"` or `"conversation"`. This is a small/fast prompt (~200ms) that looks at the available Typer commands and decides if the utterance is asking to execute one. Avoids brittle keyword heuristics.

### 1b. Create `main_unified.py`

Single entry point that:

- Starts RealtimeSTT (same config as current typer assistant -- tiny.en, beam_size=8, etc.)
- On each utterance: trigger word check -> router classification -> dispatch to TyperAgent or PlainAssistant
- Both agents share the same ElevenLabs TTS
- Recorder stops during LLM+TTS, restarts after (already done in both modes -- prevents self-hearing)

### 1c. Update `assistant_config.yml`

Add a `unified_assistant` section. Keep the two separate sections for backward compatibility.

### 1d. Create `jarvis.sh`

New launcher script for the unified mode. Keep `ada.sh` as-is.

---

## Phase 2: DevTools Commands

**Goal:** Read-only voice access to gh, Claude CLI, and Cursor.

### 2a. Create `commands/devtools.py`

New Typer commands file with:

**GitHub CLI wrappers (read-only):**

- `list-prs` -- `gh pr list`
- `view-pr` -- `gh pr view <number>`
- `list-issues` -- `gh issue list`
- `view-issue` -- `gh issue view <number>`
- `list-runs` -- `gh run list` (CI/CD)
- `view-run` -- `gh run view <id>`
- `repo-status` -- `gh repo view` + `git status`

**Claude CLI wrappers (read-only):**

- `list-claude-sessions` -- query recent Claude Code sessions
- `claude-status` -- check if Claude is running, what session

**Cursor Cloud Agents:**

Cursor doesn't expose a public CLI/API for cloud agent status. The practical option is checking agent-created PRs/branches via gh. Add a `list-agent-prs` command that filters PRs by bot/agent authors.

### 2b. Create `prompts/devtools-commands.xml`

Prompt template for the devtools command set (same pattern as `typer-commands.xml`).

### 2c. Update router

The router needs to know about both `commands/template.py` and `commands/devtools.py` to classify correctly.

---

## Phase 3: Reliability

### 3a. Verify dependencies install on Manjaro

Run `uv sync`, fix any issues with RealtimeSTT/CUDA/PyAudio.

### 3b. Fix the DeepSeek API endpoint

Currently hardcoded to `https://api.deepseek.com/beta`. Verify this is still correct or update.

### 3c. Upgrade Whisper model default

Current default is `tiny.en` which is fast but inaccurate. Since the target has NVIDIA+CUDA, bump to `small.en` for the unified mode (better accuracy, ~1.5s transcription, acceptable for conversation).

### 3d. Handle edge cases

- Self-hearing prevention is already handled (recorder.stop/start around TTS)
- Add the "ignoring own speech" check from PlainAssistant into the unified flow
- Graceful error recovery if DeepSeek API is down

---

## What Stays Unchanged

- All existing dependencies in `pyproject.toml`
- `commands/template.py` (SQLite demo commands)
- Both original entry points (`main_typer_assistant.py`, `main_base_assistant.py`)
- `modules/deepseek.py`, `modules/ollama.py`, `modules/base_assistant.py`, `modules/typer_agent.py` -- reused as-is

---

## Completed Work

- [x] **Web search module** (`modules/web_search.py`) -- DuckDuckGo-backed `search()`, `search_news()`, `get_answer()` functions. No API key required.
- [x] **Web search Typer commands** (`commands/template.py`) -- `web-search`, `web-news`, `web-answer` commands added.
- [x] **Dependency** -- `duckduckgo-search>=7.0.0` added to `pyproject.toml`.
- [x] **Phase 1: Unified Entry Point + Intent Router**
  - `modules/router.py` -- DeepSeek-based intent classifier (`command` vs `conversation`).
  - `main_unified.py` -- Single entry point: STT -> trigger word -> router -> TyperAgent or PlainAssistant.
  - `assistant_config.yml` -- Added `unified_assistant` section (name: Jarvis).
  - `jarvis.sh` -- Launcher script for unified mode.
- [x] **Phase 2: DevTools Commands**
  - `commands/devtools.py` -- Read-only Typer wrappers: `list-prs`, `view-pr`, `list-issues`, `view-issue`, `list-runs`, `view-run`, `repo-status`, `claude-status`, `list-agent-prs`.
  - `prompts/devtools-commands.xml` -- Prompt template for devtools command set.
  - `main_unified.py` -- Router now loads both `commands/template.py` and `commands/devtools.py`.
