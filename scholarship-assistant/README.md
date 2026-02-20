# Scholarship Assistant

Voice-driven scholarship application assistant that builds a master profile, fills forms automatically, and tracks application history to avoid duplicates.

**Components:** Python FastAPI backend, Tampermonkey userscript. Uses DeepSeek for field matching and profile extraction, ElevenLabs for TTS, Whisper for STT.

---

## Setup

```bash
cd scholarship-assistant
pip install -r requirements.txt
```

**Environment:** Create a `.env` in the parent project root (`always-on-ai-assistant/`) with:

- `DEEPSEEK_API_KEY` — required for field matching and profile import
- `ELEVEN_API_KEY` — for voice output (optional if using `TEXT_INPUT_MODE=1`)
- `TEXT_INPUT_MODE=1` — type responses instead of speaking (avoids mic/audio setup issues)

**Profile & data:** Stored in `~/.scholarship-assistant/`:
- `profile.json` — your master profile
- `history.json` — application history (dedup)

---

## CLI

**Start the server** (default: `http://127.0.0.1:8741`):

```bash
python -m server.main
```

**Run the init interview** (voice or text-guided profile setup):

```bash
python -m server.main --init
```

With text-only mode (no mic/TTS):

```bash
TEXT_INPUT_MODE=1 python -m server.main --init
```

---

## Profile Import

Import content from `.txt` or `.pdf` files into your profile. DeepSeek parses and extracts profile updates. Supports:

- **Tab-separated Q&A** — `Question\tAnswer(s)\tLast Answered` (in .txt)
- **Free-form text** — raw pasted content (in .txt)
- **PDF documents** — e.g. resume/CV (text is extracted via PyMuPDF)

**Import and apply:**

```bash
python -m server.main --import-file /path/to/qa.txt
python -m server.main --import-file /path/to/resume.pdf
```

**Preview updates without applying:**

```bash
python -m server.main --import-file /path/to/resume.pdf --dry-run
```

The import adds to your existing profile and uses the `extended` section for data that doesn't fit the standard schema.

**Profile review:** After a successful import, a review phase runs automatically. DeepSeek analyzes the profile for duplicates, near-duplicates (e.g. "BS" vs "Bachelor of Science"), misplaced data, and inconsistencies—then applies cleanup. Skip with `--no-review`:

```bash
python -m server.main --import-file /path/to/qa.txt --no-review
```

**Review only** (clean up an existing profile without re-importing):

```bash
python -m server.main --review-profile
python -m server.main --review-profile --dry-run   # preview cleanup
```

---

## Userscript

Install `userscript/scholarship-assistant.user.js` in Tampermonkey. When on a scholarship form page, click the trigger to send fields to the backend and auto-fill from your profile.

---

## More

See [PLAN.md](PLAN.md) for architecture and design details.
