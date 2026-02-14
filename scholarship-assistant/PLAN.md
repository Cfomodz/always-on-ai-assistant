# ScholarShip Assistant — Project Plan

## Overview

A voice-driven, accessibility-focused scholarship application assistant that eliminates repetitive form-filling. The system consists of a Python backend API and a Tampermonkey userscript. On click, the userscript sends the current page's form fields to the backend, which uses DeepSeek (JSON mode) to match fields against a master profile, auto-fills high-confidence matches, voice-confirms medium-confidence matches, and voice-asks the user for unknowns — adding every new answer to the profile permanently. Essays and open-ended fields are handled via voice transcription only (no AI writing). A deduplication system tracks application history to prevent applying to the same scholarship twice across different sites.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Browser (Tampermonkey Userscript)                       │
│  - Trigger button (floating, click to activate)          │
│  - Scrapes all form fields (label, type, options, id)    │
│  - POSTs field data to local backend                     │
│  - Receives fill instructions, applies them to DOM       │
│  - Copies essay text to clipboard if field unfillable     │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP (localhost:8741)
┌──────────────────────▼──────────────────────────────────┐
│  Python Backend (FastAPI)                                │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Profile Mgr  │  │ Field Matcher │  │ History Tracker│  │
│  │ (JSON store) │  │ (DeepSeek)   │  │ (dedup engine) │  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐                      │
│  │ Voice Out    │  │ Voice In     │                      │
│  │ (ElevenLabs)│  │ (faster-     │                      │
│  │              │  │  whisper)    │                      │
│  └─────────────┘  └──────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

### Reuse from Parent Project

This module lives under `scholarship-assistant/` and is invoked independently, but leverages the parent project's infrastructure:

| What | Source | How |
|---|---|---|
| DeepSeek API client | `modules/deepseek.py` | Import `json_prompt()`, `conversational_prompt()`, `prefix_prompt()` |
| ElevenLabs TTS | `modules/base_assistant.py` | Reuse ElevenLabs client pattern with `ELEVEN_API_KEY` |
| Config pattern | `modules/assistant_config.py` | Add `scholarship_assistant` section to `assistant_config.yml` |
| Logging | `modules/utils.py` | Import `setup_logging()`, `create_session_logger_id()` |
| Environment | `.env` / `python-dotenv` | Shares `DEEPSEEK_API_KEY`, `ELEVEN_API_KEY` |
| STT | `RealtimeSTT` (already in deps) | Reuse `AudioToTextRecorder` pattern from main assistants |

---

## Components

### 1. Profile Manager

**Storage:** `~/.scholarship-assistant/profile.json`

**Structure:** A flat-ish JSON document organized by category. Every field has a canonical key and a list of known phrasings (built up over time as DeepSeek encounters new question wordings).

**Profile categories and fields (init interview covers all of these):**

- **Personal**
  - Full legal name, preferred name, pronouns
  - Date of birth, age
  - SSN last 4 (optional, prompted with explanation)
  - Phone, email, mailing address
  - Citizenship, residency status, state of residence
  - Race/ethnicity, gender identity, marital status
  - Veteran status

- **Disability**
  - Disability status (yes/no)
  - Disability type(s), specific conditions
  - Accommodations received/requested
  - Documentation status

- **Education — Current**
  - University name, campus
  - Degree type, major(s), minor(s), concentration
  - Expected enrollment date, expected graduation date
  - Full-time/part-time status
  - Current GPA (if transfer credits exist)
  - Student ID

- **Education — History**
  - Prior institutions (name, dates, degree, GPA, credits earned)
  - High school (name, graduation year, GPA, class rank)
  - Standardized tests (SAT, ACT, GRE, etc. — test name, date, scores by section)
  - Certifications (name, issuing body, date, expiration)
  - Relevant coursework, credits transferred

- **Professional**
  - Years of experience
  - Current/most recent employer, title, dates
  - Prior positions (employer, title, dates, brief description)
  - Skills, programming languages, technologies
  - Professional memberships/associations

- **Financial**
  - FAFSA filed (yes/no), EFC if known
  - Household income range
  - Dependents
  - Currently receiving aid (types, amounts)
  - Employment status during school

- **Extracurricular / Community**
  - Volunteer work, community service
  - Leadership roles
  - Awards, honors, publications
  - Organizational memberships

- **Essays / Narrative Snippets** (transcribed, never AI-written)
  - Stored as key-value: `{ "prompt_summary": "transcribed_response" }`
  - Reusable across similar prompts if user confirms

### 2. Field Matcher (DeepSeek Integration)

**Model:** DeepSeek chat API, JSON mode.

**Input per request:**
```json
{
  "fields": [
    {
      "id": "form_field_dom_id",
      "label": "What is your racial/ethnic background?",
      "type": "select",
      "options": ["White", "Black or African American", "..."],
      "required": true
    }
  ],
  "profile": { "/* full profile JSON */" : "" }
}
```

**Output per request:**
```json
{
  "matches": [
    {
      "field_id": "form_field_dom_id",
      "profile_key": "personal.race_ethnicity",
      "value": "White",
      "confidence": 0.95,
      "reasoning": "Field asks for racial background, maps to profile race_ethnicity"
    }
  ],
  "unmatched": [
    {
      "field_id": "other_field_id",
      "label": "Name of tribal affiliation",
      "type": "text",
      "is_essay": false
    }
  ]
}
```

**Confidence tiers:**
- **≥ 0.8** — Auto-fill silently.
- **0.6–0.8** — Voice-confirm: reads the question and proposed answer, waits for "yes"/"no"/correction.
- **< 0.6 (non-essay)** — Voice-ask: reads the question aloud, transcribes user's answer, adds to profile, fills field.
- **Essay/open-ended detected** — Special flow (see §4).

**Field type handling:**
- Text inputs → insert string
- Select/dropdown → match closest option string
- Radio/checkbox → match value(s)
- Date fields → format from profile date to field's expected format
- Multi-select → map array of values

### 3. Voice System

**Text-to-Speech (ElevenLabs):**
- Voice ID: `UQoLnPXvf18gaKpLzfb8`
- Model: cheapest available (currently `eleven_flash_v2_5`)
- Used for: init interview questions, mid-stream confirmations/questions, essay readback
- Audio playback via `sounddevice` or `pyaudio`

**Speech-to-Text (faster-whisper):**
- Model: `base.en` or `small.en` (balance speed/accuracy, configurable)
- VAD (voice activity detection) for auto-stop after silence
- Light cleanup pipeline post-transcription:
  - Remove filler words (um, uh, like, you know)
  - Fix punctuation via simple heuristics
  - No rephrasing, no word replacement, no AI rewriting
- Used for: init interview answers, mid-stream answers, essay dictation

**Interaction loop (for confirms and asks):**
```
1. ElevenLabs speaks the question/confirmation
2. faster-whisper listens for response
3. Backend processes response
4. If confirmation: "yes" → fill, "no" → re-ask or skip
5. If new answer: clean up → add to profile → fill field
```

### 4. Essay / Open-Ended Flow

When the field matcher identifies a field as essay or open-ended (long text, textarea, or prompt-like label):

1. **Check profile** for a previously transcribed response to a similar prompt.
2. If match found → voice-read it back → ask "Would you like to reuse this, modify it, or start fresh?"
3. If new:
   a. Voice-read the prompt aloud.
   b. Record user's spoken response via faster-whisper.
   c. Light cleanup (punctuation/filler only).
   d. Voice-read the transcription back to the user.
   e. Ask: "Sound good, or would you like to redo it?"
   f. Save to profile under a summary key for future reuse.
4. **Insert into field** if possible (set textarea value + trigger input events).
5. If DOM insertion fails → copy to clipboard → voice-notify user to paste manually.

### 5. Application History & Dedup Engine

**Storage:** `~/.scholarship-assistant/history.json`

**Record per application:**
```json
{
  "id": "uuid",
  "timestamp": "ISO-8601",
  "url": "https://...",
  "scholarship_name": "Jane Doe Memorial Scholarship",
  "organization": "National Disability Foundation",
  "amount": "$2,500",
  "deadline": "2026-05-01",
  "status": "submitted",
  "fields_filled": 23,
  "fields_manual": 2,
  "essays": ["prompt_summary_key_1"],
  "notes": ""
}
```

**Dedup logic:**
- Before processing a page, extract scholarship name + organization from page content (via DeepSeek).
- Fuzzy match against history (normalized org name + scholarship name, threshold ~0.85 similarity).
- If duplicate detected → voice-alert: "It looks like you've already applied to [name] through [org] on [date]. Want to skip this one?"
- User can override if it's a renewal or different cycle.

### 6. Tampermonkey Userscript

**Trigger:** Floating button injected on all pages (positioned bottom-right, draggable). Click to activate.

**On activation:**
1. Scrape all visible form fields on the page:
   - `<input>`, `<select>`, `<textarea>`, `<radio>`, `<checkbox>`
   - For each: extract `id`, `name`, associated `<label>` text (or `placeholder`, `aria-label`, nearest text node), `type`, `options` (if select), `required` attribute.
2. Also scrape page title, h1/h2 headings, and any visible scholarship name/org for dedup.
3. POST everything to `http://localhost:8741/analyze`.
4. Receive fill instructions.
5. Apply fills to DOM:
   - Set `.value` and dispatch `input`, `change`, `blur` events (for React/Angular/Vue compatibility).
   - For selects: match option text, set `.selectedIndex`.
   - For checkboxes/radios: set `.checked` and dispatch `click`.
6. For fields requiring user interaction (confirm/ask/essay) — the backend handles voice I/O and returns the final value to fill when ready.
7. Visual indicators on filled fields:
   - Green subtle outline = auto-filled
   - Yellow = confirmed
   - Blue = user-provided new answer
   - Red = skipped / needs manual attention

**Communication protocol:**
- Uses Server-Sent Events (SSE) or WebSocket for streaming updates from backend during voice interactions (so the page updates in real-time as fields are resolved).

### 7. Init Interview

**Triggered on first run** (no profile.json found) or via CLI flag `--init`.

**Flow:**
1. Voice greeting: introduces the system, explains what data will be collected and why.
2. Walks through each profile category sequentially.
3. For each field:
   - Speaks the question naturally (not robotic form language).
   - Example: "What university are you enrolling at?" not "Please state your institution name."
   - Listens via faster-whisper.
   - Confirms: "Got it — [University of Montana], starting [March 2026]. Sound right?"
   - On correction: re-records.
4. For optional/sensitive fields (SSN, income, disability details):
   - Explains why it's asked and that it's optional.
   - Respects "skip" or "pass."
5. Saves profile incrementally (not all-at-end) so a crash doesn't lose progress.
6. Completion summary: reads back a high-level overview, asks if anything needs correction.

**Estimated time:** 15–25 minutes for a thorough interview.

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend framework | FastAPI |
| LLM / NLP | DeepSeek API (JSON mode) via parent `modules/deepseek.py` |
| TTS | ElevenLabs API (eleven_flash_v2_5) |
| STT | RealtimeSTT (whisper-based, already in parent deps) |
| Audio I/O | sounddevice + numpy |
| Profile storage | JSON (flat file) |
| History storage | JSON (flat file) |
| Fuzzy matching (dedup) | thefuzz (fuzzywuzzy successor) |
| Userscript | Tampermonkey (vanilla JS) |
| Backend ↔ Userscript | REST + WebSocket (localhost:8741) |

---

## File Structure

```
scholarship-assistant/
├── PLAN.md
├── requirements.txt
│
├── server/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app, routes, WebSocket
│   ├── config.py                 # Env loading, paths, constants
│   ├── profile_manager.py        # Load/save/update profile JSON
│   ├── field_matcher.py          # DeepSeek integration, confidence scoring
│   ├── voice.py                  # ElevenLabs TTS + RealtimeSTT STT + cleanup
│   ├── history.py                # Application history, dedup logic
│   ├── interview.py              # Init interview orchestration
│   └── essay_handler.py          # Essay/open-ended flow
│
├── userscript/
│   └── scholarship-assistant.user.js   # Tampermonkey userscript
│
└── data/                         # Created at runtime (~/.scholarship-assistant/)
    ├── profile.json
    └── history.json
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/analyze` | Receive scraped fields + page context, return fill plan |
| WS | `/ws` | WebSocket for real-time voice interaction during fill |
| GET | `/profile` | Return current profile |
| PATCH | `/profile` | Update specific profile fields |
| GET | `/history` | Return application history |
| POST | `/history/check` | Dedup check (scholarship name + org) |
| POST | `/init` | Start init interview (voice-driven) |
| GET | `/status` | Health check |

---

## Interaction Flow (Happy Path)

```
User clicks Tampermonkey button on scholarship page
  → Userscript scrapes form fields + page metadata
  → POST /analyze
  → Backend runs dedup check
    → If duplicate: voice-warns, asks to continue or skip
  → Backend sends fields + profile to DeepSeek (JSON mode)
  → DeepSeek returns matches with confidence scores
  → For each field:
      ≥0.8: return fill instruction immediately
      0.6-0.8: voice-confirm via WebSocket, wait for yes/no
      <0.6 (factual): voice-ask, transcribe, add to profile, fill
      essay: voice-read prompt, transcribe response, readback, confirm, fill or clipboard
  → Userscript applies all fills to DOM
  → Backend logs to history.json
  → Voice: "All done. [N] fields filled, [M] need your attention."
```

---

## Edge Cases & Considerations

- **Multi-page forms:** Some applications span multiple pages. The userscript should be re-clickable per page. History tracks partial progress.
- **CAPTCHAs:** Cannot be automated. Voice-notify user to complete manually.
- **File upload fields:** Cannot be auto-filled. Voice-notify and skip.
- **Session timeouts:** Long voice interactions on slow forms. Consider filling in batches (auto-fill high confidence first, then walk through the rest).
- **Rate limiting:** DeepSeek API rate limits. Batch fields into single requests where possible (send all fields at once, not one-by-one).
- **Audio device issues:** Graceful fallback to text input if mic/speaker not available.
- **Profile versioning:** Append a `_version` field; if profile schema changes, migrate forward.

---

## Build Order

1. **Phase 1 — Core profile + voice infrastructure**
   - Config/env loading (reuse parent patterns)
   - Profile manager (CRUD on JSON)
   - Voice module (ElevenLabs TTS + RealtimeSTT STT + cleanup)
   - Init interview (full walkthrough, builds profile)

2. **Phase 2 — Field matching + backend API**
   - DeepSeek integration (JSON mode field matching via parent `modules/deepseek.py`)
   - FastAPI routes (/analyze, /profile, /status)
   - WebSocket for real-time voice interaction
   - Confidence tier logic (auto-fill / confirm / ask)

3. **Phase 3 — Tampermonkey userscript**
   - DOM scraping (fields, labels, options, page metadata)
   - Communication with backend (POST + WebSocket)
   - DOM filling (value setting + event dispatch for framework compat)
   - Visual indicators

4. **Phase 4 — Essay handling + history/dedup**
   - Essay detection and flow
   - Transcription → readback → confirm → fill/clipboard
   - History tracking
   - Dedup engine (fuzzy match on org + name)

5. **Phase 5 — Polish**
   - Error handling, retries, graceful degradation
   - Audio device fallbacks
   - Multi-page form support
   - Profile editing via voice ("update my GPA to 3.8")
