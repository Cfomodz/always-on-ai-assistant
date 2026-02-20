"""
FastAPI backend for the Scholarship Assistant.
Serves on localhost:8741. Communicates with Tampermonkey userscript via REST + WebSocket.
"""

import asyncio
import json
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

# Ensure the scholarship-assistant package root is importable so that
# "from server.X import ..." works whether the file is run directly or
# imported as part of the project.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.config import HOST, PORT, ensure_data_dir
from server.profile_manager import (
    load_profile,
    save_profile,
    update_profile,
    profile_exists,
)
from server.field_matcher import match_fields, categorize_matches
from server.history import (
    load_history,
    add_record,
    check_duplicate,
    extract_scholarship_info,
)
from server.voice import (
    speak,
    ask_and_listen,
    confirm,
    speak_acknowledgment,
    speak_processing,
    speak_completion,
    speak_error,
    speak_startup,
    warmup_cache,
)
from server.essay_handler import handle_essay
from server.interview import run_interview
from server.profile_import import import_into_profile, load_content_from_path, review_profile

logger = logging.getLogger("scholarship-assistant")


# --- Pydantic models ---


class FormField(BaseModel):
    id: str = ""
    name: str = ""
    label: str = ""
    type: str = "text"
    options: list[str] = []
    required: bool = False
    value: str = ""


class PageContext(BaseModel):
    title: str = ""
    url: str = ""
    headings: list[str] = []
    visible_text: str = ""


class AnalyzeRequest(BaseModel):
    fields: list[FormField]
    page_context: PageContext


class ProfileUpdate(BaseModel):
    updates: dict[str, object]


class ProfileImportRequest(BaseModel):
    content: str
    dry_run: bool = False


class DedupCheck(BaseModel):
    scholarship_name: str
    organization: str


# --- App lifecycle ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_dir()
    logger.info(f"Scholarship Assistant backend starting on {HOST}:{PORT}")

    # Warm up TTS cache in background so fillers are instant
    def _warmup():
        try:
            warmup_cache()
            speak_startup()
        except Exception as e:
            logger.warning(f"TTS cache warmup failed: {e}")

    warmup_thread = threading.Thread(target=_warmup, daemon=True)
    warmup_thread.start()

    yield
    logger.info("Scholarship Assistant backend shutting down")


app = FastAPI(
    title="Scholarship Assistant",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],  # Tampermonkey bypasses CORS; no other origins allowed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- REST Endpoints ---


@app.get("/status")
async def status():
    from modules.tts_cache import cache_stats
    return {
        "status": "ok",
        "profile_exists": profile_exists(),
        "tts_cache": cache_stats(),
    }


@app.post("/cache/warmup")
async def cache_warmup_endpoint():
    """Manually trigger TTS cache warmup for all filler/acknowledgment audio."""
    def _warmup():
        return warmup_cache()

    loop = asyncio.get_event_loop()
    count = await loop.run_in_executor(None, _warmup)
    return {"status": "ok", "lines_cached": count}


@app.get("/profile")
async def get_profile():
    return load_profile()


@app.patch("/profile")
async def patch_profile(body: ProfileUpdate):
    updated = update_profile(body.updates)
    return updated


@app.get("/profile/import", response_class=HTMLResponse)
async def profile_import_page():
    """Serve the profile import UI page."""
    html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Scholarship Assistant — Import Profile</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      max-width: 720px;
      margin: 2rem auto;
      padding: 0 1rem;
      color: #333;
    }
    h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
    p { color: #666; font-size: 0.95rem; margin-bottom: 1rem; }
    textarea {
      width: 100%;
      min-height: 280px;
      padding: 12px;
      font-family: inherit;
      font-size: 14px;
      border: 1px solid #ccc;
      border-radius: 8px;
      resize: vertical;
    }
    textarea:focus { outline: none; border-color: #1a73e8; }
    .actions { display: flex; gap: 12px; align-items: center; margin-top: 12px; }
    button {
      padding: 10px 20px;
      font-size: 15px;
      border: none;
      border-radius: 8px;
      cursor: pointer;
    }
    button.primary {
      background: #1a73e8;
      color: white;
    }
    button.primary:hover { background: #1557b0; }
    button.secondary {
      background: #f1f3f4;
      color: #333;
    }
    button.secondary:hover { background: #e8eaed; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .result {
      margin-top: 1.5rem;
      padding: 1rem;
      border-radius: 8px;
      display: none;
    }
    .result.success { background: #e6f4ea; border: 1px solid #34a853; }
    .result.error { background: #fce8e6; border: 1px solid #ea4335; }
    .result pre { margin: 0.5rem 0 0; font-size: 13px; overflow-x: auto; }
    .result h3 { margin: 0 0 0.5rem; font-size: 1rem; }
    label { display: flex; align-items: center; gap: 8px; font-size: 14px; }
  </style>
</head>
<body>
  <h1>Import Profile Data</h1>
  <p>Paste Q&A pairs (tab-separated: Question, Answer, Last Answered) or raw text below. DeepSeek will extract and merge the data into your scholarship profile.</p>
  <textarea id="content" placeholder="Paste your Q&A data here...

Example (tab-separated):
Question	Answer(s)	Last Answered
What is your citizenship status?	US Citizen	6/13/2025
Which college are you currently attending?	WESTERN GOVERNORS UNIVERSITY (UT)	2/13/2026
..."></textarea>
  <div class="actions">
    <button id="importBtn" class="primary">Import</button>
    <button id="dryRunBtn" class="secondary">Preview (dry run)</button>
  </div>
  <div id="result" class="result"></div>
  <script>
    const content = document.getElementById('content');
    const importBtn = document.getElementById('importBtn');
    const dryRunBtn = document.getElementById('dryRunBtn');
    const resultEl = document.getElementById('result');
    const API = window.location.origin;

    async function runImport(dryRun) {
      const text = content.value.trim();
      if (!text) {
        alert('Please paste some content to import.');
        return;
      }
      importBtn.disabled = true;
      dryRunBtn.disabled = true;
      resultEl.style.display = 'none';
      try {
        const res = await fetch(API + '/profile/import', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content: text, dry_run: dryRun })
        });
        const data = await res.json();
        resultEl.className = 'result ' + (data.error ? 'error' : 'success');
        resultEl.style.display = 'block';
        let msg = data.summary || '';
        if (data.updates && Object.keys(data.updates).length > 0) {
          msg += '\\n\\nUpdates applied: ' + Object.keys(data.updates).length;
          msg += '\\n' + JSON.stringify(data.updates, null, 2);
        }
        if (data.error) msg = 'Error: ' + data.error;
        if (data.skipped && data.skipped.length) {
          msg += '\\n\\nSkipped: ' + data.skipped.join(', ');
        }
        resultEl.innerHTML = '<h3>' + (dryRun ? 'Preview' : 'Result') + '</h3><pre>' + msg.replace(/</g, '&lt;') + '</pre>';
      } catch (e) {
        resultEl.className = 'result error';
        resultEl.style.display = 'block';
        resultEl.innerHTML = '<h3>Error</h3><pre>' + e.message + '</pre>';
      }
      importBtn.disabled = false;
      dryRunBtn.disabled = false;
    }
    importBtn.onclick = () => runImport(false);
    dryRunBtn.onclick = () => runImport(true);
  </script>
</body>
</html>
"""
    return HTMLResponse(html)


@app.post("/profile/import")
async def profile_import(body: ProfileImportRequest):
    """Import Q&A pairs or raw text into the profile via DeepSeek."""
    result = await run_in_threadpool(
        import_into_profile,
        body.content,
        body.dry_run,
    )
    return result


@app.get("/history")
async def get_history():
    return load_history()


@app.post("/history/check")
async def history_check(body: DedupCheck):
    existing = check_duplicate(body.scholarship_name, body.organization)
    if existing:
        return {"duplicate": True, "record": existing}
    return {"duplicate": False, "record": None}


@app.post("/init")
async def start_init():
    """Start the init interview in a background thread (voice-driven)."""
    if profile_exists():
        return JSONResponse(
            status_code=409,
            content={"error": "Profile already exists. Use --init flag to re-run."},
        )

    def _run():
        try:
            run_interview()
        except Exception as e:
            logger.error(f"Interview failed: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"status": "interview_started"}


@app.post("/analyze")
async def analyze(body: AnalyzeRequest):
    """
    Main endpoint: receive scraped fields, run dedup + field matching,
    return fill plan. Voice interactions for confirm/ask/essay happen via /ws.
    """
    fields_raw = [f.model_dump() for f in body.fields]
    page_ctx = body.page_context.model_dump()

    # Step 1: Extract scholarship info for dedup
    scholarship_info = await run_in_threadpool(extract_scholarship_info, page_ctx)

    # Step 2: Dedup check
    duplicate = await run_in_threadpool(
        check_duplicate,
        scholarship_info.get("scholarship_name", ""),
        scholarship_info.get("organization", ""),
    )

    # Step 3: Field matching via DeepSeek
    match_result = await run_in_threadpool(match_fields, fields_raw)
    categorized = categorize_matches(match_result)

    return {
        "scholarship_info": scholarship_info,
        "duplicate": {
            "is_duplicate": duplicate is not None,
            "record": duplicate,
        },
        "fill_plan": {
            "auto_fill": categorized["auto_fill"],
            "confirm": categorized["confirm"],
            "ask": categorized["ask"],
            "essay": categorized["essay"],
            "skip": categorized["skip"],
        },
        "stats": {
            "total_fields": len(fields_raw),
            "auto_fill_count": len(categorized["auto_fill"]),
            "confirm_count": len(categorized["confirm"]),
            "ask_count": len(categorized["ask"]),
            "essay_count": len(categorized["essay"]),
            "skip_count": len(categorized["skip"]),
        },
    }


# --- WebSocket for real-time voice interaction ---


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket for real-time voice interaction during form filling.
    The userscript connects here after receiving the fill plan from /analyze.

    Protocol:
    - Client sends: {"action": "confirm", "field_id": "...", "label": "...", "value": "..."}
    - Client sends: {"action": "ask", "field_id": "...", "label": "..."}
    - Client sends: {"action": "essay", "field_id": "...", "label": "..."}
    - Server responds: {"field_id": "...", "value": "...", "status": "filled|skipped"}
    """
    await ws.accept()
    logger.info("WebSocket: client connected")

    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action")
            field_id = data.get("field_id", "")
            label = data.get("label", "")
            value = data.get("value", "")

            result = {"field_id": field_id, "value": "", "status": "skipped"}

            if action == "confirm":
                # Voice-confirm a proposed value
                def _do_confirm():
                    confirmed, correction = confirm(label, value)
                    if confirmed:
                        return value
                    elif correction:
                        # Save correction to profile
                        profile_key = data.get("profile_key", "")
                        if profile_key:
                            update_profile({profile_key: correction})
                        return correction
                    return None

                loop = asyncio.get_event_loop()
                filled_value = await loop.run_in_executor(None, _do_confirm)

                if filled_value:
                    result = {"field_id": field_id, "value": filled_value, "status": "filled"}

            elif action == "ask":
                # Voice-ask for an unknown field
                def _do_ask():
                    response = ask_and_listen(f"The application is asking: {label}")
                    if response:
                        profile_key = data.get("profile_key", "")
                        if profile_key:
                            update_profile({profile_key: response})
                    return response

                loop = asyncio.get_event_loop()
                filled_value = await loop.run_in_executor(None, _do_ask)

                if filled_value:
                    result = {"field_id": field_id, "value": filled_value, "status": "filled"}

            elif action == "essay":
                # Play acknowledgment so user knows we're starting the essay flow
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, speak_acknowledgment)

                def _do_essay():
                    return handle_essay(field_id, label)

                essay_text = await loop.run_in_executor(None, _do_essay)

                if essay_text:
                    result = {"field_id": field_id, "value": essay_text, "status": "filled"}

            elif action == "done":
                # Client signals all fields processed
                def _do_summary():
                    stats = data.get("stats", {})
                    filled = stats.get("filled", 0)
                    manual = stats.get("manual", 0)
                    speak_completion()
                    speak(
                        f"All done. {filled} fields filled, "
                        f"{manual} need your attention."
                    )

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, _do_summary)

                # Log to history
                page_ctx = data.get("page_context", {})
                scholarship_info = data.get("scholarship_info", {})
                add_record(
                    url=page_ctx.get("url", ""),
                    scholarship_name=scholarship_info.get("scholarship_name", ""),
                    organization=scholarship_info.get("organization", ""),
                    amount=scholarship_info.get("amount", ""),
                    deadline=scholarship_info.get("deadline", ""),
                    fields_filled=data.get("stats", {}).get("filled", 0),
                    fields_manual=data.get("stats", {}).get("manual", 0),
                )

                result = {"status": "complete"}

            await ws.send_json(result)

    except WebSocketDisconnect:
        logger.info("WebSocket: client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await ws.send_json({"error": str(e)})
        except Exception:
            pass


# --- CLI entry point ---

if __name__ == "__main__":
    import argparse
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("scholarship-assistant")

    parser = argparse.ArgumentParser(description="Scholarship Assistant Backend")
    parser.add_argument("--init", action="store_true", help="Run the init interview")
    parser.add_argument(
        "--import-file",
        metavar="FILE",
        help="Import Q&A from a .txt file into the profile (uses DeepSeek to parse)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --import-file or --review-profile: show proposed updates without applying",
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="With --import-file: skip profile review (dedup/cleanup) after import",
    )
    parser.add_argument(
        "--review-profile",
        action="store_true",
        help="Run profile review only (deduplicate, fix misplaced data, clean inconsistencies)",
    )
    parser.add_argument("--host", default=HOST, help=f"Host (default: {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port (default: {PORT})")
    args = parser.parse_args()

    if args.import_file:
        ensure_data_dir()
        path = Path(args.import_file)
        if not path.exists():
            logger.error(f"File not found: {path}")
            sys.exit(1)
        try:
            content = load_content_from_path(path)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            sys.exit(1)
        result = import_into_profile(content, dry_run=args.dry_run)
        if result.get("error"):
            logger.error(result["error"])
            sys.exit(1)
        print(result["summary"])
        if result.get("updates"):
            print(f"\nUpdates ({'preview' if args.dry_run else 'applied'}):")
            for k, v in result["updates"].items():
                print(f"  {k}: {v}")
        if result.get("skipped"):
            print(f"\nSkipped: {', '.join(result['skipped'])}")
        if args.dry_run and result.get("updates"):
            print("\n(Run without --dry-run to apply)")
        # Profile review (dedup/cleanup) after import, unless dry-run or --no-review
        elif not args.dry_run and not args.no_review and result.get("applied"):
            print("\n--- Profile review ---")
            review_result = review_profile(dry_run=False)
            if review_result.get("error"):
                logger.error(review_result["error"])
            else:
                print(review_result["summary"])
                if review_result.get("updates"):
                    print("Cleanup applied:")
                    for k, v in review_result["updates"].items():
                        print(f"  {k}: {v}")
                if review_result.get("skipped"):
                    print(f"Skipped: {', '.join(review_result['skipped'])}")
    elif args.review_profile:
        ensure_data_dir()
        result = review_profile(dry_run=args.dry_run)
        if result.get("error"):
            logger.error(result["error"])
            sys.exit(1)
        print(result["summary"])
        if result.get("updates"):
            print(f"\nUpdates ({'preview' if args.dry_run else 'applied'}):")
            for k, v in result["updates"].items():
                print(f"  {k}: {v}")
        if result.get("skipped"):
            print(f"\nSkipped: {', '.join(result['skipped'])}")
        if args.dry_run and result.get("updates"):
            print("\n(Run without --dry-run to apply)")
    elif args.init:
        ensure_data_dir()
        run_interview()
    else:
        logger.info(f"Starting server on {args.host}:{args.port}")
        uvicorn.run(
            "server.main:app",
            host=args.host,
            port=args.port,
            reload=False,
            log_level="info",
        )
