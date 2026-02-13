"""
API integrations: Open-Meteo, Datamuse, DeepL, Wolfram Alpha, Todoist,
local Plane API, local n8n API.
"""
import json
import logging
import os
import re
from typing import Any, Optional

import httpx

logger = logging.getLogger("main")

# --- Open-Meteo (free, no key) ---
OPEN_METEO_BASE = "https://api.open-meteo.com/v1"


def open_meteo_weather(
    latitude: float = 40.7128,
    longitude: float = -74.0060,
    location: Optional[str] = None,
) -> str:
    """Get weather from Open-Meteo. Pass location for place name only (uses NYC coords)."""
    if location and not (latitude and longitude):
        # Use geocoding - Open-Meteo has a geocoding API
        try:
            with httpx.Client(timeout=10) as client:
                geo = client.get(
                    f"https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": location, "count": 1},
                )
                geo.raise_for_status()
                data = geo.json()
                if data.get("results"):
                    r = data["results"][0]
                    latitude = r["latitude"]
                    longitude = r["longitude"]
                    location = r.get("name", location)
        except Exception as e:
            logger.warning(f"Geocoding failed for '{location}': {e}")
            latitude, longitude = 40.7128, -74.0060

    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                f"{OPEN_METEO_BASE}/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                    "daily": "temperature_2m_max,temperature_2m_min,weather_code",
                    "timezone": "auto",
                    "forecast_days": 3,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.error(f"Open-Meteo failed: {e}")
        return f"Weather error: {e}"

    current = data.get("current", {})
    daily = data.get("daily", {})
    temps = daily.get("temperature_2m_max", [])
    units = data.get("current_units", {}).get("temperature_2m", "°C")
    loc_str = f" for {location}" if location else f" ({latitude}, {longitude})"

    lines = [
        f"Weather{loc_str}:",
        f"  Current: {current.get('temperature_2m', 'N/A')}{units}, "
        f"humidity {current.get('relative_humidity_2m', 'N/A')}%, "
        f"wind {current.get('wind_speed_10m', 'N/A')} km/h",
    ]
    if temps:
        lines.append(f"  Next days highs: {', '.join(str(t) for t in temps[:3])}{units}")
    return "\n".join(lines)


# --- Datamuse (free, no key) ---
DATAMUSE_BASE = "https://api.datamuse.com"


def datamuse_words(
    means_like: Optional[str] = None,
    sounds_like: Optional[str] = None,
    spelled_like: Optional[str] = None,
    related_to: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """Query Datamuse words API. Use ml, sl, sp, or rel params."""
    params = {"max": max_results}
    if means_like:
        params["ml"] = means_like
    if sounds_like:
        params["sl"] = sounds_like
    if spelled_like:
        params["sp"] = spelled_like
    if related_to:
        params["rel_trg"] = related_to
    if len(params) <= 1:
        return "Provide at least one of: means_like, sounds_like, spelled_like, related_to"

    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{DATAMUSE_BASE}/words", params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.error(f"Datamuse failed: {e}")
        return f"Datamuse error: {e}"

    if not data:
        return "No words found."
    words = [w.get("word", "") for w in data if w.get("word")]
    return ", ".join(words) if words else "No words found."


# --- DeepL (needs DEEPL_API_KEY) ---
def deepl_translate(
    text: str,
    target_lang: str = "EN",
    source_lang: Optional[str] = None,
) -> str:
    """Translate text via DeepL. Requires DEEPL_API_KEY."""
    key = os.getenv("DEEPL_API_KEY")
    if not key:
        return "DeepL requires DEEPL_API_KEY. Set it in your environment."

    # Free API uses api-free.deepl.com
    url = "https://api-free.deepl.com/v2/translate"
    payload = {
        "auth_key": key,
        "text": text,
        "target_lang": target_lang.upper()[:2],
    }
    if source_lang:
        payload["source_lang"] = source_lang.upper()[:2]

    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(url, data=payload)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.error(f"DeepL failed: {e}")
        return f"DeepL error: {e}"

    translations = data.get("translations", [])
    if translations:
        return translations[0].get("text", "")
    return "No translation returned."


# --- Wolfram Alpha (needs WOLFRAM_APP_ID) ---
def wolfram_query(query: str) -> str:
    """Query Wolfram Alpha. Requires WOLFRAM_APP_ID."""
    app_id = os.getenv("WOLFRAM_APP_ID")
    if not app_id:
        return "Wolfram Alpha requires WOLFRAM_APP_ID. Get one at developer.wolframalpha.com"

    try:
        with httpx.Client(timeout=20) as client:
            resp = client.get(
                "https://api.wolframalpha.com/v2/query",
                params={"input": query, "appid": app_id, "output": "plaintext"},
            )
            resp.raise_for_status()
            # Parse XML-ish response for plaintext pods
            text = resp.text
    except Exception as e:
        logger.error(f"Wolfram failed: {e}")
        return f"Wolfram error: {e}"

    # Simple extraction of plaintext from XML
    plains = re.findall(r"<plaintext[^>]*>([^<]*)</plaintext>", text)
    lines = [p.strip() for p in plains if p.strip()]
    if lines:
        return "\n".join(lines[:5])  # First 5 pods
    return "No result from Wolfram Alpha."


# --- Todoist (needs TODOIST_API_TOKEN) ---
TODOIST_BASE = "https://api.todoist.com/rest/v2"


def todoist_list_tasks(project_id: Optional[str] = None) -> str:
    """List Todoist tasks. Requires TODOIST_API_TOKEN."""
    token = os.getenv("TODOIST_API_TOKEN")
    if not token:
        return "Todoist requires TODOIST_API_TOKEN. Get one at todoist.com/app/settings/integrations"

    headers = {"Authorization": f"Bearer {token}"}
    params = {}
    if project_id:
        params["project_id"] = project_id

    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                f"{TODOIST_BASE}/tasks",
                headers=headers,
                params=params or None,
            )
            resp.raise_for_status()
            tasks = resp.json()
    except Exception as e:
        logger.error(f"Todoist list failed: {e}")
        return f"Todoist error: {e}"

    if not tasks:
        return "No tasks found."
    lines = []
    for t in tasks[:20]:
        content = t.get("content", "")
        due = t.get("due", {})
        due_str = due.get("date", "") if isinstance(due, dict) else str(due)
        lines.append(f"- {content}" + (f" (due: {due_str})" if due_str else ""))
    return "\n".join(lines)


def todoist_add_task(content: str, project_id: Optional[str] = None) -> str:
    """Add a Todoist task. Requires TODOIST_API_TOKEN."""
    token = os.getenv("TODOIST_API_TOKEN")
    if not token:
        return "Todoist requires TODOIST_API_TOKEN."

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"content": content}
    if project_id:
        payload["project_id"] = project_id

    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                f"{TODOIST_BASE}/tasks",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.error(f"Todoist add failed: {e}")
        return f"Todoist error: {e}"

    return f"Task added: {data.get('content', content)} (ID: {data.get('id', '')})"


# --- Local Plane API ---
def plane_status(base_url: str = "http://localhost:8000") -> str:
    """Check local Plane instance status. Default http://localhost:8000."""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{base_url.rstrip('/')}/api/v1/health/")
            resp.raise_for_status()
            return f"Plane API at {base_url}: OK"
    except httpx.ConnectError:
        return f"Plane API at {base_url}: Connection refused (is Plane running?)"
    except Exception as e:
        return f"Plane API error: {e}"


# --- Local n8n API ---
def n8n_status(base_url: str = "http://localhost:5678") -> str:
    """Check local n8n instance status. Default http://localhost:5678."""
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{base_url.rstrip('/')}/healthz")
            resp.raise_for_status()
            return f"n8n at {base_url}: OK"
    except httpx.ConnectError:
        return f"n8n at {base_url}: Connection refused (is n8n running?)"
    except Exception as e:
        return f"n8n error: {e}"
