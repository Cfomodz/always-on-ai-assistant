// ==UserScript==
// @name         Scholarship Assistant
// @namespace    scholarship-assistant
// @version      0.1.0
// @description  Voice-driven scholarship form auto-filler. Click the floating button to analyze and fill forms.
// @author       ScholarShip Assistant
// @match        *://*/*
// @grant        GM_xmlhttpRequest
// @grant        GM_setClipboard
// @grant        GM_notification
// @connect      localhost
// @connect      127.0.0.1
// @run-at       document-idle
// ==/UserScript==

(function () {
  "use strict";

  const BACKEND_URL = "http://localhost:8741";
  const WS_URL = "ws://localhost:8741/ws";

  // ─── Visual Indicators ───────────────────────────────────────────

  const INDICATOR_STYLES = {
    auto_fill: "outline: 2px solid #4CAF50; outline-offset: -2px;", // green
    confirmed: "outline: 2px solid #FFC107; outline-offset: -2px;", // yellow
    user_provided: "outline: 2px solid #2196F3; outline-offset: -2px;", // blue
    skipped: "outline: 2px solid #F44336; outline-offset: -2px;", // red
  };

  function markField(fieldEl, category) {
    if (fieldEl && INDICATOR_STYLES[category]) {
      fieldEl.style.cssText += INDICATOR_STYLES[category];
    }
  }

  // ─── DOM Field Scraping ──────────────────────────────────────────

  function getLabelForField(field) {
    // 1. Explicit <label for="...">
    if (field.id) {
      const label = document.querySelector(`label[for="${CSS.escape(field.id)}"]`);
      if (label) return label.textContent.trim();
    }

    // 2. Wrapping <label>
    const parentLabel = field.closest("label");
    if (parentLabel) {
      const clone = parentLabel.cloneNode(true);
      // Remove the input itself from the clone to get just the label text
      const inputs = clone.querySelectorAll("input, select, textarea");
      inputs.forEach((el) => el.remove());
      const text = clone.textContent.trim();
      if (text) return text;
    }

    // 3. aria-label
    if (field.getAttribute("aria-label")) {
      return field.getAttribute("aria-label");
    }

    // 4. aria-labelledby
    const labelledBy = field.getAttribute("aria-labelledby");
    if (labelledBy) {
      const labelEl = document.getElementById(labelledBy);
      if (labelEl) return labelEl.textContent.trim();
    }

    // 5. placeholder
    if (field.placeholder) return field.placeholder;

    // 6. Nearest preceding text node (sibling or parent child)
    const prev = field.previousElementSibling;
    if (prev && prev.textContent.trim().length < 200) {
      return prev.textContent.trim();
    }

    // 7. name or id as fallback
    return field.name || field.id || "";
  }

  function getFieldType(field) {
    const tag = field.tagName.toLowerCase();
    if (tag === "select") return "select";
    if (tag === "textarea") return "textarea";
    if (tag === "input") return field.type || "text";
    return "text";
  }

  function getOptions(field) {
    if (field.tagName.toLowerCase() === "select") {
      return Array.from(field.options).map((o) => ({
        value: o.value,
        text: o.textContent.trim(),
      }));
    }
    // For radio groups
    if (field.type === "radio" && field.name) {
      return Array.from(document.querySelectorAll(`input[name="${CSS.escape(field.name)}"]`)).map(
        (r) => ({
          value: r.value,
          text: getLabelForField(r) || r.value,
        })
      );
    }
    return [];
  }

  function scrapeFields() {
    const selectors = "input, select, textarea";
    const fields = [];
    const seen = new Set();

    document.querySelectorAll(selectors).forEach((field) => {
      // Skip hidden, submit, button, file fields
      if (
        field.type === "hidden" ||
        field.type === "submit" ||
        field.type === "button" ||
        field.type === "file" ||
        field.type === "image" ||
        field.type === "reset"
      ) {
        return;
      }

      // Skip invisible fields
      const style = window.getComputedStyle(field);
      if (style.display === "none" || style.visibility === "hidden") return;

      // Deduplicate radio buttons (only include once per name)
      if (field.type === "radio") {
        if (seen.has(field.name)) return;
        seen.add(field.name);
      }

      const fieldData = {
        id: field.id || field.name || `field_${fields.length}`,
        name: field.name || "",
        label: getLabelForField(field),
        type: getFieldType(field),
        options: getOptions(field).map((o) => o.text || o.value),
        required: field.required || field.getAttribute("aria-required") === "true",
      };

      fields.push(fieldData);
    });

    return fields;
  }

  function scrapePageContext() {
    const headings = Array.from(document.querySelectorAll("h1, h2, h3"))
      .slice(0, 10)
      .map((h) => h.textContent.trim());

    // Get a snippet of visible text (for dedup context)
    const bodyText = document.body.innerText || "";
    const visibleText = bodyText.substring(0, 2000);

    return {
      title: document.title,
      url: window.location.href,
      headings: headings,
      visible_text: visibleText,
    };
  }

  // ─── DOM Filling ─────────────────────────────────────────────────

  function setFieldValue(fieldId, value) {
    // Try by id first, then by name
    let field =
      document.getElementById(fieldId) ||
      document.querySelector(`[name="${CSS.escape(fieldId)}"]`);

    if (!field) return null;

    const type = getFieldType(field);

    if (type === "select") {
      // Coerce to string to guard against numeric/boolean values from the LLM
      const strValue = String(value);
      // Match option by text or value
      const options = Array.from(field.options);
      const match =
        options.find(
          (o) => o.textContent.trim().toLowerCase() === strValue.toLowerCase()
        ) ||
        options.find((o) => o.value.toLowerCase() === strValue.toLowerCase()) ||
        options.find((o) =>
          o.textContent.trim().toLowerCase().includes(strValue.toLowerCase())
        );
      if (match) {
        field.value = match.value;
      }
    } else if (type === "radio") {
      // Coerce to string to guard against numeric/boolean values from the LLM
      const strValue = String(value);
      // Find the radio with matching value or label
      const radios = document.querySelectorAll(
        `input[name="${CSS.escape(field.name)}"]`
      );
      radios.forEach((r) => {
        const radioLabel = getLabelForField(r) || r.value;
        if (
          r.value.toLowerCase() === strValue.toLowerCase() ||
          radioLabel.toLowerCase().includes(strValue.toLowerCase())
        ) {
          r.checked = true;
          r.dispatchEvent(new Event("click", { bubbles: true }));
        }
      });
    } else if (type === "checkbox") {
      const shouldCheck =
        value === true ||
        value === "true" ||
        value === "yes" ||
        value === "1";
      field.checked = shouldCheck;
      field.dispatchEvent(new Event("click", { bubbles: true }));
    } else {
      // Text, textarea, date, email, etc.
      field.value = value;
    }

    // Dispatch events for React/Angular/Vue compatibility
    field.dispatchEvent(new Event("input", { bubbles: true }));
    field.dispatchEvent(new Event("change", { bubbles: true }));
    field.dispatchEvent(new Event("blur", { bubbles: true }));

    return field;
  }

  // ─── Backend Communication ───────────────────────────────────────

  function postToBackend(path, data) {
    return new Promise((resolve, reject) => {
      GM_xmlhttpRequest({
        method: "POST",
        url: `${BACKEND_URL}${path}`,
        headers: { "Content-Type": "application/json" },
        data: JSON.stringify(data),
        onload: (response) => {
          try {
            resolve(JSON.parse(response.responseText));
          } catch (e) {
            reject(new Error("Invalid JSON response"));
          }
        },
        onerror: (err) =>
          reject(new Error(`Backend unavailable: ${err.statusText || "connection refused"}`)),
      });
    });
  }

  function connectWebSocket() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(WS_URL);
      ws.onopen = () => resolve(ws);
      ws.onerror = (err) => reject(err);
    });
  }

  // ─── Main Fill Pipeline ──────────────────────────────────────────

  async function runFillPipeline() {
    const statusEl = document.getElementById("sa-status");
    const setStatus = (msg) => {
      if (statusEl) statusEl.textContent = msg;
    };

    try {
      setStatus("Scraping fields...");

      const fields = scrapeFields();
      const pageContext = scrapePageContext();

      if (fields.length === 0) {
        setStatus("No form fields found.");
        return;
      }

      setStatus(`Found ${fields.length} fields. Analyzing...`);

      // Send to backend for matching
      const result = await postToBackend("/analyze", {
        fields: fields,
        page_context: pageContext,
      });

      // Handle duplicate warning
      if (result.duplicate && result.duplicate.is_duplicate) {
        const rec = result.duplicate.record;
        setStatus(
          `Duplicate detected: ${rec.scholarship_name} (${rec.timestamp})`
        );
        // Backend will voice-warn; wait for user decision via WS
      }

      const plan = result.fill_plan;
      let filledCount = 0;
      let manualCount = 0;

      // Phase 1: Auto-fill high-confidence matches
      setStatus(`Auto-filling ${plan.auto_fill.length} fields...`);
      for (const match of plan.auto_fill) {
        const fieldEl = setFieldValue(match.field_id, match.value);
        if (fieldEl) {
          markField(fieldEl, "auto_fill");
          filledCount++;
        }
      }

      // Phase 2: Connect WebSocket for interactive fields
      if (
        plan.confirm.length > 0 ||
        plan.ask.length > 0 ||
        plan.essay.length > 0
      ) {
        setStatus("Connecting for voice interaction...");
        let ws;
        try {
          ws = await connectWebSocket();
        } catch (e) {
          setStatus("Could not connect WebSocket. Fill remaining fields manually.");
          plan.confirm.forEach((m) => {
            const el = setFieldValue(m.field_id, m.value);
            if (el) markField(el, "confirmed");
            filledCount++;
          });
          manualCount += plan.ask.length + plan.essay.length;
          return;
        }

        // Process confirms
        for (const match of plan.confirm) {
          setStatus(`Confirming: ${match.field_id}...`);
          ws.send(
            JSON.stringify({
              action: "confirm",
              field_id: match.field_id,
              label: match.reasoning || match.field_id,
              value: match.value,
              profile_key: match.profile_key || "",
            })
          );
          const response = await waitForWsMessage(ws);
          if (response.status === "filled") {
            const el = setFieldValue(response.field_id, response.value);
            if (el) markField(el, "confirmed");
            filledCount++;
          } else {
            manualCount++;
            const el =
              document.getElementById(match.field_id) ||
              document.querySelector(`[name="${CSS.escape(match.field_id)}"]`);
            markField(el, "skipped");
          }
        }

        // Process asks
        for (const match of plan.ask) {
          setStatus(`Asking: ${match.field_id}...`);
          ws.send(
            JSON.stringify({
              action: "ask",
              field_id: match.field_id,
              label: match.reasoning || match.label || match.field_id,
              profile_key: match.profile_key || "",
            })
          );
          const response = await waitForWsMessage(ws);
          if (response.status === "filled") {
            const el = setFieldValue(response.field_id, response.value);
            if (el) markField(el, "user_provided");
            filledCount++;
          } else {
            manualCount++;
          }
        }

        // Process essays
        for (const essay of plan.essay) {
          setStatus(`Essay: ${essay.label || essay.field_id}...`);
          ws.send(
            JSON.stringify({
              action: "essay",
              field_id: essay.field_id,
              label: essay.label || "",
            })
          );
          const response = await waitForWsMessage(ws);
          if (response.status === "filled") {
            const el = setFieldValue(response.field_id, response.value);
            if (el) {
              markField(el, "user_provided");
              filledCount++;
            } else {
              // DOM insertion failed — copy to clipboard
              GM_setClipboard(response.value);
              GM_notification({
                text: "Essay copied to clipboard. Paste it manually.",
                title: "Scholarship Assistant",
                timeout: 5000,
              });
              manualCount++;
            }
          } else {
            manualCount++;
          }
        }

        // Signal done
        ws.send(
          JSON.stringify({
            action: "done",
            stats: { filled: filledCount, manual: manualCount },
            page_context: pageContext,
            scholarship_info: result.scholarship_info,
          })
        );

        ws.close();
      }

      // Mark skipped fields
      for (const field of plan.skip) {
        const el =
          document.getElementById(field.field_id) ||
          document.querySelector(`[name="${CSS.escape(field.field_id)}"]`);
        if (el) markField(el, "skipped");
        manualCount++;
      }

      setStatus(
        `Done! ${filledCount} filled, ${manualCount} need attention.`
      );
    } catch (err) {
      console.error("[ScholarshipAssistant]", err);
      setStatus(`Error: ${err.message}`);
    }
  }

  function waitForWsMessage(ws) {
    return new Promise((resolve, reject) => {
      ws.onmessage = (event) => {
        resolve(JSON.parse(event.data));
      };
      ws.onerror = (err) => reject(new Error("WebSocket error"));
      ws.onclose = () => reject(new Error("WebSocket closed"));
    });
  }

  // ─── Floating Trigger Button ─────────────────────────────────────

  function injectButton() {
    const importUrl = `${BACKEND_URL}/profile/import`;
    const container = document.createElement("div");
    container.id = "sa-container";
    container.innerHTML = `
      <style>
        #sa-container {
          position: fixed;
          bottom: 20px;
          right: 20px;
          z-index: 2147483647;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        #sa-trigger {
          width: 56px;
          height: 56px;
          border-radius: 50%;
          background: #1a73e8;
          color: white;
          border: none;
          cursor: pointer;
          font-size: 24px;
          box-shadow: 0 4px 12px rgba(0,0,0,0.3);
          display: flex;
          align-items: center;
          justify-content: center;
          transition: background 0.2s, transform 0.2s;
        }
        #sa-trigger:hover {
          background: #1557b0;
          transform: scale(1.05);
        }
        #sa-trigger:active {
          transform: scale(0.95);
        }
        #sa-trigger.loading {
          background: #fb8c00;
          pointer-events: none;
        }
        #sa-status {
          position: absolute;
          bottom: 64px;
          right: 0;
          background: #333;
          color: #fff;
          padding: 8px 12px;
          border-radius: 8px;
          font-size: 13px;
          white-space: nowrap;
          opacity: 0;
          transition: opacity 0.2s;
          pointer-events: none;
        }
        #sa-status:not(:empty) {
          opacity: 1;
        }
        #sa-import-link {
          position: absolute;
          bottom: 64px;
          right: 0;
          font-size: 11px;
          color: #999;
          text-decoration: none;
          opacity: 0;
          transition: opacity 0.2s;
        }
        #sa-container:hover #sa-import-link {
          opacity: 1;
        }
        #sa-import-link:hover {
          color: #1a73e8;
        }
      </style>
      <div id="sa-status"></div>
      <a id="sa-import-link" href="${importUrl}" target="_blank" rel="noopener" title="Import Q&amp;A or raw text into profile">
        Import profile
      </a>
      <button id="sa-trigger" title="Scholarship Assistant — Click to auto-fill">
        🎓
      </button>
    `;

    document.body.appendChild(container);

    // Make draggable
    let isDragging = false;
    let dragOffsetX, dragOffsetY;
    let startX, startY;

    const btn = document.getElementById("sa-trigger");

    container.addEventListener("mousedown", (e) => {
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      dragOffsetX = e.clientX - container.getBoundingClientRect().left;
      dragOffsetY = e.clientY - container.getBoundingClientRect().top;
    });

    document.addEventListener("mousemove", (e) => {
      if (!isDragging) return;
      container.style.left = e.clientX - dragOffsetX + "px";
      container.style.top = e.clientY - dragOffsetY + "px";
      container.style.right = "auto";
      container.style.bottom = "auto";
    });

    document.addEventListener("mouseup", (e) => {
      if (isDragging) {
        isDragging = false;
      }
    });

    btn.addEventListener("click", (e) => {
      // Check if mouse moved significantly (drag vs click)
      const dist = Math.sqrt(
        Math.pow(e.clientX - startX, 2) + Math.pow(e.clientY - startY, 2)
      );
      if (dist > 5) return;

      if (btn.classList.contains("loading")) return;

      btn.classList.add("loading");
      runFillPipeline().finally(() => {
        btn.classList.remove("loading");
      });
    });
  }

  // ─── Init ────────────────────────────────────────────────────────

  // Check if backend is running before injecting
  GM_xmlhttpRequest({
    method: "GET",
    url: `${BACKEND_URL}/status`,
    onload: (response) => {
      try {
        const data = JSON.parse(response.responseText);
        if (data.status === "ok") {
          injectButton();
          if (!data.profile_exists) {
            console.log(
              "[ScholarshipAssistant] No profile found. Run the init interview first."
            );
          }
        }
      } catch (e) {
        console.log("[ScholarshipAssistant] Backend not responding properly.");
      }
    },
    onerror: () => {
      console.log(
        "[ScholarshipAssistant] Backend not running on " + BACKEND_URL
      );
    },
  });
})();
