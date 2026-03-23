/**
 * Index.js  --  Engineering Generator
 * Backend: LM Studio local server (no internet, no token)
 */

// =============================================================================
// State
// =============================================================================

let accumulatedParams = {};
let readyToRun        = false;
let pipelineRunning   = false;

// =============================================================================
// Initialisation
// =============================================================================

window.onload = function () {
  checkStatus();
  setInterval(checkStatus, 15000);
  document.getElementById("user-input").focus();
};

// =============================================================================
// LM Studio status check
// =============================================================================

function checkStatus() {
  fetch("/status")
    .then(r => r.json())
    .then(d => {
      const dot   = document.getElementById("status-dot");
      const txt   = document.getElementById("status-text");
      const badge = document.getElementById("session-badge");

      if (d.available) {
        dot.className   = "status-dot online";
        txt.textContent = d.model + " (online)";
        txt.style.color = "#4caf50";

      } else if (d.backend_fmt === "hf_openai" && d.hf_token_set === false) {
        // HuggingFace token missing
        dot.className   = "status-dot offline";
        txt.textContent = "HF token not set";
        txt.style.color = "#f97316";
        if (!window._hfWarnShown) {
          window._hfWarnShown = true;
          appendMsg("error",
            "HuggingFace token not set.\n\n" +
            "Set HF_TOKEN before starting the server:\n" +
            "  Windows: set HF_TOKEN=hf_xxxxxxxxxxxx\n" +
            "  Linux  : export HF_TOKEN=hf_xxxxxxxxxxxx\n\n" +
            "Get a free token at: https://huggingface.co/settings/tokens"
          );
        }

      } else if (d.backend_fmt === "hf_openai" && d.hf_token_set === true) {
        // Token set but model unreachable (loading or network issue)
        dot.className   = "status-dot offline";
        txt.textContent = "HF model loading...";
        txt.style.color = "#f97316";

      } else {
        dot.className   = "status-dot offline";
        txt.textContent = "LLM offline";
        txt.style.color = "#f44336";
        if (!window._offlineShown) {
          window._offlineShown = true;
          appendMsg("error",
            "LLM backend not running.\n\n" +
            (d.notes ? "Backend: " + d.notes + "\n\n" : "") +
            "For LM Studio:\n" +
            "  lms load qwen3-4b\n" +
            "  lms server start --port 1234"
          );
        }
      }

      // Session badge
      if (d.session_id) {
        badge.textContent = "Session " + d.session_id +
          (d.active_sessions > 1
            ? "  |  " + d.active_sessions + " active"
            : "");
      }
    })
    .catch(() => {
      document.getElementById("status-dot").className = "status-dot offline";
      document.getElementById("status-text").textContent = "Server error";
    });
}

// =============================================================================
// Send message
// =============================================================================

function sendMessage() {
  const input = document.getElementById("user-input");
  const text  = input.value.trim();
  if (!text || pipelineRunning) return;

  appendMsg("user", text);
  input.value = "";
  autoResize(input);
  hideSuggestions();
  showTyping();
  setDisabled(true);

  fetch("/chat", {
    method  : "POST",
    headers : { "Content-Type": "application/json" },
    body    : JSON.stringify({ message: text }),
  })
  .then(r => r.json())
  .then(d => {
    removeTyping();
    setDisabled(false);

    if (d.error === "connection_error") {
      appendMsg("error", d.reply);
      return;
    }

    if (d.error === "timeout") {
      appendMsg("error", d.reply);
      return;
    }

    if (d.error === "empty_response") {
      appendMsg("error", d.reply);
      return;
    }

    if (d.error === "pipeline_busy") {
      appendMsg("system", "Pipeline is still running. Please wait.");
      return;
    }

    // Build reply HTML
    let html = escHtml(d.reply);

    // Show accumulated params
    if (d.params && Object.keys(d.params).length > 0) {
      updateParams(d.params);
      html += "<pre>" + escHtml(JSON.stringify(d.params, null, 2)) + "</pre>";
    }

    // Show ready badge and Run button
    if (d.ready_to_run) {
      html += '<br><span class="ready-badge">Ready to run pipeline</span>';
      readyToRun = true;
      document.getElementById("run-btn").classList.add("visible");
    }

    // Show pipeline result card if pipeline already ran
    if (d.pipeline_result) {
      html += buildResultCard(d.pipeline_result);
    }

    appendMsg("assistant", html, true);
    input.focus();
  })
  .catch(err => {
    removeTyping();
    setDisabled(false);
    appendMsg("error", "Request failed: " + err);
  });
}

// =============================================================================
// Run pipeline (explicit Run button)
// =============================================================================

function runPipeline() {
  if (pipelineRunning || !readyToRun) return;

  appendMsg("user", "Run the pipeline with the current parameters.");
  showTyping();
  setDisabled(true);

  fetch("/run", {
    method  : "POST",
    headers : { "Content-Type": "application/json" },
    body    : JSON.stringify({ params: accumulatedParams }),
  })
  .then(r => r.json())
  .then(d => {
    removeTyping();
    setDisabled(false);

    if (d.status === "ok") {
      const html = "Pipeline complete!" + buildResultCard({ result: d.result });
      appendMsg("assistant", html, true);
    } else if (d.status === "busy") {
      appendMsg("system", "Pipeline is already running for your session.");
    } else {
      appendMsg("error", "Pipeline error: " + (d.error || "unknown"));
    }
  })
  .catch(err => {
    removeTyping();
    setDisabled(false);
    appendMsg("error", "Run failed: " + err);
  });
}

// =============================================================================
// Pipeline result card
// =============================================================================

function buildResultCard(pipelineResult) {
  const geo = pipelineResult.geometry
           || pipelineResult.result
           || {};

  if (!geo || Object.keys(geo).length === 0) return "";

  const rows = [
    ["MTOW",         (geo.MTOW_kg    || "?") + " kg"],
    ["Wingspan",     (geo.wingspan_m || "?") + " m"],
    ["Wing area",    (geo.S_m2       || "?") + " m2"],
    ["Aspect ratio", (geo.AR         || "?")],
    ["Mean chord",   (geo.c_mean_m   || "?") + " m"],
    ["Fuselage len", (geo.l_fus_m    || "?") + " m"],
    ["Motor power",  (geo.P_motor_W  || "?") + " W"],
  ];

  if (geo.CL)      rows.push(["CL",      geo.CL.toFixed(4)]);
  if (geo.LD)      rows.push(["L/D",     geo.LD.toFixed(2)]);
  if (geo.breguet) rows.push(["Breguet", geo.breguet.toFixed(3)]);

  let html = '<div class="result-card">';
  html    += '<div class="result-title">UAV Geometry Synthesis</div>';
  rows.forEach(function(row) {
    html += '<div class="result-row">'
          + '<span>' + escHtml(row[0]) + '</span>'
          + '<span>' + escHtml(String(row[1])) + '</span>'
          + '</div>';
  });
  html += '</div>';
  return html;
}

// =============================================================================
// Reset conversation
// =============================================================================

function resetChat() {
  fetch("/reset", { method: "POST" })
    .then(function() {
      document.getElementById("messages").innerHTML =
        '<div class="msg system">Conversation reset.</div>';
      accumulatedParams    = {};
      readyToRun           = false;
      window._offlineShown = false;
      resetParamDisplay();
      document.getElementById("run-btn").classList.remove("visible");
      showSuggestions();
    })
    .catch(function(err) { appendMsg("error", "Reset failed: " + err); });
}

// =============================================================================
// DOM helpers
// =============================================================================

function appendMsg(role, content, isHtml) {
  var div     = document.createElement("div");
  div.className = "msg " + role;
  if (isHtml) { div.innerHTML  = content; }
  else        { div.textContent = content; }
  document.getElementById("messages").appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "end" });
}

function showTyping() {
  var t     = document.createElement("div");
  t.id        = "typing";
  t.className = "typing";
  t.innerHTML = "<span></span><span></span><span></span>";
  document.getElementById("messages").appendChild(t);
  t.scrollIntoView({ behavior: "smooth", block: "end" });
}

function removeTyping() {
  var t = document.getElementById("typing");
  if (t) { t.remove(); }
}

function setDisabled(state) {
  document.getElementById("send-btn").disabled   = state;
  document.getElementById("user-input").disabled = state;
  pipelineRunning = state;
}

function updateParams(params) {
  Object.assign(accumulatedParams, params);
  Object.entries(accumulatedParams).forEach(function(entry) {
    var el = document.getElementById("p-" + entry[0]);
    if (el) {
      el.textContent = String(entry[1]);
      el.classList.remove("unset");
    }
  });
}

function resetParamDisplay() {
  document.querySelectorAll(".param-value").forEach(function(el) {
    el.textContent = "--";
    el.classList.add("unset");
  });
}

function handleKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 160) + "px";
}

function suggest(el) {
  document.getElementById("user-input").value = el.textContent;
  sendMessage();
}

function hideSuggestions() {
  document.getElementById("suggestions").style.display = "none";
}

function showSuggestions() {
  document.getElementById("suggestions").style.display = "flex";
}

function escHtml(str) {
  return String(str)
    .replace(/&/g,  "&amp;")
    .replace(/</g,  "&lt;")
    .replace(/>/g,  "&gt;")
    .replace(/"/g,  "&quot;")
    .replace(/\n/g, "<br>");
}