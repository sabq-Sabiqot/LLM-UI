"""
CONNECTION.py
==============
Connects a natural language mission description to the UAV design pipeline.

The user describes their mission in plain English.
The LLM extracts structured mission parameters and passes them to POST_TRAIN.run().

Supported backends
------------------
  lmstudio    : LM Studio CLI local server (default, no internet, no token)
  qwen        : Qwen2.5-72B via HuggingFace Inference API
  qwen_small  : Qwen2.5-7B  via HuggingFace Inference API
  deepseek    : DeepSeek-R1-Distill-Qwen-7B  via HuggingFace
  deepseek_large : DeepSeek-R1-Distill-Qwen-32B via HuggingFace
  hf_custom   : any HuggingFace model  (set HF_MODEL below)
  ollama_qwen    : Qwen2.5 running locally via Ollama
  ollama_deepseek: DeepSeek running locally via Ollama
  ollama      : any other Ollama model
  openrouter  : OpenRouter cloud (free models available)

Quick start -- LM Studio CLI (local, no token)
----------------------------------------------
  1. irm https://lmstudio.ai/install.ps1 | iex
  2. lms get qwen3-4b
  3. lms load qwen3-4b
  4. lms server start --port 1234
  5. python llm_ui.py

Quick start -- HuggingFace (cloud, free token)
----------------------------------------------
  1. Get token: https://huggingface.co/settings/tokens
  2. set HF_TOKEN=hf_xxxxxxxxxxxx
  3. Set BACKEND = "qwen"
  4. python llm_ui.py
"""

import os
import re
import sys
import json
import requests
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# BACKEND CONFIG  <-- change BACKEND to switch model/provider
# =============================================================================

# Choose backend:
#   "lmstudio"       -- LM Studio CLI local server   (default)
#   "qwen"           -- Qwen2.5-72B  via HuggingFace
#   "qwen_small"     -- Qwen2.5-7B   via HuggingFace
#   "deepseek"       -- DeepSeek-R1-Distill-7B  via HuggingFace
#   "deepseek_large" -- DeepSeek-R1-Distill-32B via HuggingFace
#   "hf_custom"      -- any HuggingFace model  (set HF_MODEL below)
#   "ollama_qwen"    -- Qwen2.5 local via Ollama
#   "ollama_deepseek"-- DeepSeek local via Ollama
#   "ollama"         -- any Ollama model
#   "openrouter"     -- OpenRouter cloud
BACKEND = "lmstudio"

# HuggingFace token (required only for "qwen", "deepseek", "hf_custom")
# Get one free at https://huggingface.co/settings/tokens
# set HF_TOKEN=hf_xxxxxxxxxxxx   (Windows PowerShell)
# export HF_TOKEN=hf_xxxxxxxxxxxx  (Linux / Mac)
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Custom HuggingFace model ID (used when BACKEND = "hf_custom")
HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"

_HF_BASE = "https://api-inference.huggingface.co"

BACKEND_CONFIGS = {
    # ── LM Studio local ───────────────────────────────────────────────────
    "lmstudio": {
        "base_url"   : "http://localhost:1234",
        "endpoint"   : "/v1/chat/completions",
        # Paste exact name from: lms ps
        # Common Qwen3 names: qwen3-4b | qwen3-4b-instruct | qwen3-4b-instruct-q4_k_m
        "model"      : "qwen3-4b",
        "api_key"    : "lm-studio",   # LM Studio needs a value but ignores it
        "format"     : "openai",
        "gpu_layers" : -1,            # -1 = all layers on GPU (RTX 3050 4GB)
                                      # lower to 20 or 28 if VRAM runs out
        "notes"      : "Qwen3-4B via LM Studio CLI -- local, no internet",
    },

    # ── HuggingFace cloud ─────────────────────────────────────────────────
    "qwen": {
        "base_url" : _HF_BASE,
        "endpoint" : "/models/Qwen/Qwen2.5-72B-Instruct/v1/chat/completions",
        "model"    : "Qwen/Qwen2.5-72B-Instruct",
        "api_key"  : HF_TOKEN,
        "format"   : "hf_openai",
        "notes"    : "Qwen2.5-72B via HuggingFace Inference API (free tier)",
    },
    "qwen_small": {
        "base_url" : _HF_BASE,
        "endpoint" : "/models/Qwen/Qwen2.5-7B-Instruct/v1/chat/completions",
        "model"    : "Qwen/Qwen2.5-7B-Instruct",
        "api_key"  : HF_TOKEN,
        "format"   : "hf_openai",
        "notes"    : "Qwen2.5-7B via HuggingFace -- faster, less accurate",
    },
    "deepseek": {
        "base_url" : _HF_BASE,
        "endpoint" : "/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/v1/chat/completions",
        "model"    : "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "api_key"  : HF_TOKEN,
        "format"   : "hf_openai",
        "notes"    : "DeepSeek-R1-Distill-Qwen-7B via HuggingFace",
    },
    "deepseek_large": {
        "base_url" : _HF_BASE,
        "endpoint" : "/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/v1/chat/completions",
        "model"    : "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "api_key"  : HF_TOKEN,
        "format"   : "hf_openai",
        "notes"    : "DeepSeek-R1-Distill-Qwen-32B via HuggingFace -- best reasoning",
    },
    "hf_custom": {
        "base_url" : _HF_BASE,
        "endpoint" : f"/models/{HF_MODEL}/v1/chat/completions",
        "model"    : HF_MODEL,
        "api_key"  : HF_TOKEN,
        "format"   : "hf_openai",
        "notes"    : f"Custom HuggingFace model: {HF_MODEL}",
    },

    # ── Ollama local ──────────────────────────────────────────────────────
    "ollama_qwen": {
        "base_url" : "http://localhost:11434",
        "endpoint" : "/api/chat",
        "model"    : "qwen2.5:7b",        # ollama pull qwen2.5:7b
        "api_key"  : "",
        "format"   : "ollama",
        "notes"    : "Qwen2.5-7B running locally via Ollama",
    },
    "ollama_deepseek": {
        "base_url" : "http://localhost:11434",
        "endpoint" : "/api/chat",
        "model"    : "deepseek-r1:7b",    # ollama pull deepseek-r1:7b
        "api_key"  : "",
        "format"   : "ollama",
        "notes"    : "DeepSeek-R1 7B running locally via Ollama",
    },
    "ollama": {
        "base_url" : "http://localhost:11434",
        "endpoint" : "/api/chat",
        "model"    : "llama3.2",
        "api_key"  : "",
        "format"   : "ollama",
        "notes"    : "Any model via Ollama local server",
    },

    # ── OpenRouter cloud ──────────────────────────────────────────────────
    "openrouter": {
        "base_url" : "https://openrouter.ai/api",
        "endpoint" : "/v1/chat/completions",
        "model"    : "qwen/qwen-2.5-72b-instruct:free",
        "api_key"  : os.environ.get("OPENROUTER_API_KEY", ""),
        "format"   : "openai",
        "notes"    : "OpenRouter cloud (free Qwen model)",
    },
}

# =============================================================================
# MISSION PARAMETER SCHEMA
# =============================================================================

MISSION_SCHEMA = {
    "payload_kg"    : {"type": float, "default": 30.0,    "min": 0.1,   "max": 500.0,
                       "desc": "payload mass in kg"},
    "cruise_speed"  : {"type": float, "default": 18.0,    "min": 5.0,   "max": 150.0,
                       "desc": "cruise airspeed in m/s"},
    "altitude_m"    : {"type": float, "default": 500.0,   "min": 0.0,   "max": 10000.0,
                       "desc": "cruise altitude in metres"},
    "reynolds"      : {"type": float, "default": 300_000, "min": 10000, "max": 5_000_000,
                       "desc": "Reynolds number (auto-computed if not given)"},
    "mach"          : {"type": float, "default": 0.0,     "min": 0.0,   "max": 0.5,
                       "desc": "cruise Mach number"},
    "alpha"         : {"type": float, "default": 4.0,     "min": -4.0,  "max": 15.0,
                       "desc": "design angle of attack in degrees"},
    "alpha_start"   : {"type": float, "default": -4.0,    "min": -10.0, "max": 0.0,
                       "desc": "XFoil sweep start in degrees"},
    "alpha_end"     : {"type": float, "default": 16.0,    "min": 8.0,   "max": 25.0,
                       "desc": "XFoil sweep end in degrees"},
    "airfoil_method": {"type": str,   "default": "naca4",
                       "choices": ["naca4", "cst", "parsec"],
                       "desc": "airfoil parameterisation"},
    "optimize"      : {"type": bool,  "default": False,
                       "desc": "run geometry optimiser"},
    "optimizer"     : {"type": str,   "default": "bayesian",
                       "choices": ["bayesian", "genetic"],
                       "desc": "optimisation algorithm"},
    "objective"     : {"type": str,   "default": "breguet",
                       "choices": ["breguet", "max_LD", "max_CL", "min_CD"],
                       "desc": "optimisation objective"},
    "n_calls"       : {"type": int,   "default": 40,      "min": 5,     "max": 200,
                       "desc": "optimiser evaluation budget"},
    "payload_frac"  : {"type": float, "default": 0.28,    "min": 0.1,   "max": 0.5,
                       "desc": "payload / MTOW structural fraction"},
    "AR"            : {"type": float, "default": 10.0,    "min": 4.0,   "max": 20.0,
                       "desc": "wing aspect ratio"},
}

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

_SYSTEM_PROMPT = """You are a UAV design assistant. Extract mission parameters and reply in JSON.

RULES:
- Only include params the user mentioned. Use null for unknown.
- Convert units automatically: km/h/3.6=m/s, knots*0.5144=m/s, ft*0.3048=m, lb*0.4536=kg

PARAMS: payload_kg, cruise_speed(m/s), altitude_m, reynolds, mach, alpha(deg),
airfoil_method(naca4/cst/parsec), optimize(bool), optimizer(bayesian/genetic),
objective(breguet/max_LD/max_CL/min_CD), n_calls(int), AR(float)

ALWAYS reply with this exact JSON format (no extra text before it):
{"message":"your reply","params":{"payload_kg":30.0,"cruise_speed":18.0},"ready_to_run":true}

Set ready_to_run=true only when you have at least payload_kg AND cruise_speed.
"""

# =============================================================================
# SHARED RESPONSE UTILITIES
# =============================================================================

def _strip_think(content: str) -> str:
    """
    Strip <think>...</think> reasoning blocks produced by Qwen3 and DeepSeek-R1.
    Applied to every backend as a safety net.
    """
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def _check_content(data: dict, model: str) -> str:
    """
    Defensive content extraction from an OpenAI-compatible response dict.
    Raises ValueError with a clear message if the response is empty.
    """
    choices = data.get("choices", [])
    if not choices:
        raise ValueError(
            f"LLM returned empty choices.\n"
            f"  Model        : {model}\n"
            f"  Raw response : {str(data)[:300]}"
        )
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise ValueError(
            f"LLM returned empty content.\n"
            f"  Model         : {model}\n"
            f"  finish_reason : {choices[0].get('finish_reason')}\n"
            f"  Raw           : {str(data)[:300]}"
        )
    return content


# =============================================================================
# LLM CONNECTOR
# =============================================================================

class LLMConnector:
    """
    Unified connector for all supported backends.

    Formats:
      hf_openai  -- HuggingFace Inference API  (Qwen, DeepSeek)
      openai     -- OpenAI-compatible endpoint  (LM Studio, OpenRouter, llama.cpp)
      ollama     -- Ollama local server
    """

    def __init__(self, backend: str = BACKEND):
        cfg = BACKEND_CONFIGS.get(backend, BACKEND_CONFIGS["lmstudio"])
        self.backend    = backend
        self.base_url   = cfg["base_url"]
        self.endpoint   = cfg["endpoint"]
        self.model      = cfg["model"]
        self.api_key    = cfg["api_key"]
        self.fmt        = cfg["format"]
        self.gpu_layers = cfg.get("gpu_layers", None)
        self.notes      = cfg.get("notes", "")
        self.url        = self.base_url + self.endpoint
        self.history    = []
        self._add_system()

    def _add_system(self):
        self.history = [{"role": "system", "content": _SYSTEM_PROMPT}]

    def reset(self):
        self._add_system()

    # ── HuggingFace Inference API ─────────────────────────────────────────

    def _call_hf_openai(self, messages: list) -> str:
        """
        Call HuggingFace Inference API (OpenAI-compatible endpoint).
        Endpoint: /models/<org>/<model>/v1/chat/completions
        Requires a free HuggingFace token.

        Fixes applied:
        - max_tokens = 1024  (DeepSeek/Qwen think blocks need room)
        - <think> stripping  (DeepSeek-R1 and Qwen3 reasoning blocks)
        - defensive content check with clear error messages
        """
        if not self.api_key:
            raise ValueError(
                "HuggingFace token not set.\n"
                "  Get one free at https://huggingface.co/settings/tokens\n"
                "  Then set:  set HF_TOKEN=hf_xxxxxxxxxxxx  (Windows)\n"
                "             export HF_TOKEN=hf_xxxxxxxxxxxx  (Linux/Mac)"
            )

        headers = {
            "Content-Type" : "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model"      : self.model,
            "messages"   : messages,
            "temperature": 0.2,
            "max_tokens" : 1024,   # large enough for <think> block + JSON
            "stream"     : False,
        }

        resp = requests.post(self.url, json=payload,
                             headers=headers, timeout=300)

        # Surface HF-specific errors clearly
        if resp.status_code == 401:
            raise ValueError(
                "HuggingFace token invalid or expired.\n"
                "  Get a new token at https://huggingface.co/settings/tokens"
            )
        if resp.status_code == 403:
            raise ValueError(
                f"Access denied to model {self.model}.\n"
                "  Some models require accepting a license on HuggingFace.co.\n"
                f"  Visit https://huggingface.co/{self.model} and accept terms."
            )
        if resp.status_code == 503:
            raise ConnectionError(
                f"Model {self.model} is loading on HuggingFace servers.\n"
                "  Wait ~20 seconds and try again."
            )

        resp.raise_for_status()

        # Defensive content check
        content = _check_content(resp.json(), self.model)

        # Strip <think> reasoning blocks (DeepSeek-R1 and Qwen3)
        return _strip_think(content)

    # ── Ollama local ──────────────────────────────────────────────────────

    def _call_ollama(self, messages: list) -> str:
        """
        Call Ollama /api/chat endpoint.

        Fixes applied:
        - max_tokens = 1024
        - <think> stripping (Qwen3 / DeepSeek via Ollama also use think tags)
        - defensive content check
        """
        payload = {
            "model"   : self.model,
            "messages": messages,
            "stream"  : False,
            "options" : {"num_predict": 1024},
        }
        resp = requests.post(self.url, json=payload, timeout=300)
        resp.raise_for_status()

        data    = resp.json()
        content = data.get("message", {}).get("content", "")
        if not content:
            raise ValueError(
                f"Ollama returned empty content.\n"
                f"  Model : {self.model}\n"
                f"  Raw   : {str(data)[:300]}"
            )

        return _strip_think(content)

    # ── OpenAI-compatible (LM Studio, OpenRouter, llama.cpp) ─────────────

    def _call_openai(self, messages: list) -> str:
        """
        Call any OpenAI-compatible /v1/chat/completions endpoint.

        Fixes applied:
        - max_tokens = 1024  (enough for <think> block + JSON output)
        - thinking disabled  via extra_body (Qwen3 LM Studio)
        - gpu_layers         via extra_body (LM Studio RTX 3050 offload)
        - <think> stripping  (safety net for Qwen3 / DeepSeek)
        - defensive content check with clear error messages
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model"      : self.model,
            "messages"   : messages,
            "temperature": 0.2,
            "max_tokens" : 1024,   # large enough to fit <think> block + JSON
            "stream"     : False,
        }

        # extra_body -- LM Studio specific options
        extra = {}

        # Disable Qwen3 thinking mode so output is short and JSON-only.
        # If LM Studio version does not support this key it is silently ignored.
        extra["thinking"] = {"type": "disabled"}

        # GPU offload -- RTX 3050 4GB: gpu_layers = -1 (all layers on GPU)
        if self.gpu_layers is not None:
            extra["gpu_layers"] = self.gpu_layers

        if extra:
            payload["extra_body"] = extra

        resp = requests.post(self.url, json=payload,
                             headers=headers, timeout=300)

        # Surface LM Studio / server errors clearly
        if resp.status_code == 404:
            raise ConnectionError(
                f"Model '{self.model}' not found.\n"
                f"  Check loaded model : lms ps\n"
                f"  Load it with       : lms load {self.model}\n"
                f"  Restart server     : lms server start --port 1234"
            )
        if resp.status_code == 503:
            raise ConnectionError(
                "Server returned 503 -- model may still be loading.\n"
                "  Wait 10-20 seconds and try again."
            )

        resp.raise_for_status()

        # Defensive content check
        content = _check_content(resp.json(), self.model)

        # Strip any residual <think> blocks (Qwen3 safety net)
        return _strip_think(content)

    # ── Dispatch ──────────────────────────────────────────────────────────

    def _call(self, messages: list) -> str:
        if self.fmt == "hf_openai":
            return self._call_hf_openai(messages)
        elif self.fmt == "ollama":
            return self._call_ollama(messages)
        else:
            return self._call_openai(messages)

    # ── Public chat ───────────────────────────────────────────────────────

    def chat(self, user_message: str) -> dict:
        """
        Send one user message, get response, extract mission params.

        Returns dict with keys:
            message      : clean conversational reply
            params       : extracted and validated mission parameters
            ready_to_run : True when pipeline can be started
            raw          : full raw LLM response text
            error        : error type string if something went wrong
        """
        self.history.append({"role": "user", "content": user_message})

        try:
            raw = self._call(self.history)

        except ValueError as e:
            # Token / auth errors or empty response
            return {
                "message"      : (
                    f"LLM returned an empty or invalid response.\n\n"
                    f"Details: {e}\n\n"
                    "Try these fixes:\n"
                    f"  1. Check model name matches exactly:  lms ps\n"
                    f"     Current model in config: {self.model}\n"
                    "  2. Try a smaller model: lms load qwen3-1.7b\n"
                    "  3. Restart LM Studio:   lms server stop && lms server start"
                ),
                "params"       : {},
                "ready_to_run" : False,
                "raw"          : str(e),
                "error"        : "empty_response",
            }
        except requests.exceptions.ConnectionError:
            return {
                "message"      : (
                    f"Cannot connect to {self.backend} at {self.base_url}.\n\n"
                    f"{self._connection_tips()}"
                ),
                "params"       : {},
                "ready_to_run" : False,
                "raw"          : "",
                "error"        : "connection_error",
            }
        except requests.exceptions.ReadTimeout:
            return {
                "message"      : (
                    f"LLM timed out after 300s.\n\n"
                    "Try these fixes:\n"
                    "  1. Use a smaller model: lms load qwen3-1.7b\n"
                    "  2. Check GPU is active: nvidia-smi\n"
                    "  3. Restart the server:  lms server stop && lms server start"
                ),
                "params"       : {},
                "ready_to_run" : False,
                "raw"          : "",
                "error"        : "timeout",
            }
        except ConnectionError as e:
            return {
                "message"      : str(e),
                "params"       : {},
                "ready_to_run" : False,
                "raw"          : "",
                "error"        : "server_error",
            }
        except Exception as e:
            return {
                "message"      : f"LLM error: {e}",
                "params"       : {},
                "ready_to_run" : False,
                "raw"          : str(e),
                "error"        : str(e),
            }

        self.history.append({"role": "assistant", "content": raw})
        return _parse_llm_response(raw)

    def _connection_tips(self) -> str:
        if self.fmt == "hf_openai":
            return (
                "Check your internet connection and HF_TOKEN.\n"
                "  Get a token at https://huggingface.co/settings/tokens"
            )
        elif self.fmt == "ollama":
            return (
                f"Start Ollama:      ollama serve\n"
                f"Pull model:        ollama pull {self.model}"
            )
        else:
            return (
                f"Start LM Studio:   lms load {self.model}\n"
                f"                   lms server start --port 1234"
            )

    # ── Availability check ────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            if self.fmt == "hf_openai":
                if not self.api_key:
                    return False
                r = requests.head(
                    f"{self.base_url}/models/{self.model}",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5,
                )
                return r.status_code in (200, 302, 307)
            elif self.fmt == "ollama":
                r = requests.get(self.base_url + "/api/tags", timeout=3)
                return r.status_code == 200
            else:
                r = requests.get(self.base_url + "/v1/models", timeout=3)
                return r.status_code == 200
        except Exception:
            return False

    def available_models(self) -> list:
        try:
            if self.fmt == "ollama":
                r = requests.get(self.base_url + "/api/tags", timeout=5)
                if r.status_code == 200:
                    return [m["name"] for m in r.json().get("models", [])]
            elif self.fmt == "hf_openai":
                return [self.model]
            else:
                r = requests.get(self.base_url + "/v1/models", timeout=5)
                if r.status_code == 200:
                    return [m["id"] for m in r.json().get("data", [])]
        except Exception:
            pass
        return []


# =============================================================================
# RESPONSE PARSER
# =============================================================================

def _parse_llm_response(raw: str) -> dict:
    """
    Extract JSON from LLM response text.
    Handles three patterns:
      1. ```json { ... } ```  markdown code block
      2. { ... "params" ... } bare JSON with params key
      3. last { ... } block in response
    """
    json_str = None

    # Pattern 1: ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        json_str = m.group(1)

    # Pattern 2: bare JSON containing "params" key
    if not json_str:
        m = re.search(r"(\{[^{}]*\"params\"[^{}]*\})", raw, re.DOTALL)
        if m:
            json_str = m.group(1)

    # Pattern 3: last { ... } block in response
    if not json_str:
        for start in reversed(list(re.finditer(r"\{", raw))):
            candidate = raw[start.start():]
            depth = 0
            for i, ch in enumerate(candidate):
                if   ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        json_str = candidate[:i + 1]
                        break
            if json_str:
                break

    if json_str:
        try:
            data = json.loads(json_str)

            # Build a clean readable message
            msg = data.get("message", "")
            if not msg or msg.strip().startswith("{"):
                params = data.get("params", {})
                if params:
                    parts = [f"{k} = {v}" for k, v in params.items()]
                    msg = "Got it! Extracted: " + ", ".join(parts) + "."
                else:
                    msg = raw.replace(json_str, "").strip()
            if not msg:
                msg = "Parameters extracted."

            return {
                "message"      : msg,
                "params"       : _validate_params(data.get("params", {})),
                "ready_to_run" : bool(data.get("ready_to_run", False)),
                "raw"          : raw,
            }
        except json.JSONDecodeError:
            pass

    # Fallback: no JSON found -- return raw text and try regex extraction
    return {
        "message"      : raw,
        "params"       : _extract_params_from_text(raw),
        "ready_to_run" : False,
        "raw"          : raw,
    }


def _validate_params(params: dict) -> dict:
    """Validate and clamp extracted params against MISSION_SCHEMA."""
    validated = {}
    for key, val in params.items():
        if val is None or key not in MISSION_SCHEMA:
            continue
        schema = MISSION_SCHEMA[key]
        try:
            val = schema["type"](val)
        except (TypeError, ValueError):
            continue
        if "min" in schema:
            val = max(schema["min"], val)
        if "max" in schema:
            val = min(schema["max"], val)
        if "choices" in schema and val not in schema["choices"]:
            val = schema["default"]
        validated[key] = val
    return validated


def _extract_params_from_text(text: str) -> dict:
    """
    Fallback regex extraction when LLM does not return JSON.
    """
    params   = {}
    patterns = [
        (r"(\d+(?:\.\d+)?)\s*kg\s*payload",     "payload_kg",   float),
        (r"payload\s+of\s+(\d+(?:\.\d+)?)\s*kg", "payload_kg",  float),
        (r"(\d+(?:\.\d+)?)\s*m/s",               "cruise_speed", float),
        (r"(\d+(?:\.\d+)?)\s*km/h",              "cruise_speed", lambda x: float(x) / 3.6),
        (r"(\d+(?:\.\d+)?)\s*knots?",            "cruise_speed", lambda x: float(x) * 0.5144),
        (r"(\d+(?:\.\d+)?)\s*m\s+altitude",      "altitude_m",   float),
        (r"altitude\s+of\s+(\d+(?:\.\d+)?)\s*m", "altitude_m",  float),
        (r"(\d+(?:\.\d+)?)\s*ft",                "altitude_m",   lambda x: float(x) * 0.3048),
        (r"re\s*=\s*(\d+(?:\.\d+)?)",            "reynolds",     float),
        (r"reynolds\s+(\d+(?:\.\d+)?)",           "reynolds",    float),
    ]
    for pattern, key, converter in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                params[key] = converter(m.group(1))
            except Exception:
                pass
    return params


# =============================================================================
# MISSION BUILDER
# =============================================================================

def build_mission(llm_params : dict,
                  xfoil_path : str = r"./Xfoil6.99/xfoil.exe",
                  model_path : str = "uav_model.pkl",
                  output_dir : str = "connect_output") -> dict:
    """
    Merge LLM-extracted params with schema defaults to produce a complete
    POST_TRAIN-compatible mission dict.
    """
    mission = {k: v["default"] for k, v in MISSION_SCHEMA.items()}
    mission.update(llm_params)

    mission["model_path"]    = model_path
    mission["xfoil_path"]    = xfoil_path
    mission["output_dir"]    = output_dir
    mission["use_generator"] = True
    mission["payload_frac"]  = mission.get("payload_frac", 0.28)
    mission["alpha_step"]    = mission.get("alpha_step",    1.0)
    mission["taper"]         = mission.get("taper",         0.45)
    mission["fuse_frac"]     = mission.get("fuse_frac",     0.60)
    mission["n_init"]        = max(5, mission.get("n_calls", 40) // 5)
    mission["pop_size"]      = 20
    mission["verbose"]       = True

    return mission


# =============================================================================
# PROCESS MESSAGE  (called from llm_ui.py)
# =============================================================================

def process_message(message    : str,
                    connector  : LLMConnector,
                    xfoil_path : str  = r"./Xfoil6.99/xfoil.exe",
                    model_path : str  = "uav_model.pkl",
                    output_dir : str  = "connect_output",
                    auto_run   : bool = False) -> dict:
    """
    Process one user message through the LLM and optionally run the pipeline.
    """
    response = connector.chat(message)
    reply    = response["message"]
    params   = response.get("params", {})
    ready    = response.get("ready_to_run", False)

    pipeline_result = None
    mission         = None

    if ready and (auto_run or _user_confirmed(message)):
        mission = build_mission(params, xfoil_path, model_path, output_dir)

        reply += "\n\nRunning UAV pipeline with:\n"
        reply += json.dumps(
            {k: v for k, v in mission.items()
             if k in MISSION_SCHEMA or k in ("xfoil_path", "output_dir")},
            indent=2,
        )

        try:
            from ...ML_Calculation.Airfoil_Prediction import POST_TRAIN
            pipeline_result = POST_TRAIN.run(mission)
            geo  = pipeline_result.get("geometry", {})
            reply += (
                f"\n\nDone! Results:\n"
                f"  MTOW        : {geo.get('MTOW_kg',    '?')} kg\n"
                f"  Wingspan    : {geo.get('wingspan_m', '?')} m\n"
                f"  Wing area   : {geo.get('S_m2',       '?')} m2\n"
                f"  Motor power : {geo.get('P_motor_W',  '?')} W\n"
                f"  Report      : {mission['output_dir']}/post_train_report.txt"
            )
        except Exception as e:
            reply          += f"\n\nPipeline error: {e}"
            pipeline_result = {"error": str(e)}

    return {
        "reply"          : reply,
        "params"         : params,
        "ready_to_run"   : ready,
        "pipeline_result": pipeline_result,
        "mission"        : mission,
        "error"          : response.get("error"),
    }


def _user_confirmed(message: str) -> bool:
    words = ["yes", "run", "go", "start", "execute",
             "confirm", "proceed", "do it", "launch"]
    return any(w in message.lower() for w in words)


# =============================================================================
# CLI  (test without UI)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  UAV LLM Mission Interface  (CLI mode)")
    print("=" * 60)
    print(f"  Backend : {BACKEND}")

    conn = LLMConnector(BACKEND)
    print(f"  Model   : {conn.model}")
    print(f"  URL     : {conn.url}")
    print(f"  Note    : {conn.notes}")
    print("=" * 60)

    # Backend-specific startup hints
    if conn.fmt == "hf_openai":
        if not HF_TOKEN:
            print("\nERROR: HF_TOKEN not set.")
            print("  1. Get a free token at https://huggingface.co/settings/tokens")
            print("  2. set HF_TOKEN=hf_xxxxxxxxxxxx   (Windows)")
            print("     export HF_TOKEN=hf_xxxxxxxxxxxx  (Linux/Mac)")
            print("  3. Re-run this script.")
            sys.exit(1)
        if not conn.is_available():
            print("\nWARNING: Cannot reach HuggingFace API.")
            print("  Check your internet connection and token.")
        else:
            print("\nHuggingFace API ready.")
    else:
        if not conn.is_available():
            print(f"\nWARNING: Backend not reachable at {conn.base_url}")
            print(f"  {conn._connection_tips()}")
        else:
            models = conn.available_models()
            print(f"\nBackend ready.  Models: {models}")

    print("\nDescribe your UAV mission in plain English.")
    print("Commands: quit | reset | models\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            conn.reset()
            print("Conversation reset.\n")
            continue
        if user_input.lower() == "models":
            print("Models:", conn.available_models())
            print()
            continue

        result = process_message(
            user_input, conn,
            xfoil_path = r"./Xfoil6.99/xfoil.exe",
            model_path = "uav_model.pkl",
            output_dir = "connect_output",
            auto_run   = False,
        )

        print(f"\nAssistant: {result['reply']}")
        if result["params"]:
            print(f"Extracted : {json.dumps(result['params'], indent=2)}")
        if result["ready_to_run"]:
            print("[Ready -- type 'run' to execute pipeline]")
        print()