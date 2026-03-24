"""
llm_mission.py
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

_SYSTEM_PROMPT = """You are a UAV aerodynamic design assistant.

Your only job is to extract UAV mission parameters from the user message and return valid JSON.

STEP 1 - Read the user message and find these parameters if mentioned:
  payload_kg      : payload mass in kg
  cruise_speed    : airspeed in m/s  (convert: km/h divide by 3.6, knots multiply by 0.5144)
  altitude_m      : altitude in metres  (convert: ft multiply by 0.3048)
  reynolds        : Reynolds number
  mach            : Mach number
  alpha           : angle of attack in degrees
  airfoil_method  : one of naca4, cst, parsec
  optimize        : true or false
  optimizer       : one of bayesian, genetic
  objective       : one of breguet, max_LD, max_CL, min_CD
  n_calls         : integer number of optimiser evaluations
  AR              : wing aspect ratio as a number

STEP 2 - Build the params object using ONLY values the user mentioned.
  Do NOT include parameters the user did not mention.
  Do NOT use placeholder text. Use real numbers only.

STEP 3 - Reply with this JSON structure and nothing else:
  {
    "message": "write a short friendly reply here",
    "params": {
      "payload_kg": 50.0,
      "altitude_m": 1000.0
    },
    "ready_to_run": false
  }

  Set ready_to_run to true only when BOTH payload_kg AND cruise_speed are known.

EXAMPLE - if user says "50 kg drone at 1000 m altitude":
  {
    "message": "Got it! I have your payload and altitude. What is the cruise speed?",
    "params": {
      "payload_kg": 50.0,
      "altitude_m": 1000.0
    },
    "ready_to_run": false
  }

EXAMPLE - if user says "30 kg payload, 18 m/s, 500 m altitude":
  {
    "message": "All minimum parameters received. Ready to run the UAV pipeline.",
    "params": {
      "payload_kg": 30.0,
      "cruise_speed": 18.0,
      "altitude_m": 500.0
    },
    "ready_to_run": true
  }

EXAMPLE - if user says "50 kg surveillance drone at 1000 m":
  {
    "message": "Got it! I have your payload and altitude. What cruise speed do you need?",
    "params": {
      "payload_kg": 50.0,
      "altitude_m": 1000.0
    },
    "ready_to_run": false
  }

CRITICAL RULES:
  - params must ALWAYS contain the values the user mentioned as real numbers.
  - NEVER write "params": , -- that is invalid JSON and will crash the system.
  - NEVER write "params": null -- use "params": {} if truly nothing was mentioned.
  - NEVER copy example text verbatim. Replace with real numbers from the user message.
  - The JSON must be valid and parseable. Test it mentally before writing.
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

    LM Studio with Qwen3 splits the response into two fields:
      content          -- the actual reply (what we want)
      reasoning_content -- the <think> chain-of-thought block

    When finish_reason=length the model ran out of max_tokens while thinking,
    leaving content empty. We fall back to reasoning_content in that case
    and extract the JSON from it directly.

    Raises ValueError with a clear message only if both fields are empty.
    """
    choices = data.get("choices", [])
    if not choices:
        raise ValueError(
            f"LLM returned empty choices.\n"
            f"  Model        : {model}\n"
            f"  Raw response : {str(data)[:300]}"
        )

    message       = choices[0].get("message", {})
    content       = message.get("content", "") or ""
    reasoning     = message.get("reasoning_content", "") or ""
    finish_reason = choices[0].get("finish_reason", "")

    # Primary: use content if it has text
    if content.strip():
        return content.strip()

    # Fallback: finish_reason=length means tokens ran out during thinking.
    # The reasoning_content field contains the partial chain-of-thought.
    # Try to extract a JSON block from it directly.
    if reasoning.strip():
        # Look for a JSON block inside the reasoning text
        import re as _re
        m = _re.search(r"\{.*?\}", reasoning, _re.DOTALL)
        if m:
            return m.group(0)
        # No JSON found in reasoning -- return reasoning text so the
        # parser can attempt regex extraction from natural language
        return reasoning.strip()

    # Both fields are empty -- raise a clear error
    raise ValueError(
        f"LLM returned empty content.\n"
        f"  Model         : {model}\n"
        f"  finish_reason : {finish_reason}\n"
        f"  Tip: increase max_tokens or disable thinking mode\n"
        f"  Raw           : {str(data)[:300]}"
    )


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
            "max_tokens" : 2048,   # increased: Qwen3 think block needs ~1000
                                   # tokens alone, then JSON needs ~200 more
            "stream"     : False,
        }

        # extra_body -- LM Studio specific options
        extra = {}

        # Disable Qwen3 thinking mode so output is short and JSON-only.
        # "budget_tokens": 0 sets thinking token budget to zero (fastest).
        # "type": "disabled" is the LM Studio / OpenAI-style flag.
        # Both are sent -- whichever the server supports will take effect.
        extra["thinking"] = {"type": "disabled", "budget_tokens": 0}

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
        # Append /no_think to disable Qwen3 chain-of-thought per turn.
        # This forces the model to reply directly in JSON without a <think> block.
        # For non-Qwen models this suffix is harmless -- they ignore it.
        user_with_hint = user_message.strip()
        if not user_with_hint.endswith("/no_think"):
            user_with_hint = user_with_hint + " /no_think"

        self.history.append({"role": "user", "content": user_with_hint})

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

def _sanitise_json(text: str) -> str:
    """
    Fix common malformed JSON patterns produced by small LLMs.

    Patterns fixed:
      "params":,          -> "params": {}
      "params": ,         -> "params": {}
      "params":null,      -> "params": {}
      "params": null,     -> "params": {}
      "ready_to_run":,    -> "ready_to_run": false
      trailing commas     -> removed before closing brace/bracket
      "message":"your reply"  -> "message": "I need more information."
    """
    # Fix "params": null,  ->  "params": {},
    # Fix "params": ,      ->  "params": {},
    # The replacement includes the trailing comma to avoid double-commas
    text = re.sub(r'"params"\s*:\s*null\s*,', '"params": {},', text)
    text = re.sub(r'"params"\s*:\s*,',         '"params": {},', text)
    # Fix "ready_to_run": ,  ->  "ready_to_run": false
    text = re.sub(r'"ready_to_run"\s*:\s*,', '"ready_to_run": false,', text)
    # Remove double commas produced by replacements above
    text = re.sub(r',\s*,', ',', text)
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    # Fix literal placeholder messages
    text = text.replace('"your reply"',                    '"I need more information."')
    text = text.replace('"write a short friendly reply here"', '"I need more information."')
    return text


def _parse_llm_response(raw: str) -> dict:
    """
    Extract JSON from LLM response text.

    Handles four patterns:
      1. ```json { ... } ```  markdown code block
      2. { ... "params" ... } bare JSON with params key
      3. last { ... } block in response
      4. malformed JSON repaired by _sanitise_json()
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

    # Pattern 3: find the OUTERMOST { ... } block that contains "params"
    # Iterate forward (not reversed) so we get the enclosing object, not inner ones
    if not json_str:
        for start in re.finditer(r"\{", raw):
            candidate = raw[start.start():]
            depth = 0
            end_idx = None
            for i, ch in enumerate(candidate):
                if   ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            if end_idx is not None:
                block = candidate[:end_idx + 1]
                # Only accept blocks that contain both "params" and "message"
                if '"params"' in block and '"message"' in block:
                    json_str = block
                    break

    if json_str:
        # Pattern 4: try to parse, if it fails sanitise and retry
        data = None
        for attempt in (json_str, _sanitise_json(json_str)):
            try:
                data = json.loads(attempt)
                break
            except json.JSONDecodeError:
                continue

        if data is not None:
            # Build a clean readable message
            msg = data.get("message", "")
            placeholder_phrases = (
                "your reply", "write a short", "I need more information"
            )
            if not msg or msg.strip().startswith("{") or                any(p in msg for p in placeholder_phrases):
                params = data.get("params") or {}
                if params:
                    parts = [f"{k} = {v}" for k, v in params.items()]
                    msg = "Got it! Extracted: " + ", ".join(parts) + "."
                else:
                    # try to pull text before the JSON block
                    msg = raw.replace(json_str, "").strip()
            if not msg:
                msg = "Parameters extracted."

            params = data.get("params") or {}   # handle null params
            return {
                "message"      : msg,
                "params"       : _validate_params(params),
                "ready_to_run" : bool(data.get("ready_to_run", False)),
                "raw"          : raw,
            }

    # Fallback: no valid JSON at all -- use regex extraction on raw text
    extracted = _extract_params_from_text(raw)
    return {
        "message"      : raw if raw else "Could not parse response.",
        "params"       : extracted,
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
                  output_dir : str = "llm_mission_output") -> dict:
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
                    output_dir : str  = "llm_mission_output",
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
            import POST_TRAIN
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
            output_dir = "llm_mission_output",
            auto_run   = False,
        )

        print(f"\nAssistant: {result['reply']}")
        if result["params"]:
            print(f"Extracted : {json.dumps(result['params'], indent=2)}")
        if result["ready_to_run"]:
            print("[Ready -- type 'run' to execute pipeline]")
        print()