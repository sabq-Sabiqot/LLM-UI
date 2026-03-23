REM 1. Install Ollama from https://ollama.ai
REM 2. Pull a model
ollama pull llama3.2

REM 3. Ollama starts automatically. Then run:
python llm_ui.py

REM 4. Open browser
http://localhost:5000
```

**Setup — LM Studio (GUI, easier for Windows)**
```
1. Download LM Studio from https://lmstudio.ai
2. Load any GGUF model (e.g. Llama-3.2-3B-Instruct)
3. Start Local Server (default port 1234)
4. In llm_mission.py set:  BACKEND = "lmstudio"
5. python llm_ui.py
```

---

**What the UI looks like**

- Dark chat interface in the browser
- Left sidebar shows extracted parameters updating live as you type
- "Run UAV Pipeline" button appears once payload + speed are provided
- Suggestion chips for common mission descriptions

---

**Example conversations**
```
You:  Design a surveillance UAV, 30 kg payload, cruise at 90 km/h at 500 m altitude

AI:   I've extracted these parameters:
      { "payload_kg": 30.0, "cruise_speed": 25.0, "altitude_m": 500.0 }
      Ready to run the pipeline. Type "run" to execute.

You:  Also optimise the airfoil with bayesian, 40 evaluations

AI:   Added optimisation parameters:
      { "optimize": true, "optimizer": "bayesian", "n_calls": 40 }

You:  run

AI:   Running pipeline... [results appear]