"""
Base vs LoRA Model Comparison – Web Server Version
====================================================
Runs the same comparison as test_base_vs_lora.py but presents
results in a real-time web UI on http://localhost:5000

Run:
  python -X utf8 test_base_vs_lora_web.py
"""

import sys
import json
import gc
import logging
import tempfile
import time
import threading
import queue
from pathlib import Path

from flask import Flask, Response, render_template_string

# ── Configuration ───────────────────────────────────────────────
MODEL_NAME = "OpenGVLab/InternVL2-2B"
LORA_PATH  = "checkpoints/satquery-lora-exp3/best"

QUESTION = (
    "Describe this satellite image in detail. Write a structured analysis in complete sentences. "
    "Discuss visible land cover, water features, vegetation, terrain, and human-made structures. "
    "Do not guess the location."
)

SYSTEM_INSTRUCTION = (
    "You are a satellite image analysis assistant. "
    "Analyse ONLY what is visually observable in the provided image. "
    "Give a complete, detailed answer in full sentences.\n"
)

PROMPT = (
    f"{SYSTEM_INSTRUCTION}\n"
    f"<image>\n"
    f"Question: {QUESTION}\n"
    f"Answer:"
)

GEN_CONFIGS = {
    "short (current: temp=0.1, max=512, no sample)": {
        "max_new_tokens": 512,
        "do_sample": False,
        "temperature": 0.1,
    },
    "long (experimental: temp=0.7, max=512, sample)": {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    },
}

# ── Globals ─────────────────────────────────────────────────────
app = Flask(__name__)
log_queue = queue.Queue()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def emit(event_type, data):
    """Push a server-sent event to the browser."""
    log_queue.put({"type": event_type, "data": data})


# ── HTML Template ───────────────────────────────────────────────
HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Base vs LoRA Comparison</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a26;
    --border: #2a2a3a;
    --text: #e4e4ef;
    --text-dim: #8888a0;
    --accent: #7c5cfc;
    --accent-glow: rgba(124, 92, 252, 0.15);
    --green: #34d399;
    --amber: #fbbf24;
    --red: #f87171;
    --blue: #60a5fa;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', system-ui, sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }

  .bg-glow {
    position: fixed; top: -200px; left: 50%; transform: translateX(-50%);
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(124,92,252,0.08) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
  }

  .container {
    max-width: 960px; margin: 0 auto; padding: 40px 24px;
    position: relative; z-index: 1;
  }

  header {
    text-align: center; margin-bottom: 40px;
  }
  header h1 {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, var(--accent), #a78bfa, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
  }
  header p { color: var(--text-dim); font-size: 0.95rem; }

  /* Status bar */
  .status-bar {
    display: flex; align-items: center; gap: 10px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 14px 20px; margin-bottom: 24px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
  }
  .status-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--amber);
    animation: pulse 1.5s infinite;
  }
  .status-dot.done { background: var(--green); animation: none; }
  .status-dot.error { background: var(--red); animation: none; }
  @keyframes pulse {
    0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
  }

  /* Log stream */
  .log-panel {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px; margin-bottom: 24px;
    max-height: 280px; overflow-y: auto;
    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
    line-height: 1.7; color: var(--text-dim);
  }
  .log-panel .log-info { color: var(--blue); }
  .log-panel .log-warn { color: var(--amber); }
  .log-panel .log-error { color: var(--red); }
  .log-panel .log-success { color: var(--green); }

  /* Result cards */
  .results-grid {
    display: grid; grid-template-columns: 1fr; gap: 16px;
  }
  .result-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden;
    animation: fadeIn 0.5s ease-out;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .result-card .card-header {
    padding: 16px 20px;
    background: var(--surface2); border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
  }
  .result-card .card-header h3 {
    font-size: 0.9rem; font-weight: 600;
  }
  .result-card .card-header .badge {
    font-size: 0.75rem; padding: 3px 10px; border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
  }
  .badge-base { background: rgba(96,165,250,0.15); color: var(--blue); }
  .badge-lora { background: rgba(124,92,252,0.15); color: var(--accent); }
  .result-card .card-body {
    padding: 20px; font-size: 0.9rem; line-height: 1.7;
  }
  .result-card .card-footer {
    padding: 12px 20px; background: var(--surface2);
    border-top: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
    color: var(--text-dim); display: flex; gap: 20px;
  }

  /* Summary */
  .summary-card {
    background: var(--surface); border: 1px solid var(--accent);
    border-radius: 12px; padding: 24px; margin-top: 24px;
    box-shadow: 0 0 30px var(--accent-glow);
  }
  .summary-card h2 {
    font-size: 1.1rem; margin-bottom: 16px; color: var(--accent);
  }
  .summary-card .interp { color: var(--text-dim); font-size: 0.88rem; line-height: 1.7; }
  .summary-card .interp strong { color: var(--text); }

  .hidden { display: none; }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>
<div class="bg-glow"></div>
<div class="container">
  <header>
    <h1>🛰️ Base vs LoRA Comparison</h1>
    <p>InternVL2-2B · Step 7 Diagnostic Test</p>
  </header>

  <div class="status-bar">
    <div class="status-dot" id="statusDot"></div>
    <span id="statusText">Initializing…</span>
  </div>

  <div class="log-panel" id="logPanel"></div>

  <div class="results-grid" id="resultsGrid"></div>
  <div class="summary-card hidden" id="summaryCard">
    <h2>📊 Interpretation</h2>
    <div class="interp">
      <p><strong>Base detailed, LoRA short →</strong> LoRA training caused the short-answer behavior</p>
      <p><strong>Both short →</strong> Generation config / prompt problem</p>
      <p><strong>Both detailed →</strong> Routing or display bug (model is fine)</p>
    </div>
  </div>
</div>

<script>
const logPanel = document.getElementById('logPanel');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const resultsGrid = document.getElementById('resultsGrid');
const summaryCard = document.getElementById('summaryCard');

function addLog(msg, cls = '') {
  const line = document.createElement('div');
  if (cls) line.className = cls;
  line.textContent = msg;
  logPanel.appendChild(line);
  logPanel.scrollTop = logPanel.scrollHeight;
}

function addResult(data) {
  const isLora = data.label.toLowerCase().includes('lora');
  const card = document.createElement('div');
  card.className = 'result-card';
  card.innerHTML = `
    <div class="card-header">
      <h3>${data.label}</h3>
      <span class="badge ${isLora ? 'badge-lora' : 'badge-base'}">${isLora ? 'LoRA' : 'Base'}</span>
    </div>
    <div class="card-body">${data.answer}</div>
    <div class="card-footer">
      <span>Tokens: ${data.tokens}</span>
      <span>Words: ${data.words}</span>
    </div>
  `;
  resultsGrid.appendChild(card);
}

const evtSource = new EventSource('/stream');

evtSource.addEventListener('log', e => {
  const d = JSON.parse(e.data);
  addLog(d.message, d.cls || '');
});

evtSource.addEventListener('status', e => {
  const d = JSON.parse(e.data);
  statusText.textContent = d.message;
  if (d.state === 'done') { statusDot.className = 'status-dot done'; }
  else if (d.state === 'error') { statusDot.className = 'status-dot error'; }
});

evtSource.addEventListener('result', e => {
  const d = JSON.parse(e.data);
  addResult(d);
});

evtSource.addEventListener('done', e => {
  summaryCard.classList.remove('hidden');
  evtSource.close();
});

evtSource.onerror = () => {
  statusDot.className = 'status-dot error';
  statusText.textContent = 'Connection lost';
};
</script>
</body>
</html>
"""


# ── ML Logic ────────────────────────────────────────────────────

def preprocess_image(pil_img):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(pil_img.convert("RGB")).unsqueeze(0)


def find_test_image():
    temp_uploads = Path(tempfile.gettempdir()) / "satquery_uploads"
    if temp_uploads.exists():
        found = sorted(temp_uploads.glob("*.jpg")) + sorted(temp_uploads.glob("*.png"))
        if found:
            return str(found[-1])
    return None


def load_base_model():
    import torch
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    emit("log", {"message": "Loading BASE InternVL2-2B (no LoRA)…", "cls": "log-info"})
    emit("status", {"message": "Loading base model…", "state": "running"})

    load_kwargs = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    if torch.cuda.is_available():
        emit("log", {"message": f"CUDA detected: {torch.cuda.get_device_name(0)}", "cls": "log-success"})
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        emit("log", {"message": "⚠ No CUDA GPU – running on CPU (will be slow)", "cls": "log-warn"})
        load_kwargs["torch_dtype"] = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(MODEL_NAME, **load_kwargs).eval()

    emit("log", {"message": "Base model loaded ✓", "cls": "log-success"})
    return model, tokenizer


def apply_lora(model, lora_path):
    from peft import PeftModel
    emit("log", {"message": f"Applying LoRA adapter: {lora_path}", "cls": "log-info"})
    model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
    emit("log", {"message": "LoRA applied ✓", "cls": "log-success"})
    return model


def run_inference(model, tokenizer, pil_img, gen_config, label):
    import torch
    emit("log", {"message": f"Running inference: {label}", "cls": "log-info"})
    emit("status", {"message": f"Inference: {label}", "state": "running"})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pixel_values = preprocess_image(pil_img).to(device, dtype=dtype)

    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, PROMPT, gen_config)

    tokens = len(tokenizer.encode(response))
    words = len(response.split())

    emit("result", {
        "label": label,
        "answer": response,
        "tokens": tokens,
        "words": words,
    })
    emit("log", {"message": f"  → {words} words, {tokens} tokens", "cls": "log-success"})
    return response


def run_comparison():
    """Run the full comparison in a background thread."""
    import torch
    from PIL import Image

    try:
        emit("status", {"message": "Starting comparison…", "state": "running"})
        emit("log", {"message": "=" * 50, "cls": ""})
        emit("log", {"message": "STEP 7: Base InternVL2-2B vs LoRA Comparison", "cls": "log-info"})
        emit("log", {"message": "=" * 50, "cls": ""})

        # Find test image
        img_path = find_test_image()
        if img_path:
            emit("log", {"message": f"Using image: {img_path}", "cls": "log-info"})
            pil_img = Image.open(img_path).convert("RGB")
        else:
            emit("log", {"message": "No uploaded image found – creating synthetic test image", "cls": "log-warn"})
            import numpy as np
            arr = (np.random.rand(343, 343, 3) * 255).astype("uint8")
            pil_img = Image.fromarray(arr)

        results = {}

        # ── Test A: Base model ──
        emit("log", {"message": "", "cls": ""})
        emit("log", {"message": ">>> TEST A: BASE InternVL2-2B (no LoRA)", "cls": "log-info"})
        base_model, tokenizer = load_base_model()
        for cfg_name, cfg in GEN_CONFIGS.items():
            key = f"BASE | {cfg_name}"
            results[key] = run_inference(base_model, tokenizer, pil_img, cfg, key)

        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        emit("log", {"message": "Base model freed from memory ✓", "cls": "log-success"})

        # ── Test B: Base + LoRA ──
        lora_path = Path(LORA_PATH)
        if lora_path.exists():
            emit("log", {"message": "", "cls": ""})
            emit("log", {"message": ">>> TEST B: InternVL2-2B + satquery-lora-exp3", "cls": "log-info"})
            lora_model, tokenizer = load_base_model()
            lora_model = apply_lora(lora_model, str(lora_path))
            for cfg_name, cfg in GEN_CONFIGS.items():
                key = f"LORA | {cfg_name}"
                results[key] = run_inference(lora_model, tokenizer, pil_img, cfg, key)
            del lora_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            emit("log", {"message": f"[SKIP] LoRA adapter not found at {lora_path}", "cls": "log-warn"})

        emit("status", {"message": "Comparison complete ✓", "state": "done"})
        emit("done", {})

    except Exception as e:
        emit("log", {"message": f"ERROR: {e}", "cls": "log-error"})
        emit("status", {"message": f"Failed: {e}", "state": "error"})


# ── Flask Routes ────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/stream")
def stream():
    def generate():
        while True:
            try:
                item = log_queue.get(timeout=30)
                evt_type = item["type"]
                data = json.dumps(item["data"])
                yield f"event: {evt_type}\ndata: {data}\n\n"
                if evt_type == "done":
                    break
            except queue.Empty:
                # Send keepalive
                yield ": keepalive\n\n"
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Base vs LoRA Comparison – Web UI")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 50 + "\n")

    # Start comparison in background once first client connects
    _state = {"started": False}
    original_stream = stream

    @app.route("/stream", endpoint="stream_wrapper")
    def stream_with_autostart():
        if not _state["started"]:
            _state["started"] = True
            t = threading.Thread(target=run_comparison, daemon=True)
            t.start()
        return original_stream()

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
