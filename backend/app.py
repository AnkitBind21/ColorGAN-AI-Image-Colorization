"""
Phase 6b: Flask Backend API
------------------------------
POST /colorize  — accepts image, returns colorized image
GET  /health    — health check
"""

import os
import sys
import uuid
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import download_model
# Add models to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))

from inference  import load_generator, colorize_image

app = Flask(__name__)
CORS(app)

# ── Config ───────────────────────────────────────────────────
UPLOAD_DIR  = os.path.join(os.path.dirname(__file__), "uploads")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CHECKPOINT  = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "generator_latest.pt")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE  = 256
MAX_MB      = 10

os.makedirs(UPLOAD_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load model at startup ────────────────────────────────────
generator = None

def get_generator():
    global generator
    if generator is None:
        if not os.path.exists(CHECKPOINT):
            raise FileNotFoundError(
                f"No checkpoint found at {CHECKPOINT}. "
                "Train the model first (python models/train.py --train) "
                "or place a checkpoint at the expected path."
            )
        generator = load_generator(CHECKPOINT, device=DEVICE)
    return generator


# ── Routes ───────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check — also reports model status."""
    ckpt_exists = os.path.exists(CHECKPOINT)
    return jsonify({
        "status" : "ok",
        "device" : DEVICE,
        "model_ready": ckpt_exists,
        "checkpoint" : CHECKPOINT if ckpt_exists else "NOT FOUND — train model first",
    })


@app.route("/colorize", methods=["POST"])
def colorize():
    """
    POST /colorize
    Form-data: file=<image>

    Returns:
        { "success": true, "image": "filename.jpg", "url": "/results/filename.jpg" }
    """
    # ── Validate request ─────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file field in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"success": False, "error": f"Unsupported format: {ext}"}), 400

    # ── Save upload ───────────────────────────────────────────
    job_id      = uuid.uuid4().hex[:8]
    upload_path = os.path.join(UPLOAD_DIR,  f"{job_id}_input{ext}")
    result_name = f"{job_id}_colorized.jpg"
    result_path = os.path.join(RESULTS_DIR, result_name)

    file.save(upload_path)

    # ── Colorize ─────────────────────────────────────────────
    try:
        G = get_generator()
        colorized = colorize_image(G, upload_path, image_size=IMAGE_SIZE, device=DEVICE)
        Image.fromarray(colorized).save(result_path, quality=95)
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        return jsonify({"success": False, "error": f"Colorization failed: {str(e)}"}), 500
    finally:
        # Clean up upload
        if os.path.exists(upload_path):
            os.remove(upload_path)

    return jsonify({
        "success": True,
        "image"  : result_name,
        "url"    : f"/results/{result_name}",
    })


@app.route("/results/<filename>")
def serve_result(filename):
    """Serve colorized image files."""
    return send_from_directory(RESULTS_DIR, filename)


# ── Run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    print(f"[Backend] Device      : {DEVICE}")
    print(f"[Backend] Checkpoint  : {CHECKPOINT}")
    print(f"[Backend] Starting Flask on port {port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )