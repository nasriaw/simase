from pathlib import Path
from typing import Dict
import os
from uuid import uuid4

import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from joblib import load
from werkzeug.utils import secure_filename

from feature_extractor import extract_from_bytes

APP_TITLE = "Sini MASE, Deteksi Dini Makanan Sehat"
MODEL_PATH = Path("models/model.joblib")
DATASET_DIR = Path("data/train")
UPLOAD_DIR = Path("static/uploads")
LOGO_PATH = Path("static/logo.png")

app = Flask(__name__)
app.secret_key = "change-this-secret-key"  # needed for flash messages

_model_artifact: Dict = {}


def load_or_warn():
    global _model_artifact
    if MODEL_PATH.exists():
        _model_artifact = load(MODEL_PATH)
    else:
        _model_artifact = {}


def predict_image(file_bytes: bytes):
    if not _model_artifact:
        raise RuntimeError("Model belum dilatih. Jalankan: python train.py")

    feats = extract_from_bytes(file_bytes).reshape(1, -1)
    pipeline = _model_artifact["pipeline"]
    labels = _model_artifact["labels"]

    proba_supported = hasattr(pipeline[-1], "predict_proba")

    pred_idx = int(pipeline.predict(feats)[0])
    pred_label = labels[pred_idx]

    confidence = None
    probs_dict = None
    if proba_supported:
        probs = pipeline.predict_proba(feats)[0]
        confidence = float(np.max(probs))
        probs_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
    else:
        # use decision function margin if proba not available
        if hasattr(pipeline[-1], "decision_function"):
            margin = pipeline.decision_function(feats)
            confidence = float(1 / (1 + np.exp(-np.max(margin))))

    # Map label to bahasa
    label_map = {
        "healthy": "Makanan Sehat",
        "unhealthy": "Makanan Berbahaya",
    }
    pretty = label_map.get(pred_label.lower(), pred_label)

    # Map probs to pretty labels if available
    pretty_probs = None
    if probs_dict is not None:
        pretty_probs = {label_map.get(k.lower(), k): v for k, v in probs_dict.items()}

    return pretty, confidence, pretty_probs


def build_advice(label: str, confidence: float | None):
    """Return dict with title, tips, and optional links based on predicted label."""
    if label.lower().endswith("berbahaya"):
        tips = [
            "Hindari konsumsi bila terdapat bau asam/menyengat, warna/tekstur tidak wajar, atau berlendir.",
            "Utamakan membeli dari produsen/penjual tepercaya yang memiliki izin edar.",
            "Simpan bukti pembelian/kemasan. Bila timbul gejala mual, muntah, diare, segera cari pertolongan medis.",
            "Laporkan dugaan pangan berbahaya ke otoritas setempat agar dapat ditindaklanjuti.",
        ]
        links = [
            {"name": "Cek Produk di BPOM", "url": "https://cekbpom.pom.go.id"},
            {"name": "Situs Resmi BPOM", "url": "https://www.pom.go.id"},
            {"name": "Kementerian Kesehatan RI", "url": "https://www.kemkes.go.id"},
        ]
        return {"title": "Saran Edukasi (Keamanan Pangan)", "tips": tips, "links": links}
    else:
        tips = [
            "Simpan makanan pada suhu dingin (≤4°C) bila tidak langsung dikonsumsi.",
            "Panaskan hingga matang merata sebelum disajikan.",
            "Terapkan higiene: cuci tangan, pisahkan alat masak mentah dan matang.",
            "Penuhi gizi seimbang: imbangi dengan sayuran, buah, dan sumber karbohidrat sehat.",
        ]
        links = [
            {"name": "Kemenkes RI", "url": "https://www.kemkes.go.id"},
            {"name": "BPOM", "url": "https://www.pom.go.id"},
        ]
        return {"title": "Tips Konsumsi Aman", "tips": tips, "links": links}


@app.route("/", methods=["GET"]) 
def index():
    load_or_warn()
    model_ready = bool(_model_artifact)
    logo_exists = LOGO_PATH.exists()
    return render_template(
        "index.html",
        app_title=APP_TITLE,
        model_ready=model_ready,
        logo_exists=logo_exists,
    )


@app.route("/predict", methods=["POST"]) 
def predict():
    # Ambil file dari salah satu input yang tersedia (image, image_gallery, image_camera)
    file = None
    for key in ["image", "image_gallery", "image_camera"]:
        if key in request.files and request.files[key].filename != "":
            file = request.files[key]
            break
    if file is None:
        flash("Mohon pilih atau ambil gambar terlebih dahulu.")
        return redirect(url_for("index"))

    file_bytes = file.read()

    # Simpan gambar ke static/uploads untuk ditampilkan di halaman hasil
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    original_ext = Path(secure_filename(file.filename)).suffix.lower()
    if original_ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        original_ext = ".jpg"
    save_name = f"{uuid4().hex}{original_ext}"
    save_path = UPLOAD_DIR / save_name
    with open(save_path, "wb") as out_f:
        out_f.write(file_bytes)
    image_url = url_for("static", filename=f"uploads/{save_name}")

    try:
        label, conf, probs = predict_image(file_bytes)
        # Siapkan daftar probabilitas terurut menurun untuk ditampilkan
        probs_list = None
        if isinstance(probs, dict):
            probs_list = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        advice = build_advice(label, conf)
        return render_template(
            "result.html",
            app_title=APP_TITLE,
            label=label,
            confidence=conf,
            image_url=image_url,
            probs=probs_list,
            advice=advice,
            logo_exists=LOGO_PATH.exists(),
        )
    except Exception as e:
        flash(f"Terjadi kesalahan: {e}")
        return redirect(url_for("index"))


if __name__ == "__main__":
    # Jalankan di 0.0.0.0 agar bisa diakses dari HP pada jaringan yang sama
    app.run(host="0.0.0.0", port=8000, debug=True)
