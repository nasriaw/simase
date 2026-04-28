from pathlib import Path
from typing import Dict

import numpy as np
import streamlit as st
from joblib import load

from feature_extractor import extract_from_bytes

APP_TITLE = "Sini MASE - Deteksi Dini Bakso Sehat Dari Texture"
MODEL_PATH = Path("models/model.joblib")

LABEL_MAP = {
    "healthy": "Makanan Sehat",
    "unhealthy": "Makanan Berbahaya",
}


def build_advice(label: str, confidence: float | None) -> dict:
    if label.lower().endswith("berbahaya"):
        return {
            "title": "Saran Edukasi (Keamanan Pangan)",
            "tips": [
                "Hindari konsumsi bila terdapat bau asam/menyengat, warna/tekstur tidak wajar, atau berlendir.",
                "Utamakan membeli dari produsen/penjual tepercaya yang memiliki izin edar.",
                "Simpan bukti pembelian/kemasan. Bila timbul gejala mual, muntah, diare, segera cari pertolongan medis.",
                "Laporkan dugaan pangan berbahaya ke otoritas setempat agar dapat ditindaklanjuti.",
            ],
            "links": [
                {"name": "Cek Produk di BPOM", "url": "https://cekbpom.pom.go.id"},
                {"name": "Situs Resmi BPOM", "url": "https://www.pom.go.id"},
                {"name": "Kementerian Kesehatan RI", "url": "https://www.kemkes.go.id"},
            ],
        }
    return {
        "title": "Tips Konsumsi Aman",
        "tips": [
            "Simpan makanan pada suhu dingin (≤4°C) bila tidak langsung dikonsumsi.",
            "Panaskan hingga matang merata sebelum disajikan.",
            "Terapkan higiene: cuci tangan, pisahkan alat masak mentah dan matang.",
            "Penuhi gizi seimbang: imbangi dengan sayuran, buah, dan sumber karbohidrat sehat.",
        ],
        "links": [
            {"name": "Kemenkes RI", "url": "https://www.kemkes.go.id"},
            {"name": "BPOM", "url": "https://www.pom.go.id"},
        ],
    }


@st.cache_resource
def load_model() -> Dict:
    if MODEL_PATH.exists():
        return load(MODEL_PATH)
    return {}


def predict_image(file_bytes: bytes, model_artifact: Dict):
    if not model_artifact:
        raise RuntimeError("Model belum dilatih. Jalankan: python train.py")

    feats = extract_from_bytes(file_bytes).reshape(1, -1)
    pipeline = model_artifact["pipeline"]
    labels = model_artifact["labels"]

    proba_supported = hasattr(pipeline[-1], "predict_proba")

    pred_idx = int(pipeline.predict(feats)[0])
    pred_label = labels[pred_idx]
    confidence = None
    probs_dict = None

    if proba_supported:
        probs = pipeline.predict_proba(feats)[0]
        confidence = float(np.max(probs))
        probs_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}
    elif hasattr(pipeline[-1], "decision_function"):
        margin = pipeline.decision_function(feats)
        confidence = float(1 / (1 + np.exp(-np.max(margin))))

    pretty_label = LABEL_MAP.get(pred_label.lower(), pred_label)
    pretty_probs = None
    if probs_dict is not None:
        pretty_probs = {LABEL_MAP.get(k.lower(), k): v for k, v in probs_dict.items()}

    return pretty_label, confidence, pretty_probs


def main():
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    st.title(APP_TITLE)
    st.write(
        "Deteksi apakah tekstur bakso termasuk `Makanan Sehat` atau `Makanan Berbahaya` menggunakan model HOG + Logistic Regression."
    )
    st.write("Pilih salah satu metode berikut untuk mengunggah atau mengambil foto bakso.")

    model_artifact = load_model()
    if not model_artifact:
        st.warning(
            "Model tidak ditemukan di `models/model.joblib`. Pastikan model sudah dilatih dan file tersedia."
        )

    input_method = st.radio(
        "Metode input:",
        ("Unggah Gambar", "Ambil Foto Bakso"),
        index=0,
    )

    uploaded_file = None
    if input_method == "Unggah Gambar":
        uploaded_file = st.file_uploader(
            "Unggah gambar bakso",
            type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
        )
    else:
        uploaded_file = st.camera_input("Ambil foto bakso menggunakan kamera perangkat")

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        st.image(file_bytes, caption="Preview Bakso", use_column_width=True)

        try:
            label, confidence, probs = predict_image(file_bytes, model_artifact)

            st.markdown("---")
            st.markdown(f"### Prediksi: **{label}**")
            confidence_text = f"{confidence:.2f}" if confidence is not None else "N/A"
            st.write(f"**Confidence:** {confidence_text}")

            if probs:
                st.markdown("#### Probabilitas")
                probs_table = {
                    "Label": list(probs.keys()),
                    "Probability": [f"{v:.3f}" for v in probs.values()],
                }
                st.table(probs_table)

            advice = build_advice(label, confidence)
            st.markdown(f"#### {advice['title']}")
            for tip in advice["tips"]:
                st.write(f"- {tip}")

            if advice.get("links"):
                st.markdown("##### Sumber dan tautan berguna")
                for link in advice["links"]:
                    st.write(f"- [{link['name']}]({link['url']})")
        except Exception as exc:
            st.error(f"Terjadi kesalahan saat memproses gambar: {exc}")

    st.markdown("---")
    st.info(
        "Jika model belum dilatih, jalankan `python train.py` di folder ini untuk membuat file `models/model.joblib`."
    )


if __name__ == "__main__":
    main()
