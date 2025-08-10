import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model, Model
import tensorflow as tf
import io
import random, os

# --- Set seed biar prediksi stabil ---
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seeds()

# --- Load Model Autoencoder dan MLP ---
@st.cache_resource
def load_models():
    autoencoder = load_model("autoencoder_model.h5")
    mlp = load_model("mlp_model.h5")

    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("encoded_layer").output)
    return encoder, mlp

encoder, mlp = load_models()

# --- Label Kelas ---
labels = [
    'Kekurangan Nitrogen', 
    'Kekurangan Fosfor',    
    'Kekurangan Kalium',    
    'Sehat'                  
]

# --- Title App ---
st.title("🌱 Prediksi Kekurangan Nutrisi pada Tanaman Lettuce Iceberg")

# --- Pilih Sumber Gambar ---
st.subheader("Pilih Metode Input Gambar")
uploaded_file = st.file_uploader("📤 Upload Gambar Daun", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("📷 Ambil Foto dari Kamera")

# Fungsi prediksi
# Fungsi prediksi
def proses_prediksi(image):
    st.image(image, caption='🖼️ Gambar yang Diproses', use_column_width=True)

    # --- Preprocessing ---
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --- Ekstraksi Fitur via Autoencoder ---
    encoded = encoder.predict(img_array)
    flattened = encoded.reshape(1, -1)

    # --- Prediksi ---
    pred = mlp.predict(flattened)[0]  # ambil hasil prediksi array 1D
    class_idx = np.argmax(pred)
    confidence = np.max(pred)

    # --- Tampilkan hasil ---
    st.subheader("📊 Hasil Prediksi Teratas:")
    st.success(f"✅ Kelas: **{labels[class_idx]}**")
    st.info(f"🔍 Confidence: **{confidence*100:.2f}%**")

    if confidence < 0.7:
        st.warning("⚠️ Model kurang yakin terhadap prediksi ini. Coba upload gambar lain dengan kualitas lebih baik.")

    # --- Tampilkan semua persentase ---
    st.subheader("📈 Persentase Tiap Kelas:")
    for i, label in enumerate(labels):
        st.write(f"- {label}: **{pred[i]*100:.2f}%**")

    # --- Download hasil prediksi sebagai file .txt ---
    result_text = "Hasil Prediksi:\n"
    for i, label in enumerate(labels):
        result_text += f"{label}: {pred[i]*100:.2f}%\n"

    st.download_button(
        label="📥 Download Hasil Prediksi (.txt)",
        data=result_text,
        file_name="hasil_prediksi.txt",
        mime="text/plain"
    )

    # --- Download gambar input sebagai file PNG ---
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    st.download_button(
        label="📥 Download Gambar Input",
        data=buf.getvalue(),
        file_name="gambar_input.png",
        mime="image/png"
    )

# --- Cek sumber gambar yang dipilih ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    proses_prediksi(image)
elif camera_image is not None:
    image = Image.open(camera_image).convert('RGB')
    proses_prediksi(image)
else:
    st.info("📌 Silakan upload gambar atau ambil foto dari kamera untuk memulai prediksi.")