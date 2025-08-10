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
labels = ['Sehat', 'Kekurangan Nitrogen', 'Kekurangan Fosfor', 'Kekurangan Kalium']

# --- Title App ---
st.title("Prediksi Kekurangan Nutrisi pada Tanaman Lettuce Iceberg")

# --- Upload Image atau Ambil dari Kamera ---
uploaded_file = st.file_uploader("📤 Upload Gambar Daun", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("📷 Ambil Gambar dari Kamera")

# Gunakan gambar dari kamera jika ada, kalau tidak pakai upload
image_source = None
if camera_file is not None:
    image_source = camera_file
elif uploaded_file is not None:
    image_source = uploaded_file

if image_source is not None:
    image = Image.open(image_source).convert('RGB')
    st.image(image, caption='🖼️ Gambar yang Digunakan', use_column_width=True)

    # --- Preprocessing ---
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --- Ekstraksi Fitur via Autoencoder ---
    encoded = encoder.predict(img_array)
    flattened = encoded.reshape(1, -1)

    # --- Prediksi ---
    pred = mlp.predict(flattened)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)

    # --- Tampilkan hasil ---
    st.subheader("📊 Hasil Prediksi:")
    st.success(f"✅ Kelas: **{labels[class_idx]}**")
    st.info(f"🔍 Confidence: **{confidence*100:.2f}%**")

    if confidence < 0.7:
        st.warning("⚠️ Model kurang yakin terhadap prediksi ini. Coba ambil gambar dengan pencahayaan lebih baik.")

    # --- Download hasil prediksi sebagai file .txt ---
    result_text = f"Hasil Prediksi:\nKelas: {labels[class_idx]}\nConfidence: {confidence*100:.2f}%"
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
