import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model, Model
import tensorflow as tf
import io
import random, os
import pandas as pd

# Konfigurasi Halaman
st.set_page_config(
    page_title="Deteksi Kekurangan Nutrisi Daun Lettuce Iceberg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style
st.markdown(
    """
    <style>
    /* Background dan teks umum */
    .stApp {
        background-color: white !important;
        color: black !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: white !important;
        color: black !important;
    }

    /* Header */
    header[data-testid="stHeader"] {
        background-color: white !important;
    }

    /* Drag & Drop Upload */
    div[data-testid="stFileUploader"] section {
        background-color: white !important;
        border: 2px dashed #ccc !important;
        border-radius: 10px !important;
    }
    div[data-testid="stFileUploader"] p, div[data-testid="stFileUploader"] span {
        color: black !important;
    }
    /* Tombol Browse files */
    div[data-testid="stFileUploader"] button {
        background-color: white !important;
        color: black !important;
        border: 1px solid #000 !important;
    }

    /* Kamera & tombol Take Photo */
    div[data-testid="stCameraInput"] label {
        color: black !important;
        font-weight: 600 !important;
    }
    div[data-testid="stCameraInput"] button {
        background-color: white !important;
        color: black !important;
        border: 1px solid #000 !important;
    }

    /* Dropdown titik tiga */
    div[data-testid="stActionMenu"] {
        background-color: white !important;
        color: black !important;
    }
    div[data-testid="stActionMenu"] div {
        background-color: white !important;
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set Seed
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seeds()

# Load Model
@st.cache_resource
def load_models():
    autoencoder = load_model("autoencoder_model.h5")
    mlp = load_model("mlp_model.h5")
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("encoded_layer").output)
    return encoder, mlp

encoder, mlp = load_models()

# Label Kelas
labels = [
    'Kekurangan Nitrogen', 
    'Kekurangan Fosfor',    
    'Kekurangan Kalium',    
    'Sehat'                  
]

# Fungsi Prediksi
def proses_prediksi(image):
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    encoded = encoder.predict(img_array)
    flattened = encoded.reshape(1, -1)

    pred = mlp.predict(flattened)[0]
    class_idx = np.argmax(pred)
    confidence = np.max(pred)

    st.image(image, caption='🖼️ Gambar yang Diproses', use_column_width=True)
    st.subheader("📊 Hasil Prediksi Teratas:")
    st.success(f"✅ Kelas: **{labels[class_idx]}**")
    st.info(f"🔍 Confidence: **{confidence*100:.2f}%**")

    st.subheader("📈 Persentase Tiap Kelas:")
    for i, label in enumerate(labels):
        progress_val = max(0.0, min(1.0, float(pred[i])))
        st.progress(progress_val)
        st.write(f"{label}: **{progress_val*100:.2f}%**")

# Navigasi
menu = st.sidebar.radio("Navigasi", ["🏠 Home", "🔍 Prediksi", "👤 Profil"])

# Home
if menu == "🏠 Home":
    st.title("🌱 Sistem Deteksi Kekurangan Nutrisi Daun Lettuce Iceberg")
    st.write("""
    Aplikasi ini menggunakan **Deep Learning (Autoencoder + MLP)** untuk mendeteksi kekurangan nutrisi pada daun lettuce iceberg.
    
    **Kategori Deteksi**:
    - Kekurangan Nitrogen
    - Kekurangan Fosfor
    - Kekurangan Kalium
    - Kondisi Sehat
    
    **Fitur**:
    - Upload gambar daun
    - Deteksi langsung dari kamera
    - Menampilkan persentase tiap kemungkinan
    """)

# Prediksi
elif menu == "🔍 Prediksi":
    st.title("🔍 Prediksi Kekurangan Nutrisi")
    uploaded_file = st.file_uploader("📤 Upload Gambar Daun", type=["jpg", "jpeg", "png"])
    camera_image = st.camera_input("📷 Ambil Foto dari Kamera")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        proses_prediksi(image)

    elif camera_image is not None:
        image = Image.open(camera_image).convert('RGB')
        proses_prediksi(image)

# Profile
elif menu == "👤 Profil":
    st.title("Profil Pengembang")
    
    # Foto profil
    st.image("profile.jpg", caption="Sadra Din Azizi Muhammad", width=200 )
    
    st.write("""
    **Nama:** Sadra Din Azizi Muhammad  
    **Jurusan:** S1 Teknik Informatika - Universitas Islam Sultan Agung  
    **Proyek:** Deteksi Kekurangan Nutrisi pada Daun Lettuce Iceberg  
    **Deskripsi:** Skripsi ini mengembangkan sebuah sistem deteksi kekurangan nutrisi pada daun lettuce iceberg menggunakan metode Deep Learning berbasis arsitektur Autoencoder untuk ekstraksi fitur dan Multi-Layer Perceptron (MLP) untuk klasifikasi. Sistem ini dirancang untuk mengidentifikasi empat kategori kondisi daun, yaitu kekurangan nitrogen, kekurangan fosfor, kekurangan kalium, dan kondisi sehat, dengan memanfaatkan dataset citra daun yang telah diproses dan dinormalisasi. Hasil prediksi ditampilkan melalui antarmuka web interaktif berbasis Streamlit, dilengkapi dengan visualisasi persentase keyakinan model untuk setiap kategori. Penelitian ini diharapkan dapat membantu pemilik atau peneliti tanaman lettuce iceberg dalam melakukan diagnosis cepat dan akurat terhadap kondisi nutrisi tanaman, sehingga dapat meningkatkan efisiensi pemeliharaan dan produktivitas pertanian.
    """)
    
    st.markdown("📧 Email: [sadraazizi1305@gmail.com](mailto:sadraazizi1305@gmail.com)")
    st.markdown("💼 GitHub: [Klik di sini](https://github.com/SadraDinAziziMuhammad)")
    st.markdown("🌐 LinkedIn: [Klik di sini](https://www.linkedin.com/in/sadradinazizimuhammad/)")