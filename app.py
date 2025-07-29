import streamlit as st
from PIL import Image
import numpy as np
import joblib
from model.resnet_feature_extractor import extract_features
from utils.preprocess import preprocess_image

# Load model SVM
svm_model = joblib.load("model/svm_model.pkl")

# Styling
st.set_page_config(page_title="Klasifikasi Sampah Otomatis", layout="centered")
st.markdown("""
    <style>
        .main {background-color: #b3e0ff;}
        .title {
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            color: #1a1a1a;
        }
        .desc {
            text-align: center;
            font-size: 16px;
            color: #333;
        }
        .button-upload {
            background-color: #2563eb;
            color: white;
            font-weight: bold;
            padding: 0.7em 1.2em;
            border: none;
            border-radius: 8px;
        }
        .button-classify {
            background-color: #34d399;
            color: black;
            font-weight: bold;
            padding: 0.7em 1.2em;
            border: none;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Halaman
st.markdown('<div class="title">Klasifikasi Sampah Otomatis</div>', unsafe_allow_html=True)
st.markdown("""
<div class="desc">
    Aplikasi ini menggunakan teknologi AI untuk mengenali jenis sampah secara otomatis, seperti organik, plastik, logam, dan lainnya.
</div><br>
""", unsafe_allow_html=True)

# Upload gambar
uploaded_file = st.file_uploader("Upload Gambar Sampah", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar Sampah yang Diupload', use_column_width=True)

    # Tombol klasifikasi
    if st.button("Klasifikasikan Sampah"):
        with st.spinner("Mengklasifikasikan..."):
            img_array = preprocess_image(image)
            features = extract_features(img_array)
            prediction = svm_model.predict([features])[0]

        st.success(f"Hasil Klasifikasi: **{prediction}**")
        st.balloons()
else:
    st.info("Silakan upload gambar untuk diklasifikasikan.")

st.markdown('<br><a href="/">← Kembali ke Beranda</a>', unsafe_allow_html=True)
