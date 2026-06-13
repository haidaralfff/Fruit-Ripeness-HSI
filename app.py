import streamlit as st
import cv2
import pickle
import time
import os
import sys
import numpy as np
from PIL import Image

# Tambahkan src ke path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from image_utils import load_and_preprocess_image
from feature_extraction import extract_features

st.set_page_config(page_title="Deteksi Kematangan Buah HSI", layout="centered")

st.title("🍎 Deteksi Kematangan Buah (HSI + Random Forest)")
st.write("Unggah gambar buah (Apel, Pisang, Jeruk, Mangga, atau Tomat) untuk memprediksi tingkat kematangannya.")

# Memuat model
@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'random_forest.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

if model is None:
    st.error("Model belum dilatih! Silakan jalankan `notebooks/03_Model_Training_Evaluation.ipynb` terlebih dahulu.")
else:
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Tampilkan gambar yang diunggah
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang diunggah', use_container_width=True)
        
        if st.button("Deteksi Kematangan"):
            # Konversi gambar ke format OpenCV (numpy array)
            img_array = np.array(image.convert('RGB'))
            
            # Ubah RGB ke BGR karena OpenCV menggunakan BGR
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            with st.spinner('Memproses dan mengekstrak fitur HSI...'):
                start_time = time.time()
                
                # Preprocessing secara manual sesuai dengan image_utils.py tapi menerima numpy array, bukan file path
                target_size = (128, 128)
                img_resized = cv2.resize(img_bgr, target_size)
                img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
                
                # Ekstraksi Fitur
                features = extract_features(img_blurred)
                
                # Prediksi
                prediction = model.predict([features])[0]
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000
                
            st.success("Proses selesai!")
            
            # Tampilan hasil
            st.markdown(f"### Hasil Prediksi: **{prediction.upper()}**")
            st.caption(f"⏱️ Waktu inferensi (Latency): {latency:.2f} ms")
            
            # Tampilkan fitur yang diekstrak (opsional, untuk edukasi)
            with st.expander("Lihat Detail Fitur HSI yang Diekstrak"):
                st.write({
                    "Mean Hue": round(features[0], 4),
                    "Std Hue": round(features[1], 4),
                    "Mean Saturation": round(features[2], 4),
                    "Std Saturation": round(features[3], 4),
                    "Mean Intensity": round(features[4], 4),
                    "Std Intensity": round(features[5], 4),
                })
