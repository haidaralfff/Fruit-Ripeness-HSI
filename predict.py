import argparse
# pyrefly: ignore [missing-import]
import cv2
import pickle
import time
import os
import sys

# Tambahkan src ke path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# pyrefly: ignore [missing-import]
from image_utils import load_and_preprocess_image

# pyrefly: ignore [missing-import]
from feature_extraction import extract_features

def main():
    parser = argparse.ArgumentParser(description="Deteksi Kematangan Buah dengan HSI dan Random Forest")
    parser.add_argument("--image", type=str, required=True, help="Path ke gambar yang akan dideteksi")
    args = parser.parse_args()

    # Periksa apakah model sudah dilatih
    model_path = os.path.join('models', 'random_forest.pkl')
    if not os.path.exists(model_path):
        print("Error: Model belum ditemukan. Harap jalankan notebook 03_Model_Training_Evaluation.ipynb terlebih dahulu untuk melatih dan menyimpan model.")
        return

    # Memuat model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Periksa gambar
    if not os.path.exists(args.image):
        print(f"Error: Gambar {args.image} tidak ditemukan.")
        return

    print("Memproses gambar...")
    start_time = time.time()
    
    # Preprocessing
    img_preprocessed = load_and_preprocess_image(args.image)
    if img_preprocessed is None:
        print("Error: Gagal memuat gambar.")
        return
        
    # Ekstraksi Fitur
    features = extract_features(img_preprocessed)
    
    # Prediksi
    # features adalah list 1D berisi 6 elemen, kita harus menjadikannya array 2D
    prediction = model.predict([features])
    
    end_time = time.time()
    latency = (end_time - start_time) * 1000

    print("====================================")
    print(f"Hasil Prediksi : {prediction[0].upper()}")
    print(f"Latency        : {latency:.2f} ms")
    print("====================================")

if __name__ == "__main__":
    main()
