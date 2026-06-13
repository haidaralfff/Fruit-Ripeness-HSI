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
from image_utils import load_and_preprocess_image, get_fruit_bounding_box

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
    prediction = model.predict([features])
    
    end_time = time.time()
    latency = (end_time - start_time) * 1000

    pred_label = prediction[0].upper()
    print("====================================")
    print(f"Hasil Prediksi : {pred_label}")
    print(f"Latency        : {latency:.2f} ms")
    print("====================================")
    
    # --- VISUALISASI HASIL DENGAN KOTAK HIJAU ---
    
    img_ori = cv2.imread(args.image)
    bbox = get_fruit_bounding_box(args.image)
    
    if img_ori is not None and bbox is not None:
        x, y, w, h = bbox
        # Gambar kotak hijau (B=0, G=255, R=0) dengan ketebalan 3
        cv2.rectangle(img_ori, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        # Tulis label prediksi di atas kotak
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_ori, pred_label, (x, max(y-10, 20)), font, 0.9, (0, 255, 0), 2)
        
        # Tampilkan gambar (Tekan sembarang tombol untuk menutup)
        cv2.imshow("Hasil Deteksi Kematangan Buah", img_ori)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
