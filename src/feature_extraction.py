# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np

def rgb_to_hsi(image):
    """
    Konversi gambar dari format RGB ke HSI (Hue, Saturation, Intensity).
    """
    img = image.astype(np.float32) / 255.0
    B, G, R = img[:,:,0], img[:,:,1], img[:,:,2]
    
    numerator = 0.5 * ((R - G) + (R - B))
    denominator = np.sqrt((R - G)**2 + (R - B)*(G - B))
    theta = np.arccos(numerator / (denominator + 1e-8))
    
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H / (2 * np.pi)
    
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 * min_rgb / (R + G + B + 1e-8))
    
    I = (R + G + B) / 3.0
    return H, S, I

def extract_features(img):
    """
    Ekstrak fitur (Mean dan Standar Deviasi) dari channel H, S, dan I.
    Fungsi ini secara cerdas mengabaikan background hitam murni (0,0,0) 
    agar perhitungan murni hanya berdasarkan warna objek (buah).
    """
    H, S, I = rgb_to_hsi(img)
    
    # Filter background hitam murni (nilai Intensity mendekati 0)
    valid_pixels_mask = I > 0.01
    
    if not np.any(valid_pixels_mask):
        valid_pixels_mask = np.ones_like(I, dtype=bool)

    H_valid = H[valid_pixels_mask]
    S_valid = S[valid_pixels_mask]
    I_valid = I[valid_pixels_mask]
    
    return [
        np.mean(H_valid), np.std(H_valid),
        np.mean(S_valid), np.std(S_valid),
        np.mean(I_valid), np.std(I_valid)
    ]
