# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np

def load_and_preprocess_image(filepath, target_size=(128, 128)):
    """
    Load an image, remove its background, and resize it.
    Uses Saturation-based Otsu's thresholding and contour filling 
    to robustly segment fruits without creating holes from specular highlights.
    """
    img = cv2.imread(filepath)
    if img is None:
        return None
        
    img_resized = cv2.resize(img, target_size)
    img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
    
    # --- Robust Background Removal ---
    hsv = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    
    _, mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        
        # Crop gambar sesuai Bounding Box agar pas dengan buah
        x, y, w, h = cv2.boundingRect(c)
        img_cropped = img_blurred[y:y+h, x:x+w]
        
        return cv2.resize(img_cropped, target_size)
    
    return img_blurred

def get_fruit_bounding_box(filepath):
    """
    Mengambil koordinat Bounding Box dari buah asli tanpa mengubah ukuran gambar.
    Berguna untuk menggambar kotak deteksi di script prediksi.
    """
    img = cv2.imread(filepath)
    if img is None:
        return None
    
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    
    _, mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(c)
    return None
