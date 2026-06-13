import cv2
import numpy as np

def load_and_preprocess_image(filepath, target_size=(128, 128)):
    img = cv2.imread(filepath)
    if img is None:
        return None
    img_resized = cv2.resize(img, target_size)
    img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
    return img_blurred
