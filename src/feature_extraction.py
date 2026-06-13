import cv2
import numpy as np

def rgb_to_hsi(image):
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
    H, S, I = rgb_to_hsi(img)
    return [
        np.mean(H), np.std(H),
        np.mean(S), np.std(S),
        np.mean(I), np.std(I)
    ]
