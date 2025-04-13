import cv2
import numpy as np

def load_image(image_path):
    """Load image from path"""
    return cv2.imread(image_path)

def is_blank_image(image, threshold=0.95):
    """Check if image is mostly blank/white"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.countNonZero(gray) / (gray.size) < (1 - threshold)

def preprocess_image(image, target_size=(300, 100)):
    """Preprocess image for model input"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    
    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Resize while maintaining aspect ratio
    h, w = cleaned.shape
    new_w = int(target_size[1] * w / h)
    resized = cv2.resize(cleaned, (new_w, target_size[1]))
    
    # Pad to target width
    delta_w = target_size[0] - new_w
    if delta_w > 0:
        padded = cv2.copyMakeBorder(
            resized, 0, 0, 0, delta_w,
            cv2.BORDER_CONSTANT, value=0)
    else:
        padded = resized[:, :target_size[0]]
    
    # Normalize
    normalized = padded / 255.0
    return np.expand_dims(normalized, axis=-1)