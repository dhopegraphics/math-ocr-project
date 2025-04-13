import numpy as np
import cv2
from skimage.transform import rotate
from skimage.util import random_noise

def random_rotation(image, max_angle=5):
    angle = np.random.uniform(-max_angle, max_angle)
    return rotate(image, angle, mode='edge')

def random_scale(image, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    h, w = image.shape[:2]
    new_size = (int(w * scale), int(h * scale))
    scaled = cv2.resize(image, new_size)
    
    # Pad/crop to original size
    if scale > 1:
        # Crop center
        start_x = (scaled.shape[1] - w) // 2
        start_y = (scaled.shape[0] - h) // 2
        return scaled[start_y:start_y+h, start_x:start_x+w]
    else:
        # Pad with zeros
        delta_w = w - scaled.shape[1]
        delta_h = h - scaled.shape[0]
        return cv2.copyMakeBorder(
            scaled, 
            delta_h//2, delta_h - delta_h//2,
            delta_w//2, delta_w - delta_w//2,
            cv2.BORDER_CONSTANT, value=0)

def augment_image(image):
    if np.random.rand() > 0.5:
        image = random_rotation(image)
    if np.random.rand() > 0.5:
        image = random_scale(image)
    if np.random.rand() > 0.2:
        image = random_noise(image, var=0.001)
    return image