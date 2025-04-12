import numpy as np
import cv2
from scipy.ndimage import interpolation

def random_rotation(image, max_angle=10):
    """Rotate image by random angle between -max_angle and max_angle"""
    angle = np.random.uniform(-max_angle, max_angle)
    return interpolation.rotate(image, angle, reshape=False, mode='nearest')

def random_scale(image, scale_range=(0.9, 1.1)):
    """Randomly scale image"""
    h, w = image.shape[:2]
    scale = np.random.uniform(scale_range[0], scale_range[1])
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(image, (new_w, new_h))
    
    # Pad or crop to original size
    if scale > 1:
        # Crop
        start_h, start_w = (new_h - h) // 2, (new_w - w) // 2
        img = img[start_h:start_h+h, start_w:start_w+w]
    else:
        # Pad
        pad_h, pad_w = h - new_h, w - new_w
        img = np.pad(img, ((pad_h//2, pad_h - pad_h//2), 
                      (pad_w//2, pad_w - pad_w//2)), mode='constant')
    
    return img

def add_noise(image, noise_factor=0.05):
    """Add random noise to image"""
    noise = np.random.randn(*image.shape) * noise_factor
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image

def apply_augmentations(image):
    """Apply random augmentations to image"""
    if np.random.random() > 0.5:
        image = random_rotation(image)
    if np.random.random() > 0.5:
        image = random_scale(image)
    if np.random.random() > 0.5:
        image = add_noise(image)
    return image