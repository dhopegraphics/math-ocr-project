import numpy as np
import cv2
from scipy.ndimage import rotate

def random_rotation(image, max_angle=10):
    angle = np.random.uniform(-max_angle, max_angle)
    return rotate(image, angle, reshape=False, mode='nearest')

def random_scale(image, scale_range=(0.9, 1.1)):
    h, w = image.shape[:2]
    scale = np.random.uniform(*scale_range)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(image, (new_w, new_h))

    if scale > 1:
        start_h, start_w = (new_h - h) // 2, (new_w - w) // 2
        img = img[start_h:start_h+h, start_w:start_w+w]
    else:
        pad_h, pad_w = h - new_h, w - new_w
        img = np.pad(img, ((pad_h//2, pad_h - pad_h//2),
                           (pad_w//2, pad_w - pad_w//2)), mode='constant')
    return img

def add_noise(image, noise_factor=0.05):
    noise = np.random.randn(*image.shape) * noise_factor
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def apply_augmentations(image, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if np.random.rand() > 0.5:
        image = random_rotation(image)
    if np.random.rand() > 0.5:
        image = random_scale(image)
    if np.random.rand() > 0.5:
        image = add_noise(image)
    return image
