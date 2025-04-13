import os
import numpy as np
from PIL import Image
from .im2latex_loader import load_formulas, parse_lst_file, crop_formula_image
from src.config import IMAGE_HEIGHT, IMAGE_WIDTH




def preprocess_image(image):
    """
    Resize image to fixed model input size and normalize it.
    Ensures output shape is (H, W, 1).
    """
    from PIL import Image
    import numpy as np
    from src.config import IMAGE_HEIGHT, IMAGE_WIDTH

    if isinstance(image, str):
        image = Image.open(image).convert('L')  # Convert to grayscale

    elif image.mode != 'L':
        image = image.convert('L')

    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = np.array(image).astype(np.float32) / 255.0  # Normalize

    # Add channel dim if missing
    if img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)

    return img_array


def load_im2latex_dataset(base_dir, split='train'):
    from .im2latex_loader import load_formulas, parse_lst_file, crop_formula_image
    import os

    lst_path = os.path.join(base_dir, f"im2latex_{split}.lst")
    formulas_path = os.path.join(base_dir, "im2latex_formulas.lst")
    images_dir = os.path.join(base_dir, "formula_images")

    print(f"[INFO] Loading formulas from: {formulas_path}")
    formulas = load_formulas(formulas_path)

    print(f"[INFO] Parsing list file: {lst_path}")
    lst_data = parse_lst_file(lst_path)
    print(f"[INFO] Total entries in list file: {len(lst_data)}")

    images = []
    labels = []

    for i, item in enumerate(lst_data):
        img_path = os.path.join(images_dir, f"{item['image_name']}.png")
        if not os.path.exists(img_path):
            print(f"[❌] Missing image: {img_path}")
            continue

        cropped = crop_formula_image(img_path)
        if cropped is None:
            print(f"[⚠️] Skipped blank or failed crop: {img_path}")
            continue

        images.append(cropped)
        labels.append(formulas[item['formula_idx']])

    print(f"[✅] Final loaded images: {len(images)} / {len(lst_data)}")
    return images, labels