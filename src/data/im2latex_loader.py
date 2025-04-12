import os
import tarfile
from PIL import Image
import numpy as np

def extract_images(tar_path, output_dir):
    import tarfile, os

    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 10:
        print(f"[INFO] Skipping extraction; images already exist at: {output_dir}")
        return

    print(f"[INFO] Extracting images to: {output_dir}")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=output_dir)

    print(f"[✅] Extraction complete! Images now available in: {output_dir}")

def load_formulas(formulas_path):
    """Load LaTeX formulas from im2latex_formulas.lst with fallback encoding"""
    with open(formulas_path, 'r', encoding='utf-8', errors='replace') as f:
        formulas = [line.strip() for line in f]
    return formulas

def parse_lst_file(lst_path):
    """Parse im2latex_[train|val|test].lst files"""
    data = []
    with open(lst_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                data.append({
                    'formula_idx': int(parts[0]),
                    'image_name': parts[1],
                    'render_type': parts[2]
                })
    return data


def crop_formula_image(image_path):
    from PIL import Image
    import numpy as np

    try:
        img = Image.open(image_path).convert("L")
        img = np.array(img)

        if img.ndim != 2:
            print(f"[⚠️] Not grayscale: {image_path}")
            return None

        non_empty = np.where(img < 255)
        if len(non_empty[0]) == 0:
            print(f"[ℹ️] Blank image (all white): {image_path}")
            return None

        min_y, max_y = non_empty[0].min(), non_empty[0].max()
        min_x, max_x = non_empty[1].min(), non_empty[1].max()

        padding = 10
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(img.shape[1], max_x + padding)
        max_y = min(img.shape[0], max_y + padding)

        cropped = img[min_y:max_y, min_x:max_x]
        return Image.fromarray(cropped)
    except Exception as e:
        print(f"[💥] Error loading image: {image_path} — {e}")
        return None