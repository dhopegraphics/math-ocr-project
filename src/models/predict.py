import tensorflow as tf
import numpy as np
from .model import build_crnn_model
from src.data.preprocessing import preprocess_image
from src.config import CHARACTER_SET  # Optional: if you're loading characters from config

class MathOCR:
    def __init__(self, model_path='saved_models/best_model.keras'):
        self.characters = CHARACTER_SET  # Load your character set from a central config
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'ctc_loss': tf.keras.backend.ctc_batch_cost},
            compile=False
        )

    def decode_prediction(self, pred):
         pred_indices = np.argmax(pred, axis=-1)
         print(f"[DEBUG] Argmax indices: {pred_indices}")

         decoded = []
         prev_idx = -1
         blank_idx = len(self.characters)  # index of CTC blank

         for idx in pred_indices:
             if idx != prev_idx and idx != blank_idx:
                if idx < len(self.characters):
                   decoded.append(self.characters[idx])
             prev_idx = idx
 
         return ''.join(decoded)
     

    def predict(self, image_path):
        """
        Run OCR prediction on a single image and return LaTeX.
        """
        # Preprocess image
        img = preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Predict
        pred = self.model.predict(img)
        print(f"[DEBUG] Raw output shape: {pred.shape}")
        print(f"[DEBUG] Argmax indices: {np.argmax(pred[0], axis=-1)}")

        # Decode to text
        math_text = self.decode_prediction(pred[0])
        print(f"[DEBUG] Decoded text: {math_text}")

        # Convert to LaTeX
        latex = self.text_to_latex(math_text)
        print(f"[DEBUG] Converted to LaTeX: {latex}")

        return latex

    def text_to_latex(self, text):
        """
        Simple placeholder for LaTeX conversion — you can make this more advanced.
        """
        replacements = {
            '^': '^',
            '_': '_',
            '/': '\\frac{}{}',
            'sqrt': '\\sqrt{}',
            'sum': '\\sum_{}^{}',
            'int': '\\int_{}^{}'
        }

        for k, v in replacements.items():
            text = text.replace(k, v)

        return text