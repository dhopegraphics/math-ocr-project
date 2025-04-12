import tensorflow as tf
import numpy as np
from .model import build_crnn_model
from src.data.preprocessing import preprocess_image

class MathOCR:
    def __init__(self, model_path='saved_models/best_model.h5'):
        self.model = tf.keras.models.load_model(model_path, 
                              custom_objects={'ctc_loss': tf.keras.backend.ctc_batch_cost})
        
        self.characters = "0123456789+-=(){}[]abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,;:!?/\\&@#$%^*_|"  # Add all math symbols
    
    def decode_prediction(self, pred):
        """Convert model output to text"""
        # Use greedy search to find most likely characters
        pred_text = []
        for time_step in pred:
            char_idx = np.argmax(time_step)
            if char_idx != len(self.characters):  # Blank character (CTC)
                pred_text.append(self.characters[char_idx])
        
        # Merge repeated characters
        final_text = []
        prev_char = None
        for char in pred_text:
            if char != prev_char:
                final_text.append(char)
            prev_char = char
        
        return ''.join(final_text)
    
    def predict(self, image_path):
        """Predict math expression from image"""
        # Preprocess image
        img = preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Predict
        pred = self.model.predict(img)
        
        # Decode prediction
        math_text = self.decode_prediction(pred[0])
        
        # Convert to LaTeX (simple example)
        latex = self.text_to_latex(math_text)
        
        return latex
    
    def text_to_latex(self, text):
        """Simple text to LaTeX conversion (you'll need to expand this)"""
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