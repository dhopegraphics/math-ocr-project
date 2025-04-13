import numpy as np
from ..data.tokenizer import Tokenizer
import tensorflow as tf  # Main import
from ..data.preprocessing import preprocess_image

class Predictor:
    def __init__(self, model_path, tokenizer):
        self.tokenizer = tokenizer
        self.model = self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Load model using standard TensorFlow imports"""
        return tf.keras.models.load_model(model_path)
    
    def predict(self, image):
        # Preprocess image
        processed_img = preprocess_image(image)
        img_array = np.expand_dims(processed_img, axis=0)
        
        # Get encoder states
        states_value = self.model.encoder_model.predict(img_array)
        
        # Initialize target sequence
        target_seq = np.zeros((1, 1, self.tokenizer.vocab_size))
        target_seq[0, 0, self.tokenizer.token2id['<start>']] = 1
        
        # Generate sequence
        stop_condition = False
        decoded_tokens = []
        max_len = self.tokenizer.max_len
        
        while not stop_condition:
            output_tokens, h, c = self.model.decoder_model.predict(
                [target_seq] + states_value)
            
            # Sample token
            sampled_token_id = np.argmax(output_tokens[0, -1, :])
            sampled_token = self.tokenizer.id2token[sampled_token_id]
            
            if sampled_token == '<end>' or len(decoded_tokens) >= max_len:
                stop_condition = True
            else:
                decoded_tokens.append(sampled_token)
                
                # Update target sequence
                target_seq = np.zeros((1, 1, self.tokenizer.vocab_size))
                target_seq[0, 0, sampled_token_id] = 1
                
                # Update states
                states_value = [h, c]
        
        # Convert tokens to LaTeX
        latex = self.tokenizer.tokens_to_latex(decoded_tokens)
        return latex