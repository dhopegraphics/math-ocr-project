from ..data.im2latex_loader import Im2LatexDataset
from ..data.tokenizer import Tokenizer
from .model import MathOCRModel

import tensorflow as tf
import os
import numpy as np
import cv2
from tqdm import tqdm
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def train_model(data_dir, batch_size=32, epochs=50):
    """Train the Math OCR model with enhanced logging and error handling"""
    try:
        # Initialize Keras components
        Adam = tf.keras.optimizers.Adam
        EarlyStopping = tf.keras.callbacks.EarlyStopping
        ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
        TensorBoard = tf.keras.callbacks.TensorBoard

        # Load dataset
        logging.info("Loading dataset...")
        dataset = Im2LatexDataset(data_dir)
        tokenizer = Tokenizer()
        
        # Build vocabulary with progress bar
        logging.info("Building vocabulary...")
        formulas = [formula for _, formula in tqdm(dataset.train, desc="Processing formulas")]
        tokenizer.build_vocab(formulas)
        logging.info(f"Vocabulary built with {tokenizer.vocab_size} tokens")
        
        # Prepare data generators
        logging.info("Initializing data generators...")
        train_gen = DataGenerator(dataset.train, tokenizer, batch_size)
        val_gen = DataGenerator(dataset.val, tokenizer, batch_size)
        
        # Build model
        logging.info("Building model architecture...")
        model_builder = MathOCRModel(
            vocab_size=tokenizer.vocab_size,
            max_formula_len=tokenizer.max_len)
        model, encoder_model, decoder_model = model_builder.build_model()
        
        # Compile model
        logging.info("Compiling model...")
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        
        # Ensure output directories exist
        os.makedirs('saved_models', exist_ok=True)
        os.makedirs('./logs', exist_ok=True)
        
        # Create timestamp for this training session
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Enhanced callbacks
        callbacks_list = [
            ModelCheckpoint(
                f'saved_models/best_model_{timestamp}.keras',
                save_best_only=True,
                monitor='val_loss',
                save_weights_only=False,
                verbose=1),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1),
            TensorBoard(
                log_dir=f'./logs/{timestamp}',
                histogram_freq=1,
                update_freq='epoch'),
            tf.keras.callbacks.CSVLogger(f'training_log_{timestamp}.csv'),
            tf.keras.callbacks.ProgbarLogger(count_mode='steps')
        ]
        
        # Train model with enhanced logging
        logging.info(f"Starting training for {epochs} epochs...")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=1)
        
        # Save tokenizer with timestamp
        tokenizer_path = f'saved_models/tokenizer_{timestamp}.json'
        tokenizer.save(tokenizer_path)
        logging.info(f"Training completed successfully! Tokenizer saved to {tokenizer_path}")
        
        return history, model, encoder_model, decoder_model
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise

class DataGenerator:
    def __init__(self, data_pairs, tokenizer, batch_size):
        """Initialize data generator with enhanced error handling"""
        self.data_pairs = data_pairs
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.steps = len(data_pairs) // batch_size
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized generator with {len(data_pairs)} samples, {self.steps} steps per epoch")
    
    def __len__(self):
        return self.steps
    
    def __iter__(self):
        """Yield batches indefinitely with improved error handling"""
        while True:
            # Shuffle at start of each epoch
            np.random.shuffle(self.data_pairs)
            
            for i in range(self.steps):
                try:
                    batch_pairs = self.data_pairs[i*self.batch_size:(i+1)*self.batch_size]
                    X, y = self._prepare_batch(batch_pairs)
                    yield X, y
                except Exception as e:
                    self.logger.error(f"Error generating batch {i}: {str(e)}")
                    # Fallback to a known good batch
                    yield self._prepare_batch(self.data_pairs[:self.batch_size])
    
    def _prepare_batch(self, batch_pairs):
        """Prepare a batch of data with comprehensive error handling"""
        X_images = []
        y_formulas = []
        error_count = 0
        
        for img_path, formula in batch_pairs:
            try:
                # Load and validate image
                img = cv2.imread(img_path)
                if img is None:
                    raise ValueError(f"Failed to load image at {img_path}")
                    
                # Preprocess image
                preprocessed = preprocess_image(img)
                if preprocessed is None:
                    raise ValueError("Preprocessing returned None")
                    
                X_images.append(preprocessed)
                y_formulas.append(formula)
                
            except Exception as e:
                error_count += 1
                self.logger.debug(f"Skipping {img_path}: {str(e)}")
                continue
        
        # Log warnings if we skipped samples
        if error_count > 0:
            self.logger.warning(f"Skipped {error_count}/{len(batch_pairs)} problematic samples in this batch")
        
        # Fallback if batch is empty
        if not X_images:
            self.logger.error("Empty batch after filtering, using fallback")
            return self._prepare_batch(self.data_pairs[:self.batch_size])
        
        # Tokenize formulas
        try:
            y_seq = self.tokenizer.texts_to_sequences(y_formulas)
            y_padded = self.tokenizer.pad_sequences(y_seq)
            
            # Convert to one-hot
            y_onehot = np.zeros((len(y_padded), self.tokenizer.max_len, self.tokenizer.vocab_size))
            for i, seq in enumerate(y_padded):
                for t, token_id in enumerate(seq):
                    if token_id > 0:  # Skip padding
                        y_onehot[i, t, token_id] = 1
            
            # Prepare decoder input (shifted right)
            decoder_input = np.zeros_like(y_onehot)
            decoder_input[:, 1:, :] = y_onehot[:, :-1, :]
            decoder_input[:, 0, self.tokenizer.token2id['<start>']] = 1
            
            return [np.array(X_images), decoder_input], y_onehot
            
        except Exception as e:
            self.logger.error(f"Error tokenizing batch: {str(e)}")
            raise

def preprocess_image(image, target_size=(300, 100)):
    """Preprocess image for model input with validation"""
    try:
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
        
        # Normalize and add channel dimension
        normalized = padded / 255.0
        return np.expand_dims(normalized, axis=-1)
        
    except Exception as e:
        logging.error(f"Error preprocessing image: {str(e)}")
        return None