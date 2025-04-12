import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from .model import build_crnn_model
from src.data.preprocessing import preprocess_image
from src.data.augmentation import apply_augmentations
import os

def load_dataset(data_dir, num_samples=None):
    """Load and preprocess dataset"""
    images = []
    labels = []
    
    for filename in os.listdir(data_dir)[:num_samples]:
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(data_dir, filename)
            
            # Preprocess image
            img = preprocess_image(img_path)
            
            # Apply augmentation (50% chance)
            if tf.random.uniform(()) > 0.5:
                img = apply_augmentations(img)
            
            images.append(img)
            
            # TODO: Load corresponding label (you'll need to implement this)
            # labels.append(load_label_for_image(filename))
    
    return np.array(images), np.array(labels)

def train_model():
    # Parameters
    input_shape = (100, 300, 1)  # height, width, channels
    num_classes = 100  # Number of math symbols in your vocabulary
    
    # Load data
    X_train, y_train = load_dataset('data/processed/train')
    X_val, y_val = load_dataset('data/processed/val')
    
    # Build model
    model = build_crnn_model(input_shape, num_classes)
    
    # Compile model with CTC loss (you'll need to implement this)
    model.compile(optimizer='adam', 
                 loss=ctc_loss,  # Custom CTC loss function
                 metrics=['accuracy'])
    
    # Callbacks
    callbacks = [
        ModelCheckpoint('saved_models/best_model.h5', save_best_only=True),
        EarlyStopping(patience=5),
        TensorBoard(log_dir='./logs')
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks
    )
    
    return model, history