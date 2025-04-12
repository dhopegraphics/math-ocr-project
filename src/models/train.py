import tensorflow as tf
import numpy as np
import os
from src.config import CHARACTER_SET
from src.models.model import build_crnn_model
from src.data.preprocessing import preprocess_image, load_im2latex_dataset
from src.data.augmentation import apply_augmentations
from src.data.im2latex_loader import extract_images

ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
EarlyStopping = tf.keras.callbacks.EarlyStopping
TensorBoard = tf.keras.callbacks.TensorBoard

def ctc_loss(y_true, y_pred):
    batch_len = tf.shape(y_true)[0]
    input_length = tf.ones((batch_len, 1), dtype='int32') * tf.shape(y_pred)[1]
    label_length = tf.ones((batch_len, 1), dtype='int32') * tf.shape(y_true)[1]
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)


def prepare_dataset(images, labels, augment_prob=0.5):
    from collections import Counter

    processed_images = []
    processed_labels = []
    bad_indices = []
    shapes = []

    for i, (img, label) in enumerate(zip(images, labels)):
        img = preprocess_image(img)

        img = np.array(img)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        elif img.ndim == 3 and img.shape[-1] != 1:
            img = img[:, :, :1]

        # Resize to ensure shape is exactly (64, 256, 1)
        from tensorflow.image import resize_with_pad
        img = resize_with_pad(img, 64, 256).numpy()
        img = img.astype(np.float32)

        if img.shape != (64, 256, 1):
            print(f"[❌] Skipping image {i}: shape {img.shape} is invalid")
            bad_indices.append(i)
            continue

        processed_images.append(img)
        processed_labels.append([CHARACTER_SET.index(c) for c in label if c in CHARACTER_SET])
        shapes.append(img.shape)

    # 🧠 Check if shapes are actually uniform
    shape_counts = Counter(shapes)
    print(f"[📊] Shape counts: {shape_counts}")

    if len(shape_counts) > 1:
        raise ValueError(f"Inconsistent image shapes detected: {shape_counts}")

    if len(processed_images) == 0:
        raise ValueError("No valid images found after preprocessing.")

    return np.stack(processed_images), tf.keras.preprocessing.sequence.pad_sequences(
        processed_labels, maxlen=128, padding='post')
    
def train_model(dataset_path, dataset_type='IM2LATEX', epochs=50):
    if dataset_type == "IM2LATEX":
        tar_path = os.path.join(dataset_path, "formula_images.tar.gz")
        if os.path.exists(tar_path):
            print("Extracting IM2LATEX images...")
            extract_images(tar_path, os.path.join(dataset_path, "formula_images"))

        print("Loading IM2LATEX dataset...")
        images, labels = load_im2latex_dataset(dataset_path, split='train')
    else:
        raise ValueError("Unsupported dataset type")

    print("Preprocessing images...")
    X_train, y_train = prepare_dataset(images, labels)
    print(f"[DEBUG] Loaded {len(X_train)} training samples")
    if len(X_train) == 0:
        raise ValueError("No training images were loaded or preprocessed.")

    print(f"[DEBUG] Shape of first training image: {X_train[0].shape}")

    split_idx = int(0.8 * len(X_train))
    X_val, y_val = X_train[split_idx:], y_train[split_idx:]
    X_train, y_train = X_train[:split_idx], y_train[:split_idx]
    

    input_shape = X_train.shape[1:]
    num_classes = len(CHARACTER_SET)

    model = build_crnn_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss=ctc_loss)

    callbacks = [
        ModelCheckpoint('saved_models/best_model.h5', monitor='val_loss', save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        TensorBoard(log_dir='./logs')
    ]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs,
                        batch_size=32, callbacks=callbacks)

    return model, history
