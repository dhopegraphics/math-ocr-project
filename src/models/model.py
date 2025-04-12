from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Dropout

def build_crnn_model(input_shape, num_classes):
    """Build a CNN-RNN model for sequence prediction"""
    # Input layer
    input_img = Input(shape=input_shape, name='input_image')
    
    # CNN layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), name='pool1')(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), name='pool2')(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((1, 2), name='pool3')(x)
    
    # Prepare for RNN
    x = Reshape((-1, 128))(x)  # Convert to sequence
    
    # RNN layers
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    
    # Output layer
    x = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=input_img, outputs=x)
    
    return model