import tensorflow as tf

class MathOCRModel:
    def __init__(self, vocab_size, max_formula_len, input_shape=(100, 300, 1)):
        self.vocab_size = vocab_size
        self.max_formula_len = max_formula_len
        self.input_shape = input_shape
    
    def build_cnn(self):
        """Build CNN feature extractor"""
        Input = tf.keras.layers.Input
        Conv2D = tf.keras.layers.Conv2D
        BatchNorm = tf.keras.layers.BatchNormalization
        MaxPool = tf.keras.layers.MaxPooling2D
        Permute = tf.keras.layers.Permute
        TimeDistributed = tf.keras.layers.TimeDistributed
        Flatten = tf.keras.layers.Flatten
        
        input_img = Input(shape=self.input_shape)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = BatchNorm()(x)
        x = MaxPool((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNorm()(x)
        x = MaxPool((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNorm()(x)
        x = MaxPool((1, 2))(x)
        x = Permute((2, 1, 3))(x)
        x = TimeDistributed(Flatten())(x)
        
        return tf.keras.Model(input_img, x)
    
    def build_model(self):
        """Build full model with encoder-decoder architecture"""
        Input = tf.keras.layers.Input
        Bidirectional = tf.keras.layers.Bidirectional
        LSTM = tf.keras.layers.LSTM
        Concatenate = tf.keras.layers.Concatenate
        Dense = tf.keras.layers.Dense
        Model = tf.keras.Model
        
        # Encoder
        image_input = Input(shape=self.input_shape, name='image_input')
        cnn_features = self.build_cnn()(image_input)
        
        # Changed to properly handle encoder_outputs
        _, fw_h, fw_c, bw_h, bw_c = Bidirectional(
            LSTM(256, return_sequences=True, return_state=True)
        )(cnn_features)
        
        encoder_states = [
            Concatenate()([fw_h, bw_h]),
            Concatenate()([fw_c, bw_c])
        ]
        
        # Decoder
        decoder_input = Input(shape=(None, self.vocab_size), name='decoder_input')
        decoder_lstm = LSTM(512, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
        decoder_outputs = Dense(self.vocab_size, activation='softmax')(decoder_outputs)
        
        # Models
        training_model = Model(
            inputs=[image_input, decoder_input],
            outputs=decoder_outputs)
        
        encoder_model = Model(image_input, encoder_states)
        
        # Decoder inference
        state_h = Input(shape=(512,))
        state_c = Input(shape=(512,))
        decoder_outputs, h, c = decoder_lstm(
            decoder_input, initial_state=[state_h, state_c])
        decoder_outputs = Dense(self.vocab_size, activation='softmax')(decoder_outputs)
        
        decoder_model = Model(
            inputs=[decoder_input, state_h, state_c],
            outputs=[decoder_outputs, h, c])
        
        return training_model, encoder_model, decoder_model