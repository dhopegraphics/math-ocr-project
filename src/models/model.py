import tensorflow as tf

def build_crnn_model(input_shape, num_classes):
    input_img = tf.keras.Input(shape=input_shape, name='input_image')

    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_img)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((1, 2))(x)

    x = tf.keras.layers.Reshape((-1, 128))(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(num_classes + 1, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_img, outputs=x)
    return model
