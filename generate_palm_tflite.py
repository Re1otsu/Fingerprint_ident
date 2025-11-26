import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------
# 1. Создаем сеть эмбеддингов (MobileNetV3 + Embedding head)
# -----------------------------

def build_palm_embedder(input_shape=(200, 200, 3), embedding_dim=128):
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    base.trainable = False  # Заморозили веса

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    x = base(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(embedding_dim)(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)

    model = models.Model(inputs, x, name="PalmEmbeddingModel")
    return model


# -----------------------------
# 2. Сохраняем Keras-модель
# -----------------------------
model = build_palm_embedder()
model.save("palm_embedder.h5")
print("Keras model saved as palm_embedder.h5")


# -----------------------------
# 3. Конвертация в TFLite
# -----------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("palm_embedding_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as palm_embedding_model.tflite")
