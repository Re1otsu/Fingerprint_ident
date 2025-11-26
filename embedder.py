import numpy as np
import cv2
import tensorflow as tf

interpreter = tf.lite.Interpreter("palm_embedding_model.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

def get_embedding(roi):
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(inp["index"], img)
    interpreter.invoke()

    emb = interpreter.get_tensor(out["index"])[0]
    emb = emb / np.linalg.norm(emb)
    return emb
