import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

model = None

def model_fn(model_dir):
    """
    Load the model from the model directory.
    SageMaker will call this function automatically.
    """
    global model
    model_path = os.path.join(model_dir, '1', 'deepfake_detector.keras')
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded from:", model_path)
    return model

def input_fn(request_body, content_type='application/json'):
    """
    Convert the incoming request to a model input.
    """
    if content_type == 'application/json':
        data = json.loads(request_body)
        img_bytes = bytearray(data['image'])  # Expect image as list of bytes
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))

def predict_fn(input_data, model):
    """
    Run model prediction.
    """
    preds = model.predict(input_data)
    return preds

def output_fn(prediction, accept='application/json'):
    """
    Format the output as JSON.
    """
    score = prediction[0][0]
    label = 'real' if score > 0.5 else 'fake'
    confidence = score if label == 'real' else 1 - score
    response = {
        'label': label,
        'confidence': round(confidence * 100, 2)
    }
    return json.dumps(response)
