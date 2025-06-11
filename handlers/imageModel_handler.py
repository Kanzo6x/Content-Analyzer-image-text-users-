import numpy as np
import tensorflow as tf
from tensorflow import keras
import io
import base64
from PIL import Image

class ImageModel:
    def __init__(self):
        try:
            print("Loading VGG16 model from imageAnalyzer.h5...")
            self.model = keras.models.load_model('models_and_csvs/imageAnalyzer.h5')
            print("Successfully loaded VGG16 model")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def preprocess_image(self, img_data):
        # Convert base64 to image
        img = Image.open(io.BytesIO(base64.b64decode(img_data)))
        
        # Convert to RGB if image is in RGBA mode
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize and preprocess
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = tf.convert_to_tensor(img_array / 255.0, dtype=tf.float32)
        img_array = tf.expand_dims(img_array, axis=0)
        
        return img_array

    def predict_image(self, img_data):
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(img_data)
            
            # Make prediction
            prediction = self.model.predict(processed_image)[0][0]
            
            # Determine class and confidence
            is_dog = prediction > 0.5
            confidence = prediction if is_dog else 1 - prediction
            class_name = "Dog" if is_dog else "Cat"
            
            # Check for harmful content threshold
            if confidence < 0.98:
                class_name = "Not Harmful"
                
            return {
                'class_name': str(class_name),
                'confidence': float(confidence),
                'is_harmful': int(confidence >= 0.98)  # Convert bool to int
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None
