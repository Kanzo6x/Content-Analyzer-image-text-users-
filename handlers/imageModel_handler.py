import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import os

class ImageModel:
    def __init__(self):
        try:
            print("Loading model from imageAnalyzer.pth...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Initialize VGG16 model architecture
            self.model = self._create_model()
            
            # Get the absolute path to the project root directory
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(project_root, "models_and_csvs", "imageAnalyzer.pth")
            
            # Load the trained weights
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("Successfully loaded model")
            
            # Define image transformations
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def _create_model(self):
        import torchvision.models as models
        import torch.nn as nn
        from torchvision.models import VGG16_Weights
        
        # Use the new weights parameter instead of pretrained
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
            
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        return model.to(self.device)

    def preprocess_image(self, img_data):
        # Convert base64 to image
        img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        return img_tensor

    def predict_image(self, img_data):
        try:
            # Preprocess the image
            img_tensor = self.preprocess_image(img_data)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(img_tensor).item()
            
            # Determine class and confidence
            is_dog = prediction > 0.5
            confidence = prediction if is_dog else 1 - prediction
            class_name = "Dog" if is_dog else "Cat"
            
            # Check for harmful content threshold
            if confidence < 0.99:
                class_name = "Not Harmful"
                
            return {
                'class_name': str(class_name),
                'confidence': float(confidence),
                'is_harmful': int(confidence >= 0.99)
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None
