import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
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
            
            # Define image transformations (matching the training transforms)
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
        
        # Use pretrained VGG16
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
            
        # Match the new classifier architecture
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        return model.to(self.device)

    def predict_image(self, file):
        try:
            # Preprocess the image
            img = Image.open(file).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(img_tensor).item()
            
            # Determine class (harmful vs not harmful)
            is_harmful = prediction > 0.5
            confidence = prediction if is_harmful else 1 - prediction
            class_name = "Harmful" if is_harmful else "Not Harmful"
            
            return {
                'class_name': str(class_name),
                'confidence': float(confidence),
                'is_harmful': int(is_harmful)
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None