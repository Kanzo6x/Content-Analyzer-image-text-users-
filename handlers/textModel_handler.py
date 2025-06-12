import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

class ClassificationModel(nn.Module):
    def __init__(self, base_model, hidden_size=384, num_labels=2):
        super(ClassificationModel, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(embeddings)
        return logits

class ToxicityModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        # Load base transformer model
        self.base_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        # Init classification model
        self.model = ClassificationModel(self.base_model)
        
        # Load trained weights
        try:
            print("Loading model weights from blocktext.pth...")
            self.model.load_state_dict(torch.load("models_and_csvs/blocktext.pth", map_location=self.device))
            print("Successfully loaded model weights from blocktext.pth")
        except Exception as e:
            print(f"Error loading model weights: {str(e)}")
            print("Using base model instead")

        self.model.to(self.device)
        self.model.eval()

    def predict_toxicity(self, text, max_length=128):
        try:
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                prediction = torch.argmax(outputs, dim=1).item()

            return prediction
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return 0