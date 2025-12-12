import torch
import torch.nn as nn
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import uvicorn

# --- 1. Define Model Architecture (Must match training exactly) ---
class CNN(nn.Module):
    def __init__(self, out_1=32, out_2=64, dense_nodes=128):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, out_1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_1)
        self.relu1 = nn.ReLU()
        self.max2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(out_1, out_2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_2)
        self.relu2 = nn.ReLU()
        self.max2d_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Helper to calc flat features
        self._to_linear = None
        self._calculate_flat_features()

        self.fc1 = nn.Linear(self._to_linear, dense_nodes)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(dense_nodes, 1)

    def _calculate_flat_features(self):
        # Dummy input matching training size (224x224)
        dummy = torch.zeros(1, 3, 224, 224)
        x = self.relu1(self.bn1(self.conv1(dummy)))
        x = self.max2d_1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.max2d_2(x)
        self._to_linear = x.view(1, -1).size(1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.max2d_1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.max2d_2(x)
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. Setup API & Model ---
app = FastAPI(title="Defect Detection API")

MODEL_PATH = "models/best_model.pt" # Path from your training script
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

# Preprocessing pipeline (Must match training)
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.on_event("startup")
def load_model():
    global model
    try:
        model = CNN().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read Image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # 2. Transform
    tensor = transform_pipeline(image).unsqueeze(0).to(DEVICE)
    
    # 3. Inference
    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()
        
    # 4. Interpret Result (Threshold 0.5)
    prediction = "Defect" if prob > 0.5 else "No Defect"
    confidence = prob if prob > 0.5 else 1 - prob
    
    return {
        "filename": file.filename,
        "prediction": prediction,
        "probability": prob,
        "confidence": f"{confidence*100:.2f}%"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)