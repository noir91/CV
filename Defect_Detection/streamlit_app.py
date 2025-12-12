import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# --- 1. Model Architecture (Copied from training) ---
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

        self._to_linear = None
        self._calculate_flat_features()

        self.fc1 = nn.Linear(self._to_linear, dense_nodes)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(dense_nodes, 1)

    def _calculate_flat_features(self):
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

# --- 2. Configuration & Caching ---
MODEL_PATH = "models/best_model.pt" # Ensure this file is in your GitHub Repo root!
DEVICE = torch.device('cpu') # Cloud free tiers usually don't have GPU, force CPU

@st.cache_resource
def load_model():
    try:
        model = CNN().to(DEVICE)
        # map_location=torch.device('cpu') is crucial for cloud deployment
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_image(image):
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform_pipeline(image).unsqueeze(0).to(DEVICE)

# --- 3. Streamlit UI ---
st.set_page_config(page_title="Defect Detector", page_icon="🔍")

st.title("🔍 Industrial Defect Detection")
st.write("Upload an image to check for defects.")

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)

    with col2:
        st.write("### Analysis")
        if st.button("Run Inspection"):
            with st.spinner("Analyzing..."):
                tensor = process_image(image)
                
                with torch.no_grad():
                    output = model(tensor)
                    prob = torch.sigmoid(output).item()
                
                prediction = "Defect" if prob > 0.5 else "No Defect"
                confidence = prob if prob > 0.5 else 1 - prob
                
                if prediction == "Defect":
                    st.error(f" **Result:** {prediction}")
                else:
                    st.success(f" **Result:** {prediction}")
                    
                st.metric("Confidence Score", f"{confidence*100:.2f}%")