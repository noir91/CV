import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
import time
import random
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
class Config:
    # System
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    NUM_WORKERS = 8
    
    # Data
    DATA_PATH = r'/media/noir/SSD/Defect/'
    IMG_SIZE = (224, 224)
    
    # Hyperparameters
    EPOCHS = 10
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 2
    
    # Model Architecture (Defaults)
    OUT_1 = 32
    OUT_2 = 64
    DENSE_LAYERS = 128
    
    # Logging
    SAVE_DIR = ""
    MODEL_NAME = "best_model.pt"

# --- Reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- Data Loading ---
def get_datasets(data_path):
    print(f"Loading data from: {data_path}")
    
    transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor()
    ])
    
    try:
        dataset = ImageFolder(data_path, transform=transform)
    except FileNotFoundError:
        print(f"Error: Path '{data_path}' not found. Please check Config.DATA_PATH.")
        return None, None, None

    # Split size
    total_length = len(dataset)
    indices = list(range(total_length))
    random.shuffle(indices)

    train_size = int(0.70 * total_length)
    val_size = int(0.15 * total_length)

    # Indexes 
    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    # Subsets
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)
    
    print(f"Data Loaded: Train({len(train_subset)}), Val({len(val_subset)}), Test({len(test_subset)})")
    return train_subset, val_subset, test_subset

# --- Model Definition ---
class CNN(nn.Module):
    def __init__(self, out_1=Config.OUT_1, out_2=Config.OUT_2, dense_nodes=Config.DENSE_LAYERS):
        super(CNN, self).__init__()
        
        # Convolution Layer 1
        self.conv1 = nn.Conv2d(3, out_1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_1)
        self.relu1 = nn.ReLU()
        self.max2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolution Layer 2
        self.conv2 = nn.Conv2d(out_1, out_2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_2)
        self.relu2 = nn.ReLU()
        self.max2d_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dynamic Flat Features Calculation
        self._to_linear = None
        self._calculate_flat_features()

        # Dense Layers
        self.fc1 = nn.Linear(self._to_linear, dense_nodes)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(dense_nodes, 1) # Binary classification output

    def _calculate_flat_features(self):
        # Pass a dummy input to calculate the flattened size automatically
        dummy = torch.zeros(1, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
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

# --- Training Helper Functions ---

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1) # BCE requires [Batch, 1]

        output = model(imgs)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        
        preds = (torch.sigmoid(output) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

    if total == 0: return 0.0, 0.0
    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            output = model(imgs)
            loss = criterion(output, labels)

            running_loss += loss.item() * imgs.size(0)
            
            preds = (torch.sigmoid(output) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)
            
    if total == 0: return 0.0, 0.0
    return running_loss / total, correct / total

def plot_metrics(history, save_dir="."):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.style.use('ggplot')
    plt.figure(figsize=(14, 6))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r--', label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r--', label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path)
    print(f"Curves saved to: {save_path}")
    # plt.show() # Uncomment if running in a notebook/environment with display

# --- Main Logic ---

def train_pipeline(model, trainset, valset):
    # 1. Loaders
    train_loader = DataLoader(trainset, shuffle=True, batch_size=Config.BATCH_SIZE, 
                              num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(valset, shuffle=False, batch_size=Config.BATCH_SIZE, 
                            num_workers=Config.NUM_WORKERS, pin_memory=True)

    # 2. Setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    save_path = os.path.join(Config.SAVE_DIR, Config.MODEL_NAME)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"Starting training on {Config.DEVICE}...")

    # 3. Loop
    for epoch in range(Config.EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)
        
        # Record
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        duration = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{Config.EPOCHS}] ({duration:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Checkpoint & Early Stopping
        if val_loss < best_val_loss:
            print(f"--> Improved. Saving model to {save_path}")
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break
                
    return history

if __name__ == "__main__":
    set_seed(Config.SEED)
    
    # 1. Get Data
    train_subset, val_subset, test_subset = get_datasets(Config.DATA_PATH)
    
    if train_subset:
        # 2. Init Model
        model = CNN().to(Config.DEVICE)
        
        # 3. Train
        history = train_pipeline(model, train_subset, val_subset)
        
        # 4. Visualize
        plot_metrics(history, save_dir=Config.SAVE_DIR)
        
        print("Running final evaluation on Test Set...")
        model.load_state_dict(torch.load(os.path.join(Config.SAVE_DIR, Config.MODEL_NAME)))
        test_loader = DataLoader(test_subset, batch_size=Config.BATCH_SIZE, shuffle=False)
        test_loss, test_acc = validate(model, test_loader, nn.BCEWithLogitsLoss(), Config.DEVICE)
        print(f"Test Set Results - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")