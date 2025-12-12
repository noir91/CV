# Industrial Defect Detection System

This project implements an automated visual inspection tool leveraging Deep Learning to detect manufacturing defects. The system utilizes a Convolutional Neural Network (CNN) to classify images with **98.95% accuracy**, demonstrating the application of computer vision in quality control automation.

### Project Structure

The repository is organized into distinct modules for training, inference, and user interaction.

```plaintext
CV/
├── Defect Detection/
│   ├── models/                
│   │   ├── best_model.pt      # Serialized PyTorch model state dictionary
│   │   └── training_curves.png# Visualization of training metrics
│   │
│   ├── api.py                 # Backend API (FastAPI) for inference logic
│   │
│   ├── app.py                 # Frontend UI (Streamlit) for user interaction
│   │
│   ├── train.py               # Script for model training and validation
│   │
│   └── requirements.txt       # Project dependencies
```
### Architecture

The application adopts a microservices-oriented architecture to decouple the user interface from the inference logic:

    Frontend: Built with Streamlit, serving as the user interface for image upload and result visualization.

    Backend: Developed using FastAPI, this service handles image preprocessing, tensor conversion, and model inference.

    Core Model: A custom Convolutional Neural Network (CNN) implemented in PyTorch, optimized for binary classification (Defect vs. No Defect).

### Performance & Analysis

The model was trained with an early stopping mechanism to prevent overfitting. Training was halted automatically when validation loss ceased to improve.
 Training Results

    Test Set Accuracy: 98.95%

    Best Validation Loss: 0.0284 (Achieved at Epoch 4)

    Stopping Condition: Early stopping triggered at Epoch 6.

### Visual Analysis

1. Loss Curve (Left) The training loss demonstrates a consistent downward trend, indicating effective learning. The validation loss decreases sharply until Epoch 4 (~0.028), which represents the model's peak generalization point. Subsequent epochs show a plateau in validation loss, justifying the decision to trigger early stopping at Epoch 6 to maintain model robustness.

2. Accuracy Curve (Right) Both training and validation accuracy converge rapidly, exceeding 98% by Epoch 2. The close proximity of the training and validation curves suggests low variance and a highly stable model.
Usage
1. Installation

Install the required dependencies:
Bash

pip install -r requirements.txt

2. Execution

The system requires both the backend and frontend services to be active. Run the following commands in separate terminals:

Terminal 1 (Backend Service)
Bash

python api.py
##### The API will initialize at [http://0.0.0.0:8000](http://0.0.0.0:8000)

Terminal 2 (Frontend Interface)
Bash

streamlit run app.py
##### Access the interface via the URL provided (typically http://localhost:8501)

Technology Stack

    Language: Python 3.10+

    Deep Learning: PyTorch, Torchvision

    Web Frameworks: FastAPI, Streamlit

    Data Visualization: Matplotlib
