# Industrial Defect Detection System

This project implements an automated visual inspection tool leveraging Deep Learning to detect manufacturing defects. The system utilizes a Convolutional Neural Network (CNN) to classify images with **98.95% accuracy**, demonstrating the application of computer vision in quality control automation.

## Project Structure

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

