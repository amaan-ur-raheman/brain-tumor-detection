# 🧠 Brain Tumor Detection

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-EE4C2C.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **AI-powered brain tumor classification from MRI scans using deep learning**

A production-ready deep learning system that classifies brain MRI images into four categories: **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**. Built with PyTorch and deployed as a REST API using Flask.

---

## ✨ Features

- 🎯 **91.81% Test Accuracy** on balanced 4-class classification
- 🚀 **REST API** for seamless integration
- 🖼️ **Real-time Inference** on uploaded MRI scans
- 📊 **Confidence Scores** for all predictions
- 🌐 **CORS Enabled** for cross-origin requests
- 🔒 **Safe Model Loading** with `weights_only=True`
- ⚡ **GPU Acceleration** support (CUDA/MPS)

---

## 🏗️ Architecture

### Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CNN Architecture                        │
├─────────────────────────────────────────────────────────────┤
│  Input: 224 × 224 × 3 (RGB Image)                           │
│                                                             │
│  Conv Block 1:  3 → 32  channels  + BatchNorm + ReLU        │
│  Conv Block 2: 32 → 64  channels  + BatchNorm + ReLU        │
│  Conv Block 3: 64 → 128 channels  + BatchNorm + ReLU        │
│  Conv Block 4: 128 → 256 channels  + BatchNorm + ReLU       │
│                                                             │
│  Flatten → FC(256×14×14 → 128) → Dropout(0.1)               │
│              → FC(128 → 4)                                  │
│                                                             │
│  Output: 4 classes [glioma, meningioma, no_tumor, pituitary]│
└─────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component            | Technology          |
| -------------------- | ------------------- |
| **Deep Learning**    | PyTorch 2.10+       |
| **Backend**          | Flask 3.1+          |
| **Image Processing** | Pillow, torchvision |
| **CORS**             | flask-cors          |
| **Training**         | Jupyter Lab         |

---

## 📊 Dataset

The model is trained on a balanced dataset of brain MRI images:

| Category      | Training Images | Testing Images | Total     |
| ------------- | --------------- | -------------- | --------- |
| 🟠 Glioma     | 1,400           | 400            | 1,800     |
| 🟡 Meningioma | 1,400           | 400            | 1,800     |
| 🟢 No Tumor   | 1,400           | 400            | 1,800     |
| 🟣 Pituitary  | 1,400           | 400            | 1,800     |
| **Total**     | **5,600**       | **1,600**      | **7,200** |

### Data Preprocessing

- **Resize**: 224 × 224 pixels
- **Normalization**: ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- **Augmentation** (Training only):
    - Random horizontal/vertical flip (p=0.5)
    - Random rotation (±20°)
    - Random resized crop (scale 0.8-1.0)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- uv (recommended) or pip

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/amaan-ur-raheman/brain-tumor-detection.git
cd brain-tumor-detection
```

2. **Create virtual environment**

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using Python
python -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**

```bash
uv sync
# Or: pip install -e .
```

4. **Download the pre-trained model**

Ensure `brain_tumor_model.pth` is in the project root, or train your own using `main.ipynb`.

---

## 🖥️ Usage

### Running the API Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### API Endpoints

#### `POST /predict`

Upload an MRI image for classification.

**Request:**

```bash
curl -X POST -F "file=@/path/to/mri_scan.jpg" http://localhost:5000/predict
```

**Response:**

```json
{
    "prediction": "glioma",
    "confidence": 94.23,
    "all": [
        { "class": "glioma", "confidence": 94.23 },
        { "class": "meningioma", "confidence": 3.12 },
        { "class": "pituitary", "confidence": 1.98 },
        { "class": "no_tumor", "confidence": 0.67 }
    ]
}
```

#### `GET /`

Serves the static HTML frontend (if `static/index.html` exists).

### Supported Image Formats

- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)

---

## 🧪 Training

To train the model from scratch:

1. **Prepare your data**
    - Organize images in `data/Training/` and `data/Testing/`
    - Structure: `data/{split}/{class}/{image}.jpg`

2. **Open the training notebook**

```bash
jupyter lab main.ipynb
```

3. **Run all cells**

The notebook will:

- Load and preprocess data
- Train the CNN for 40 epochs
- Evaluate on test set
- Save the model as `brain_tumor_model.pth`

### Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Epochs        | 40               |
| Batch Size    | 32               |
| Optimizer     | Adam (lr=1e-4)   |
| Loss Function | CrossEntropyLoss |
| Device        | CUDA/MPS/CPU     |

### Training Results

```
Epoch 40 Loss: 28.6584
Test Loss: 0.685
Test Accuracy: 91.81%
```

---

## 📁 Project Structure

```
brain-tumor-detection/
├── 📄 app.py                      # Flask REST API
├── 📓 main.ipynb                  # Training notebook
├── 📋 pyproject.toml              # Project dependencies
├── 📖 README.md                   # This file
├── 🧠 brain_tumor_model.pth       # Pre-trained weights
├── 🧠 brain_tumor_model.pkl       # Joblib serialized model
├── 📂 data/
│   ├── 📂 Training/               # Training images (5,600)
│   │   ├── 📂 glioma/
│   │   ├── 📂 meningioma/
│   │   ├── 📂 notumor/
│   │   └── 📂 pituitary/
│   └── 📂 Testing/                # Testing images (1,600)
│       ├── 📂 glioma/
│       ├── 📂 meningioma/
│       ├── 📂 notumor/
│       └── 📂 pituitary/
└── 📂 static/                     # Static files (HTML frontend)
    └── 📄 index.html
```

---

## 🛡️ Security Considerations

⚠️ **For Research Purposes Only**

This application is intended for research and educational purposes. It is **not** a medical device and should not be used for clinical diagnosis. Always consult qualified healthcare professionals for medical decisions.

### Security Features

- ✅ Safe model loading with `weights_only=True`
- ✅ Input validation for uploaded files
- ✅ Error handling for invalid images
- ⚠️ CORS is currently open (`CORS(app)`) - restrict in production
- ⚠️ Debug mode enabled - disable in production

---

## 🐳 Docker Deployment (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 5000
CMD ["python", "app.py"]
```

```bash
docker build -t brain-tumor-detection .
docker run -p 5000:5000 brain-tumor-detection
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset sourced from [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- Built with [PyTorch](https://pytorch.org/) and [Flask](https://flask.palletsprojects.com/)

---

## 📧 Contact

For questions or inquiries, please open an issue on GitHub.

---

<div align="center">

**Made with ❤️ and PyTorch**

</div>
