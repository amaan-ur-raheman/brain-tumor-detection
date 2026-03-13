import io
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def build_model():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2), nn.Dropout2d(0.25),
        nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2), nn.Dropout2d(0.25),
        nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.Dropout2d(0.25),
        nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(256), nn.MaxPool2d(2), nn.Dropout2d(0.25),
        nn.Flatten(),
        nn.Linear(256 * 14 * 14, 128), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 4)
    )


CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
device = torch.device('cpu')  

model = build_model()
model.load_state_dict(torch.load('brain_tumor_model.pth', map_location=device, weights_only=True))
model.eval()
print('Model loaded successfully')


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception:
        return jsonify({'error': 'Invalid image file'}), 400

    tensor = transform(img).unsqueeze(0)   # [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    results = [
        {'class': CLASS_NAMES[i], 'confidence': round(probs[i].item() * 100, 2)}
        for i in range(len(CLASS_NAMES))
    ]
    results.sort(key=lambda x: x['confidence'], reverse=True)

    return jsonify({
        'prediction': results[0]['class'],
        'confidence': results[0]['confidence'],
        'all': results
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)