import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

st.set_page_config(
    page_title="NeuroScan",
    page_icon="🧠",
    layout="centered"
)

st.title("NeuroScan")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 16 * 16, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 4)
    ).to(device)

    model.load_state_dict(torch.load("brain_tumor_model.pth", map_location=device))
    model.eval()
    return model

model = load_model()

tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

uploaded_file = st.file_uploader("Upload an MRI scan", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded MRI Scan")

    img_tensor = tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = probs.argmax().item()

    prediction = class_names[pred]

    st.divider()
    st.subheader("Prediction")

    if prediction == "No Tumor":
        st.success(f"**{prediction}**")
    else:
        st.error(f"**{prediction}**")

    st.divider()
    st.subheader("Class Probabilities")

    for i, name in enumerate(class_names):
        prob = probs[i].item()
        st.write(f"{name}: {prob:.1%}")
        st.progress(prob, text="")

st.divider()
st.caption("For research use only. Not a substitute for professional medical diagnosis.")
