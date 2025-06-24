import streamlit as st
import os
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Animals-10 Classifier",
    page_icon="🐾",
    layout="centered"
)

# Configuration
MODEL_PATH = "model.pth"
DATA_ROOT = "Animals10"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model

# Load model
model = load_model()

# Preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Class names
classes = ["Butterfly", "Cat", "Chicken", "Cow", "Dog",
           "Elephant", "Horse", "Sheep", "Spider", "Squirrel"]

def predict_image(img):
    """Predict the class of an uploaded image"""
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    probabilities = torch.softmax(logits, 1)
    prob, idx = probabilities.max(1)
    
    # Get all class probabilities for display
    all_probs = {classes[i]: float(probabilities[0][i]) for i in range(len(classes))}
    
    return classes[idx.item()], float(prob), all_probs

# Streamlit UI
st.title("🐾 Animals-10 Classifier")
st.write("Upload an image to predict the animal class!")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['jpg', 'jpeg', 'png'],
    help="Upload an image of an animal to classify"
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        # Make prediction
        with st.spinner("Classifying..."):
            predicted_class, confidence, all_probabilities = predict_image(image)
        
        # Display main prediction
        st.success(f"**Prediction: {predicted_class}**")
        st.write(f"**Confidence: {confidence*100:.1f}%**")
        
        # Display confidence bar
        st.progress(confidence)
    
    # Show all class probabilities
    st.subheader("All Class Probabilities")
    
    # Sort probabilities in descending order
    sorted_probs = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, prob in sorted_probs:
        st.write(f"**{class_name}:** {prob*100:.2f}%")
        st.progress(prob)

# Add some information about the model
with st.expander("About this model"):
    st.write("""
    This classifier uses a ResNet-18 model trained on the Animals-10 dataset.
    It can classify images into 10 different animal categories:
    
    - Butterfly
    - Cat  
    - Chicken
    - Cow
    - Dog
    - Elephant
    - Horse
    - Sheep
    - Spider
    - Squirrel
    
    The model was trained using PyTorch and achieves good accuracy on the validation set.
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and PyTorch")