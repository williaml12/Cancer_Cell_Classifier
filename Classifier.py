import os
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models


# Function for loading the ResNet34 model with error handling
@st.cache_resource
def load_model():
    model_path = 'cancer_cell_classifier.pth'

    # Check if the file exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None

    try:
        # Load the pre-trained ResNet34 model and modify the final layer
        model = models.resnet34()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # Adjust for binary classification (cancer vs. non-cancer)

        # Load the saved model state_dict
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Set the model to evaluation mode

        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


with st.spinner('Model is being loaded..'):
    model = load_model()

if model is None:
    st.stop()  # Stop execution if the model could not be loaded

st.markdown('<p class="uploader-text">Upload an image (TIF/JPG/PNG)</p>', unsafe_allow_html=True)

# Modify the file_uploader to accept TIF images as well
file = st.file_uploader("", type=["jpg", "png", "tif", "tiff"])


# Function to preprocess the image and make predictions using the PyTorch model
def import_and_predict(image_data, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = ImageOps.fit(image_data, (224, 224), Image.BILINEAR)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()


if file is None:
    st.text("Please upload an image.")
else:
    # Load the image based on its file type
    image = Image.open(file)

    # Convert TIF image to RGB if needed (streamlit will handle display for jpg/png)
    if file.type in ["image/tiff", "image/tif"]:
        image = image.convert("RGB")

    # Display the image in the Streamlit app
    st.image(image, width=300)

    # Make predictions
    prediction = import_and_predict(image, model)

    class_names = ['Cancer cell', 'Non-cancer cell']

    # Display the detected class
    detected_class = class_names[prediction]
    st.markdown(f'<p class="custom-text">The detected class is: {detected_class}</p>', unsafe_allow_html=True)
