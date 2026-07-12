import streamlit as st
from streamlit_drawable_canvas import st_canvas
from keras.models import load_model
import numpy as np
import cv2
import os

# Set up page layout and design
st.set_page_config(page_title="Digit Recognizer", layout="wide")
st.title("🖌️ Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) inside the black box below and the model will predict it in real-time!")

# Load Keras Model
@st.cache_resource
def load_my_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return load_model(os.path.join(base_dir, "bestmodel.h5"))

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'bestmodel.h5' is in the same directory.")
    st.stop()

# Configure the canvas parameters
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",  # Match black background — prevents white fill inside closed strokes
    stroke_width=8,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=300,
    width=900,
    drawing_mode="freedraw",
    key="canvas",
)

# If canvas has drawing
if canvas_result.image_data is not None:
    img = canvas_result.image_data
    
    # Check if anything has been drawn (non-black pixels)
    if np.any(img[:, :, :3] > 0):
        # Convert RGBA canvas image to grayscale
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
        
        # Crop the drawing to the exact bounding box to match training dataset formatting
        coords = np.argwhere(gray > 0)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Crop with padding
            padding = 15
            h, w = gray.shape
            x_min = max(x_min - padding, 0)
            x_max = min(x_max + padding, w)
            y_min = max(y_min - padding, 0)
            y_max = min(y_max + padding, h)
            
            cropped = gray[y_min:y_max, x_min:x_max]
            
            # Pad cropped region to be SQUARE before resizing
            # This preserves aspect ratio and matches square MNIST digits
            crop_h, crop_w = cropped.shape
            diff = abs(crop_h - crop_w)
            if crop_h > crop_w:
                pad_left = diff // 2
                pad_right = diff - pad_left
                cropped = np.pad(cropped, ((0, 0), (pad_left, pad_right)), 'constant', constant_values=0)
            elif crop_w > crop_h:
                pad_top = diff // 2
                pad_bot = diff - pad_top
                cropped = np.pad(cropped, ((pad_top, pad_bot), (0, 0)), 'constant', constant_values=0)
            
            # Add a small border margin, then resize to 28x28 and normalize
            cropped = np.pad(cropped, 8, 'constant', constant_values=0)
            final_image = cv2.resize(cropped, (28, 28)) / 255.0
            
            # Reshape to Keras batch format (1, 28, 28, 1)
            final_image = final_image.reshape(1, 28, 28, 1)
            
            # Predict
            predictions = model.predict(final_image)
            predicted_class = np.argmax(predictions)
            confidence = predictions[0][predicted_class]
            
            # Display Prediction
            st.markdown(f"<h2 style='text-align: center; color: #FF4B4B;'>Prediction: {predicted_class}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align: center;'>Confidence: {confidence:.2%}</h4>", unsafe_allow_html=True)
            
            # Optional: Show what the processed image looks like to the neural network
            with st.sidebar:
                st.write("### Model's View (28x28)")
                st.image(final_image.reshape(28, 28), width=150, clamp=True)
