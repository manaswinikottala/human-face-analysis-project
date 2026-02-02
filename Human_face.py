import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2


# Page Config (MUST be first Streamlit command)

st.set_page_config(
    page_title="Face Detection",
    layout="centered"
)

st.title("Welcome ")


# Background color

st.markdown("""
<style>
.stApp {
    background-color: #ffe6f0;
}
h1, h2, h3 {
    color: #cc0066;
}
</style>
""", unsafe_allow_html=True)


# Load model

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Human_face_analysisCNN.h5")

model = load_model()

IMG_SIZE = (224, 224)


# Face detector

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# Image preprocessing

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# UI

st.write("Choose **Upload Image** or **Camera Capture**")

option = st.radio("Select Input Method", ["Upload Image", "Camera"])

image = None


# Upload image

if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "📂 Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")


# Camera input

elif option == "Camera":
    camera_image = st.camera_input("📸 Take a photo")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")


# Prediction

if image is not None:
    st.image(image, caption="Selected Image", width="stretch")

    if st.button("🔍 Predict"):
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        # 🐶 Animal / Object
        if len(faces) == 0:
            st.warning("❓ Other / No Face Detected (Animal or Object)")

        # 🧑 Face detected
        else:
            img = preprocess_image(image)
            pred = model.predict(img)[0][0]

            if 0.4 < pred < 0.6:
                st.warning("❓ Uncertain Face (Other)")

            elif pred >= 0.6:
                st.success(f"👤 Human Detected ({pred*100:.2f}%)")

            else:
                st.error(f"🤖 AI Generated Face ({(1-pred)*100:.2f}%)")

            st.progress(float(pred))
            st.caption(f"Confidence score: {pred:.3f}")
