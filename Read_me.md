# 🧑 Human Face Analysis using Deep Learning

This project detects whether an image contains a **Human Face**, an **AI‑Generated Face**, or **No Face / Other Object** using a **CNN (Convolutional Neural Network)** and **OpenCV Haar Cascade**. The application is deployed using **Streamlit** for an interactive user experience.

---

## 📂 Project Structure

```
├── Human_face.py                 # Streamlit web application
├── Human face analysis.ipynb     # Model training & experimentation
├── Human_face_analysisCNN.h5     # Trained CNN model
├── README.md                     # Project documentation
```

---

## 🎯 Project Objective

The goal of this project is to:

* Detect whether a face is present in an image
* Classify the face as:

  * 👤 Human Face
  * 🤖 AI‑Generated Face
  * ❓ Other / No Face (animal or object)

This is useful in applications such as:

* AI‑generated content detection
* Image authentication
* Face verification pipelines

---

## 🧠 Technologies Used

* **Python**
* **TensorFlow / Keras** – CNN model
* **OpenCV** – Face detection
* **Streamlit** – Web UI
* **NumPy** – Numerical operations
* **Pillow (PIL)** – Image handling

---

## 📊 Model Overview

* Model type: **Convolutional Neural Network (CNN)**
* Input image size: **224 × 224**
* Output:

  * Probability score between 0 and 1

### Prediction Logic

| Prediction Score | Result               |
| ---------------- | -------------------- |
| ≥ 0.6            | 👤 Human Face        |
| 0.4 – 0.6        | ❓ Uncertain          |
| < 0.4            | 🤖 AI‑Generated Face |

---

## 🧹 Face Detection (Pre‑Processing)

Before classification, the image goes through:

1. **Face Detection** using Haar Cascade (`haarcascade_frontalface_default.xml`)
2. If no face is detected → classified as **Other / Object**
3. If a face is detected → image is passed to the CNN model

---

## 🖼 Image Input Methods

The application supports two input modes:

### 1️⃣ Upload Image

* Accepts `.jpg`, `.jpeg`, `.png`

### 2️⃣ Camera Capture

* Takes a live photo using webcam

---

## ⚙️ Image Pre‑Processing Steps

```python
image.resize((224,224))
image / 255.0
expand dimensions
```

This ensures compatibility with the trained CNN model.

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install streamlit tensorflow opencv-python pillow numpy
```

### 2️⃣ Run the Application

```bash
streamlit run Human_face.py
```

### 3️⃣ Open Browser

The app will open automatically or visit:

```
http://localhost:8501
```

---

## 📓 Notebook Explanation (`Human face analysis.ipynb`)

This notebook contains:

* Dataset loading
* CNN model architecture
* Training and validation
* Accuracy and loss plots
* Model saving (`.h5`)

The trained model is later used in the Streamlit app.

---

## 📌 Key Features

* Real‑time prediction
* Camera & image upload support
* Confidence score display
* Face verification before classification
* User‑friendly UI

---

## ⚠️ Limitations

* Works best with clear frontal faces
* Performance depends on training data quality
* Haar Cascade may fail for extreme angles

---

## 📈 Future Improvements

* Use MTCNN or RetinaFace for detection
* Multi‑class classification
* Deploy on cloud (AWS / HuggingFace Spaces)
* Improve model accuracy with larger dataset

---

## 👩‍💻 Author

**Manaswini**
B.Tech Student | AI & ML Enthusiast

---

⭐ If you like this project, don’t forget to star the repository!
