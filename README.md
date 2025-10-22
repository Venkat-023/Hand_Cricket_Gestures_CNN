# Hand Gesture Recognition for Cricket Numbers (0–6) using CNN & Real-Time Computer Vision

This project implements a complete deep learning pipeline to classify **cricket-style hand gestures** representing numbers **0 through 6** using a **custom-built CNN model**. The final system performs gesture recognition in **real time** from webcam input using **MediaPipe**, **HSV skin segmentation**, **OpenCV**, and **TensorFlow**.

---

## 📌 Key Features

- ✅ **Trained from scratch** using `TensorFlow` and `Keras` (no transfer learning)
- ✋ **MediaPipe** used to detect and crop only the hand region
- 🎨 **HSV skin segmentation** removes background and isolates the hand
- 🖤 Converted images to **grayscale** to reduce color noise
- ⚖️ Input data normalized and resized to 512×512
- 🧠 Model trained to classify **hand signs for numbers 0–6**
- 📈 Achieved **97% validation accuracy** with **no overfitting**
- 🎥 Integrated with **OpenCV** for real-time webcam predictions

---

## 🗂 Dataset

- **Source**: [Hand Gesture - Cricket](https://www.kaggle.com/datasets/aryanrishilamba/hand-gesture-cricket) by Aryan Rishi Lamba
- **Labels**: Numeric hand gestures from **0 to 6**
- **Preprocessing**:
  - Hand region cropped using MediaPipe Hands
  - Background removed using HSV-based skin masking
  - Converted to grayscale and resized to 512×512
  - Normalized pixel values to range [0, 1]

---

## 🧠 Model Architecture

A custom CNN built from scratch using `tensorflow.keras.models.Sequential`. Includes batch normalization and dropout for stability and regularization.

```python
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(512, 512, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))  # 7 classes (0–6 + extras if any)

Loss Function: categorical_crossentropy

Optimizer: Adam

Metrics: accuracy

📊 Results
Metric	Value
Validation Accuracy	✅ 97%
Overfitting	❌ None
Input Format	512×512 grayscale cropped hand image

🖥️ Real-Time Gesture Recognition
After training, the model was integrated with a live webcam inference system using:

✅ MediaPipe Hands – to detect and crop hand region from webcam frames

✅ HSV skin segmentation – to remove background and isolate hand

✅ Grayscale conversion – to match training format

✅ CNN model inference – to predict digit (0–6)

✅ OpenCV UI – to display prediction on screen in real time

🔄 Real-Time Pipeline Overview
Webcam → MediaPipe Hand Detection → Crop Hand Region
→ HSV Skin Segmentation → Grayscale → Resize (512×512)
→ Normalization → CNN Prediction → Show Output (OpenCV)
💻 How to Use
1️⃣ Clone the Repository
bash
git clone https://github.com/yourusername/hand-gesture-cricket-numbers.git
cd hand-gesture-cricket-numbers
2️⃣ Install Dependencies
bash
pip install -r requirements.txt
3️⃣ Prepare the Dataset
Download the dataset from Kaggle

Use the provided preprocess.py script to:

Crop hands using MediaPipe

Apply skin segmentation (HSV)

Convert to grayscale

Resize to 512×512

Split into train/test folders

4️⃣ Train the Model
bash

python train.py
5️⃣ Run Real-Time Prediction
bash

python realtime_predict.py
Your webcam will open, and the predicted gesture (0–6) will be shown on-screen.

🧾 Requirements
txt
tensorflow
opencv-python
mediapipe
numpy
matplotlib
scikit-learn
📚 References
Hand Gesture Cricket Dataset (Kaggle)

MediaPipe Hands
TensorFlow Documentation

⚠️ Disclaimer
This project is for educational and research purposes only and is not intended for commercial applications.
