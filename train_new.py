import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os
import glob

#Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#Extract hand landmarks using MediaPipe
def extract_hand_landmarks(image_path):
    landmarks = []
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks if landmarks else None

#Generate Label Map Automatically
def generate_label_map(dataset_path):
    folders = sorted(os.listdir(dataset_path)) 
    label_map = {i: folder for i, folder in enumerate(folders)}
    return label_map

#Preprocess Dataset
def preprocess_dataset(dataset_path, label_map):
    X, y = [], []
    for label, gesture in label_map.items():
        gesture_path = os.path.join(dataset_path, gesture)
        for image_path in glob.glob(f"{gesture_path}/*.jpg"):
            landmarks = extract_hand_landmarks(image_path)
            if landmarks:
                X.append(landmarks)
                y.append(label)  
    return np.array(X), np.array(y)

#dataset path
dataset_path = "gdrive/MyDrive/Colab Notebooks/datasets/Sign_Language_Gesture_Images_Dataset/Gesture_Image_Data"  # Replace with your dataset path

# Generate label map dynamically
label_map = generate_label_map(dataset_path)
print("Generated Label Map:")
for label, folder in label_map.items():
    print(f"{label}: {folder}")

# Preprocess dataset
X, y = preprocess_dataset(dataset_path, label_map)

# One-hot encode the labels
y = to_categorical(y, num_classes=len(label_map))

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classification Model
def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = (X_train.shape[1],)
num_classes = len(label_map)
model = create_model(input_shape, num_classes)

# Train the Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the Model
model.save("ASL_TO_TEXT_MP4_V2.h5")

import matplotlib.pyplot as plt

# Plot Training and Validation Accuracy and Loss
def plot_history(history):
    # Accuracy plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Loss plot
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_history(history)

# Prediction on New Images
def predict_hand_gesture(image_path):
    landmarks = extract_hand_landmarks(image_path)
    if landmarks:
        prediction = model.predict(np.array([landmarks]))
        predicted_class = np.argmax(prediction)
        gesture = label_map[predicted_class]
        confidence = np.max(prediction)
        return confidence, gesture
    else:
        return "No hand detected"
    

confidence, gesture = predict_hand_gesture("e.jpeg")
print("Predicted Gesture:", gesture)

if confidence is not None:
    print(f"Confidence Level: {confidence * 100:.2f}%")
else:
    print("No hand detected, unable to determine confidence level.")