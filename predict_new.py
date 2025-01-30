import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Extract hand landmarks using MediaPipe
def extract_hand_landmarks_from_bytes(img_data):
    landmarks = []
    
    image = Image.open(BytesIO(img_data))
    image = np.array(image)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
    return np.array([landmarks]) if landmarks else None

# Label map for predictions
def get_predicted_sign_letter(class_index):
    sign_dict = {
        0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "A",
        11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J", 20: "K",
        21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T", 30: "U",
        31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z", 36: "_"
    }

    # sign_dict = {
    #     0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
    #     11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
    #     21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4',
    #     31: '5', 32: '6', 33: '7', 34: '8', 35: '9', 36: '_'
    # }
    return sign_dict.get(class_index, "Unknown")

# Predict sign from image data
def predict_sign(img_data):
    model = load_model("ASL_TO_TEXT_MP4_V2.h5", compile=False)
    
    landmarks = extract_hand_landmarks_from_bytes(img_data)
    if landmarks is None:
        return {
            "status": 404,
            "error": "No hand detected in the image !"
        }

    # Predict
    prediction = model.predict(landmarks)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100
    predicted_letter = get_predicted_sign_letter(predicted_class)

    print(f"Class: {predicted_class}")
    print(f"Letter: {predicted_letter}")
    print(f"Confidence: {confidence:.2f}%")

    return {
        "status": 200,
        "class": int(predicted_class),
        "letter": predicted_letter,
        "confidence": f"{confidence:.2f}%"
    }