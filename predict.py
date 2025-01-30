from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi.responses import JSONResponse

def get_predicted_sign_letter(class_index):
    sign_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
    return sign_dict.get(class_index, "Unknown")

def predict_sign(img_data):
    model = load_model("ASL_TO_TEXT_MP4_V1.h5", compile=False)
    
    # Convert byte data to PIL Image
    img = Image.open(BytesIO(img_data)).convert('L')
    img = img.resize((28, 28))
    img_array = image.img_to_array(img)
    img_array = img_array.reshape((1, 28, 28, 1))
    
    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100
    predicted_letter = get_predicted_sign_letter(predicted_class)

    print(f"Class: {predicted_class}")
    print(f"Letter: {predicted_letter}")
    print(f"Confidence: {confidence:.2f}%")

    return {
        "class": int(predicted_class),
        "letter": predicted_letter,
        "confidence": f"{confidence:.2f}%"
    }