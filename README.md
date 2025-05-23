# ASL to Text MP4 Backend

This project provides a backend service for recognizing American Sign Language (ASL) gestures from images using deep learning and MediaPipe. It exposes a FastAPI endpoint for predicting ASL signs from uploaded images.

## Features

- ASL hand gesture recognition using deep learning (Keras/TensorFlow)
- Hand landmark extraction using MediaPipe
- REST API for image-based sign prediction
- Model training and evaluation scripts
- Visualization of training metrics and confusion matrix

## Project Structure

- `main.py` - FastAPI app exposing the `/predict-sign` endpoint
- `predict_new.py` - Prediction logic using the trained model and MediaPipe
- `train.py` - Model training and evaluation using the MNIST sign language dataset
- `train_new.py` - Model training using hand landmarks extracted from images
- `test_main.py` - Test FastAPI app for prediction
- `ASL_TO_TEXT_MP4_V1.h5`, `ASL_TO_TEXT_MP4_V2.h5` - Trained model files
- `requirements.txt` - Python dependencies

## Setup

1. **Clone the repository** and navigate to the project directory.

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Download or prepare your dataset** as needed for training.

4. **Train the model (optional):**
   - For MNIST-based training:
     ```sh
     python train.py
     ```
   - For hand landmark-based training:
     ```sh
     python train_new.py
     ```

5. **Run the FastAPI server:**
   ```sh
   uvicorn main:app --reload
   ```

6. **Test the API:**
   - Send a POST request to `/predict-sign` with an image file.
   - Example using `curl`:
     ```sh
     curl -X POST "http://127.0.0.1:8000/predict-sign" -F "file=@path_to_image.jpg"
     ```

## API

### `POST /predict-sign`

- **Request:** Multipart form with an image file.
- **Response:** JSON with predicted class, letter, and confidence.

Example response:
```json
{
  "status": 200,
  "class": 3,
  "letter": "C",
  "confidence": "98.45%"
}
```

## Notes

- The model expects images containing a single hand showing an ASL gesture.
- The label map and dataset path can be customized in `train_new.py`.
- For best results, use clear images with a visible hand.

## License

This project is for educational and research purposes.

---

**Files referenced:**
- [main.py](main.py)
- [predict_new.py](predict_new.py)
- [train.py](train.py)
- [train_new.py](train_new.py)
- [test_main.py](test_main.py)
- [requirements.txt](requirements.txt)
