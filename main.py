from typing import Union
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
# from predict import predict_sign as ps
from fastapi.responses import JSONResponse
import numpy as np
from predict_new import predict_sign
import os
app = FastAPI()


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
    
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         # img = Image.open(BytesIO(contents)).convert('L')
#         return ps(contents)
#     except Exception as e:
#         print('e', e)
#         return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/predict-sign")
async def predict_sign_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = predict_sign(contents)
        return JSONResponse(content=result)
    except Exception as e:
        print('e', e)
        return JSONResponse(content={"error": str(e)}, status_code=400)

# @app.post("/predict-sign")
# async def predict_sign_endpoint(file: UploadFile = File(...)):
#     try:
#         # Save the uploaded file to a temporary location
#         # contents = await file.read()
#         print(f"Filename: {file.filename}, Content-Type: {file.content_type}")
#         temp_file_path = f"/tmp/{file.filename}"
#         with open(temp_file_path, "wb") as f:
#             while chunk := await file.read(204800):  # Read in chunks of 200 KB
#                 f.write(chunk)

#         # Pass the file path to the prediction function
#         result = predict_sign(temp_file_path)

#         # Clean up the temporary file
#         os.remove(temp_file_path)

#         return JSONResponse(content=result)

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=400)