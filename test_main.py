from typing import Union
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
# from predict import predict_sign
from fastapi.responses import JSONResponse
import numpy as np
from predict_new import predict_sign

app = FastAPI()


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         # img = Image.open(BytesIO(contents)).convert('L')
#         return predict_sign(contents)
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