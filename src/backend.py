from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, UploadFile, HTTPException
from starlette.responses import StreamingResponse
from .inference import Model, plot_result
import cv2
import io
from PIL import Image
from io import BytesIO

model = Model()
# Define Pydantic models for data validation
class Category(BaseModel):
    id: int
    name: str

class Segmentation(BaseModel):
    category: Category
    confidence: float
    segments: List[List[int]]  

class ModelResult(BaseModel):
    width: int
    height: int
    segmentations: List[Segmentation]


app = FastAPI()


@app.post("/predict/json", response_model=ModelResult)
async def predict_json(file: UploadFile):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG images are allowed.")
    image_data = await file.read()
    image = Image.open(BytesIO(image_data)).convert("RGB")
    result, width, height = model.predict(image)
    ret = ModelResult(
        width = int(width),
        height = int(height),
        segmentations = [])
    for r in result:
        c = r['category']
        seg = r['segments']
        x, y = tuple([list(n) for n in zip(*seg)])
        x = [int(i * width) for i in x]
        y = [int(i * height) for i in y]
        ret.segmentations.append(Segmentation(
            category = Category(id=c['id'], name=c['name']),
            confidence = r['confidence'],
            segments = list(map(list, zip(x,y)))
        ))
    return ret

    

@app.post("/predict/file",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Return the same image with predicted segmentations."
        },
        400: {
            "description": "Invalid file type. Only image files are allowed."
        }        
    }    
)
async def predict_file(file: UploadFile):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG images are allowed.")
    image_data = await file.read()
    image = Image.open(BytesIO(image_data)).convert("RGB")
    result, w, h = model.predict(image)
    cv2img = plot_result(image, result)
    res, im_png = cv2.imencode(".png", cv2img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type='image/png')
