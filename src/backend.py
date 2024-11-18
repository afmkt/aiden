from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, UploadFile
from starlette.responses import StreamingResponse
from inference import Model, plot_result
import cv2
import io
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
    result = model.predict(file.filename)
    width = result['width'],
    height = result['height'],
    ret = ModelResult(
        width = result['width'],
        height = result['height'],
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

    

@app.post("/predict/file")
async def predict_file(file: UploadFile):
    result = model.predict(file.filename)
    cv2img = plot_result(file.filename, result)
    res, im_png = cv2.imencode(".png", cv2img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type='image/png')
