from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, UploadFile
from inference import Model

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
    image_url: str
    width: int
    height: int
    segmentations: List[Segmentation]

app = FastAPI()


@app.get("/predict", response_model=ModelResult)
async def predict(file: UploadFile):
    result = 
    return ModelResult(
        image_url="https://example.com/image.jpg",
        width = 10,
        height = 10,
        segmentations = [
            Segmentation(points=[50, 50, 250, 50, 250, 150, 50, 150])
        ])
    

@app.post("/files/")
async def create_file(file: UploadFile):
    return {"file_size": file.filename}