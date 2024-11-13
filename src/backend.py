from pydantic import BaseModel
from typing import List, Annotated
from fastapi import FastAPI, File, UploadFile


# Define Pydantic models for data validation
class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class Keypoints(BaseModel):
    x: float
    y: float
    confidence: float

class Segmentation(BaseModel):
    points: List[float]  # List of coordinates representing a polygon (x1, y1, x2, y2, ..., xn, yn)

class YOLOAnnotation(BaseModel):
    image_url: str
    bounding_boxes: List[BoundingBox]
    keypoints: List[Keypoints] = []
    segmentations: List[Segmentation] = []

app = FastAPI()

# Sample YOLO annotations (this could be dynamically loaded from files or a database)
annotations_data = {
    "image_url": "https://example.com/image.jpg",
    "bounding_boxes": [
        BoundingBox(x=50, y=50, width=200, height=100),
        BoundingBox(x=300, y=150, width=100, height=200)
    ],
    "keypoints": [
        Keypoints(x=100, y=120, confidence=0.95),
        Keypoints(x=350, y=170, confidence=0.80)
    ],
    "segmentations": [
        Segmentation(points=[50, 50, 250, 50, 250, 150, 50, 150])  # Example polygon
    ]
}

@app.get("/annotations", response_model=YOLOAnnotation)
async def get_annotations():
    return YOLOAnnotation(
        image_url="https://example.com/image.jpg",
        bounding_boxes=[
            BoundingBox(x=50, y=50, width=200, height=100),
            BoundingBox(x=300, y=150, width=100, height=200)

        ],
        keypoints=[
            Keypoints(x=100, y=120, confidence=0.95),
            Keypoints(x=350, y=170, confidence=0.80)
        ],
        segmentations=[
            Segmentation(points=[50, 50, 250, 50, 250, 150, 50, 150])  # Example polygon
        ])
    

@app.post("/files/")
async def create_file(file: UploadFile):
    return {"file_size": file.filename}