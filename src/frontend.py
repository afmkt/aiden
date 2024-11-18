import flet as ft
from fastapi import HTTPException
import requests
from typing import List
from flet import Container, Image, Column, Row, Text
from backend import ModelResult

def fetch_annotations() -> YOLOAnnotation:
    """Fetch YOLO annotations from the FastAPI server"""
    try:
        response = requests.get("http://127.0.0.1:8000/predict/json")
        response.raise_for_status()
        data = response.json()
        # Parse bounding boxes and keypoints
        return ModelResult(**data)
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail="Error fetching annotations")

def main(page: ft.Page):
    """Main function to create the Flet app"""
    page.title = "YOLO Annotation Viewer"
    
    # Fetch annotation data from FastAPI backend
    annotations = fetch_annotations()
    
    # Create image component with overlay for bounding boxes and keypoints
    img = Image(src=annotations.image_url, width=500, height=400)
    bounding_boxes = [
        Container(
            width=box.width,
            height=box.height,
            top=box.y,
            left=box.x,
            border=ft.border.all(2, ft.colors.RED),
            opacity=0.5,
        )
        for box in annotations.bounding_boxes
    ]
    keypoints = [
        Container(
            width=10,
            height=10,
            top=point.y - 5,  # Adjust for center
            left=point.x - 5,  # Adjust for center
            border=ft.border.all(1, ft.colors.GREEN),
            color=ft.colors.GREEN,
        )
        for point in annotations.keypoints
    ]
    
    # Customizable Overlay for interactivity
    overlay = Column(bounding_boxes + keypoints)

    page.add(
        Row(
            [
                img,
                overlay,  # Overlay with interactive bounding boxes and keypoints
            ]
        )
    )

# Run the Flet app
ft.app(target=main)