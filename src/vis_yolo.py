from PIL import Image, ImageDraw

def annotate_image(image_path, annotations):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for annotation in annotations:
        # Draw bounding box
        x_center, y_center, width, height = annotation['bbox']
        x_min = int((x_center - width / 2) * image.width)
        y_min = int((y_center - height / 2) * image.height)
        x_max = int((x_center + width / 2) * image.width)
        y_max = int((y_center + height / 2) * image.height)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="cyan", width=2)
        
        # Draw segmentation
        if 'segmentation' in annotation:
            points = [(x * image.width, y * image.height) for x, y in annotation['segmentation']]
            draw.polygon(points, outline="green")
        
        # Draw keypoints
        if 'keypoints' in annotation:
            for (x_kp, y_kp) in annotation['keypoints']:
                draw.ellipse(
                    [(x_kp * image.width - 3, y_kp * image.height - 3),
                     (x_kp * image.width + 3, y_kp * image.height + 3)],
                    fill="red"
                )
    
    annotated_image_path = "temp_annotated_image.png"
    image.save(annotated_image_path)
    return annotated_image_path
