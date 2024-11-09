from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import ObjectDetectionVisualizer
import json


class Visulizer(ObjectDetectionVisualizer):
    def __init__(self, img_path, scale=1.0, instance_mode=0):
        super(Visulizer, self).__init__(img_path, scale, instance_mode)
        pass
        

def predict(img_path, model_path=f"./model"):
    predictor = MultiModalPredictor.load(model_path)
    predictor.set_num_gpus(1)
    predictions = predictor.predict({'image': [img_path]})
    
    visualized = ObjectDetectionVisualizer(img_path).draw_instance_predictions(predictions.iloc[0], conf_threshold=0.1).get_image()

    from IPython.display import Image
    pil_img = Image(filename=img_path)
    # display(pil_img)
    # from PIL import Image
    from PIL import Image
    img = Image.fromarray(visualized, 'RGB')
    # display(img)
    return (pil_img, img, predictions)