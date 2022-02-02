import torch
import json
from PIL import Image 
from helpers import draw_box, url_to_img, img_to_bytes

# custom yolo weight instead of torch hub
# https://github.com/ultralytics/yolov5/issues/1605

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class PythonPredictor:

    def __init__(self, config):
        """ Download pretrained model. """
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False, classes=80)
        # checkpoint_ = torch.load('./weights/yolov5s.pt')['model']
        # model.load_state_dict(checkpoint_.state_dict())
        # copy_attr(model, checkpoint_, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())
        # self.model = model.fuse().autoshape()

    def predict(self, payload):
        """ Run a model based on url input. """

        # Inference
        img = url_to_img(payload["url"])
        results = self.model(img)

        # Draw boxes
        boxes = results.xyxy[0].numpy()
        # box_img = draw_box(img, boxes)
        lists = []
        for box in boxes:
            lists.append(int(box[-1]))

        dictionary = {}
        for item in lists:
            dictionary[item] = dictionary.get(item, 0) + 1

        return {
            "trash_amount": len(boxes),
            # "annotate": dictionary,
            # "image": img_to_bytes(box_img)
        }