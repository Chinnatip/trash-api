import torch
import base64
# from PIL import Image 
from helpers import draw_box, url_to_img, img_to_bytes
from boto.s3.connection import S3Connection
from boto.s3.key import Key

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
        box_img = draw_box(img, boxes)
        lists = []
        for box in boxes:
            lists.append(int(box[-1]))

        dictionary = {}
        for item in lists:
            dictionary[item] = dictionary.get(item, 0) + 1

        # upload image to s3
        S3_BUCKET_NAME = 'koh-assets'
        AWS_ACCESS_KEY = 'AKIA5ATH3XN45KU2YYMB'
        AWS_SECRET_KEY = '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'

        picture = "data:image/jpg;base64,"+img_to_bytes(box_img)
        conn = S3Connection(AWS_ACCESS_KEY, AWS_SECRET_KEY)
        bucket = conn.get_bucket(S3_BUCKET_NAME)
        k = Key(bucket)
        image_id = 'yyyy'
        image_key = '/gather/ariaround/store/'+ image_id + '.jpg'
        k.key = image_key
        picture = picture.replace("data:image/jpg;base64,","")
        k.set_contents_from_string(base64.b64decode(picture))
        k.set_metadata('Content-Type', 'image/jpg')
        k.set_acl('public-read')

        image_path = 'https://koh-assets.s3.ap-southeast-1.amazonaws.com' + image_key

        return {
            "trash_amount": len(boxes),
            "annotate": dictionary,
            "image": image_path
        }