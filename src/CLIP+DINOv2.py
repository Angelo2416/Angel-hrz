#!/usr/bin/env python3

import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from groundingdino.util.inference import load_model, predict
from groundingdino.util import box_ops



TEXT_PROMPT = " a milk . bowl . cereal box . bus . people . cocacola can . botlle water . lipton . mustard"
IMAGE_PATH = "/home/angel/Downloads/drinks.jpg"

config_path = "/home/angel/ANGEL/Angel_YOLO/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
model_path = "/home/angel/ANGEL/Angel_YOLO/GroundingDINO/weights/groundingdino_swint_ogc.pth"


model = load_model(config_path, model_path)
model.eval()


image_pil = Image.open(IMAGE_PATH).convert("RGB")

transform = transforms.Compose([
    transforms.ToTensor(),  # (C, H, W)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image_pil)  

boxes, logits, phrases = predict(
    model=model,
    image=image_tensor,
    caption=TEXT_PROMPT,
    box_threshold=0.3,
    text_threshold=0.25
)

image_cv = cv2.imread(IMAGE_PATH)
if image_cv is None:
    raise FileNotFoundError(f"cv2 no pudo cargar : {IMAGE_PATH}")
h, w, _ = image_cv.shape

for box, phrase in zip(boxes, phrases):
    x1, y1, x2, y2 = box_ops.box_cxcywh_to_xyxy(box.unsqueeze(0))[0]
    x1 = int(x1 * w)
    y1 = int(y1 * h)
    x2 = int(x2 * w)
    y2 = int(y2 * h)
    
    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image_cv, phrase, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


cv2.imshow("/home/angel/Downloads/imagen.jpg", image_cv)
cv2.waitKey(0)
cv2.imshow("Detección Grounding DINO", image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()

