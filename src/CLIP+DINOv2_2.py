#!/usr/bin/env python3

import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image
from groundingdino.util.inference import load_model, predict
from groundingdino.util import box_ops



BASE_DIR = Path(__file__).resolve().parent

TEXT_PROMPT = "a milk . bowl . cereal . cellphone . apple"
CONFIG_PATH = BASE_DIR / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
MODEL_PATH = BASE_DIR / "GroundingDINO" / "weights" / "groundingdino_swint_ogc.pth"


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


print("[INFO] Cargando modelo...")
model = load_model(str(CONFIG_PATH), str(MODEL_PATH))
model.eval()
print("[INFO] Modelo cargado.")


cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    raise Exception("No se pudo acceder a la cámara")

print("Presiona 'q' para salir. Presiona 'd' para detectar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(frame_rgb)
    image_tensor = transform(image_pil)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        print("Realizando detección...")
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=TEXT_PROMPT,
            box_threshold=0.3,
            text_threshold=0.25
        )

        h, w, _ = frame.shape
        for box, phrase in zip(boxes, phrases):
            x1, y1, x2, y2 = box_ops.box_cxcywh_to_xyxy(box.unsqueeze(0))[0]
            x1 = int(x1.item() * w)
            y1 = int(y1.item() * h)
            x2 = int(x2.item() * w)
            y2 = int(y2.item() * h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, phrase, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Grounding DINO - Webcam", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
