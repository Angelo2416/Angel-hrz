#!/usr/bin/env python3

import torch, clip, cv2, gc
from PIL import Image
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

gc.collect()
torch.cuda.empty_cache()
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = clip.load("ViT-B/32", device=device)

# --- Atributos ---
race = ["caucasian", "black", "american", "asian"]
sex = ["man", "woman"]
height = ["tall", "short"]
shirt = ["red", "blue", "green", "yellow", "white", "black", "orange", "pink", "brown", "dark_blue", "dark_geen", "gray", "dark_gray", "purple"]


def predict_attribute(roi_clip, options):
    prompts = [f"The person is {opt}" for opt in options]
    text_tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        image_features = model_clip.encode_image(roi_clip)
        text_features = model_clip.encode_text(text_tokens)
        similarity = (100.0 * image_features @ text_features.T)
        values, indices = similarity.softmax(dim=-1).topk(1)
    return options[indices[0, 0].item()]

image = cv2.imread("/home/angel/Downloads/personas.jpg")
outputs = predictor(image)
instances = outputs["instances"].to("cpu")

persons = instances[instances.pred_classes == 0]
num_person = len(persons.pred_classes)

total = 0

for i in range(num_person):
    mask = persons.pred_masks[i].numpy()       
    bbox = persons.pred_boxes[i].tensor.numpy()[0].astype(int)

    roi = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    roi_masked = roi * mask[bbox[1]:bbox[3], bbox[0]:bbox[2], None]


    roi_pil = Image.fromarray(cv2.cvtColor(roi_masked, cv2.COLOR_BGR2RGB))
    roi_clip = preprocess(roi_pil).unsqueeze(0).to(device)

    # Predecir atributos
    race_pred = predict_attribute(roi_clip, race)
    height_pred = predict_attribute(roi_clip, height)
    shirt_pred = predict_attribute(roi_clip, shirt)
    sex_pred = predict_attribute(roi_clip, sex)

    total += 1
    
    print(f"The peson is:{race_pred}, sex={sex_pred} height={height_pred}, shirt={shirt_pred}")
    print(f"I see {total} people")

v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
out = v.draw_instance_predictions(persons)
cv2.imshow("Resultados", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()


torch.cuda.empty_cache()
torch.cuda.ipc_collect()




