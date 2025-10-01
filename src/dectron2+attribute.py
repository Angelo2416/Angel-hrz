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
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = clip.load("ViT-B/32", device=device)

# --- Atributos generales para preguntar ---
objects = ["apple", "lemon", "can of soda", "cup", "ball", "water bottle", "orange"]
colors = ["red", "blue", "green", "yellow", "white", "black", "orange", "pink"]
materials = ["metal", "wood", "plastic", "glass", "fabric"]
shapes = ["round", "square", "rectangular", "triangular", "irregular"]


image = cv2.imread("/home/angel/Downloads/prueba3.jpeg")
outputs = predictor(image)
instances = outputs["instances"].to("cpu")


for i in range(len(instances)):
    mask = instances.pred_masks[i].numpy()
    bbox = instances.pred_boxes[i].tensor.numpy()[0].astype(int)
    roi = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    roi_masked = roi * mask[bbox[1]:bbox[3], bbox[0]:bbox[2], None]

    # Convertir a PIL y preprocesar para CLIP
    roi_pil = Image.fromarray(cv2.cvtColor(roi_masked, cv2.COLOR_BGR2RGB))
    roi_clip = preprocess(roi_pil).unsqueeze(0).to(device)

    def predict_attribute(options):
        prompts = [f"The object is {objects}" for objects in options]
        text_tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            image_features = model_clip.encode_image(roi_clip)
            text_features = model_clip.encode_text(text_tokens)
            similarity = (100.0 * image_features @ text_features.T)
            values, indices = similarity.softmax(dim=-1).topk(1)
        return options[indices[0,0].item()]

    color_pred = predict_attribute(colors)
    material_pred = predict_attribute(materials)
    shape_pred = predict_attribute(shapes)
    objects_pred = predict_attribute(objects)

    print(f"Objeto {objects_pred}: Color={color_pred}, Material={material_pred}, Forma={shape_pred}")

# --- Visualización BGR a RGB ---
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
out = v.draw_instance_predictions(instances)
cv2.imshow("Resultados", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


