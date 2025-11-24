import cv2
import numpy as np
import supervision as sv
import json
import torch
import torchvision
import random
from tqdm import tqdm
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import os
import os.path as osp
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omnitry_bench.Grounded_Segment_Anything.GroundingDINO.groundingdino.util.inference import Model


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = '../omnitry_bench/Grounded_Segment_Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
GROUNDING_DINO_CHECKPOINT_PATH = '../checkpoints/groundingdino_swint_ogc.pth'

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "../checkpoints/sam_vit_h_4b8939.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# Predict classes and hyper-param for GroundingDINO
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8


def generate_mask(path, prompt):
    CLASSES = [prompt]

    # load image
    image = Image.open(path).convert('RGB')
    image = np.array(image)[:, :, ::-1]

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # NMS post process
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    if prompt.startswith('shoe') or prompt.startswith('earrings'):
        topk = 2
    else:
        topk = 1

    detections.xyxy = detections.xyxy[:topk]
    detections.confidence = detections.confidence[:topk]
    detections.class_id = detections.class_id[:topk]

    if detections.confidence[0] < 0.5:
        return None

    # Prompting SAM with detected boxes
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)


    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _, _ 
        in detections]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    
    # output mask
    mask = torch.Tensor(np.any(detections.mask, axis=0, keepdims=True))
    return mask



if __name__ == '__main__':

    input_index_file = 'example_list_objects.json'
    output_index_file = 'example_ground_objects.json'
    
    data = json.load(open(input_index_file))
    outs = []
    for index in tqdm(data):
        
        new_objects = []
        for garment_description in index['objects']:
            tryon_path = index['image_path']
            mask = generate_mask(tryon_path, garment_description)
            if mask is None:
                continue
            mask_path = '.'.join(tryon_path.split('.')[:-1]) + '_{}_mask.jpg'.format('_'.join(garment_description.split(' ')))
            torchvision.utils.save_image(mask, mask_path)
            new_objects.append({
                'description': garment_description,
                'mask': mask_path
            })
        
        index['objects'] = new_objects
        outs.append(index)
    
    # save
    with open(output_index_file, 'w+') as f:
        f.write(json.dumps(outs, indent=4, ensure_ascii=False))