import cv2
import os
import numpy as np
import supervision as sv
import json
import torch
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from Grounded_Segment_Anything.GroundingDINO.groundingdino.util.inference import Model


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = 'Grounded_Segment_Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
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

## the OmniTryBench index file
benchmark_file = '../OmniTry_Bench/omni_vtryon_benchmark_small_v1.json'
## the Try-on result direction
result_dir = "../evaluation_results/"

to_tensor = T.ToTensor()

# Predict classes and hyper-param for GroundingDINO
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8


def generate_mask(img_path, prompt, box_thredshold=BOX_THRESHOLD):
    CLASSES = [prompt]

    # load image
    # image = cv2.imread(SOURCE_IMAGE_PATH)
    
    image = Image.open(img_path).convert('RGB')
    image = np.array(image)[:, :, ::-1]

    # import pdb; pdb.set_trace()
    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=box_thredshold,
        text_threshold=TEXT_THRESHOLD,
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

    if 'earrings' in prompt:
        topk = 2
    else:
        topk = 1

    detections.xyxy = detections.xyxy[:topk]
    detections.confidence = detections.confidence[:topk]
    detections.class_id = detections.class_id[:topk]

    # Prompting SAM with detected boxes
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            ## select the max score
            index = np.argmax(scores)
            result_masks.append(masks[index]) ## post_process_mask(masks[index])
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
    # import pdb; pdb.set_trace()
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)#, labels=labels)

    # output mask
    mask = torch.Tensor(np.any(detections.mask, axis=0, keepdims=True))
    return mask


with open(benchmark_file) as f:
    test_data = json.load(f)
    f.close()
print("th all processing data: {}".format(len(test_data)))

outs = []

all_do = 0
mask_none_num = 0
mask_01_num = 0
for index in tqdm(test_data):
    id = index['id']
    model_path = index['person']['img_path']
    garment_path = index['object']['img_path']
    garment_desc = index['garment_class']
    class_name = index['class_name']
    garment_class = index['garment_class']

    try:
        ## object mask of the object image
        garment_mask_path = '.'.join(garment_path.split('.')[:-1]) + '_mask.jpg'
        if not os.path.exists(garment_mask_path):
            mask = generate_mask(garment_path, garment_description)
            torchvision.utils.save_image(mask, garment_mask_path)
        index['garment']['mask'] = garment_mask_path

        garment_description = index['garment_class']

        tryon_path = os.path.join(result_dir, f'{class_name}_{id}.jpg')
        mask_path = '.'.join(tryon_path.split('.')[:-1]) + '_mask.jpg'

        ## object mask of the try-on gt image
        if not os.path.exists(mask_path):
            mask = generate_mask(tryon_path, garment_description)
            if mask.eq(0).all(): 
                try:
                    mask = generate_mask(tryon_path, garment_description, box_thredshold=0.1)
                    mask_path_01 = '.'.join(tryon_path.split('.')[:-1]) + '_mask_box0.1.jpg'
                    torchvision.utils.save_image(mask, mask_path_01)
                    mask_01_num += 1
                except:
                    print("set box_thredshold=0.1, still get none mask")
                    mask = torch.tensor([0.])

                if mask.eq(0).all(): 
                    image = Image.open(tryon_path).convert('RGB')
                    image = to_tensor(image)
                    mask = torch.zeros_like(image)
                    mask_none_num += 1
                    print("mask is none")
                    print(f'{class_name}_{id}')
                    print(model_path)
                    print(garment_path)
                    print(tryon_path)
            
            torchvision.utils.save_image(mask, mask_path)
        
        all_do += 1
        outs.append(index)

    except Exception as e:
        print(e)
        print("{} {} processed error!".format(id, class_name))
        print(model_path)
        print(garment_path)
        print(tryon_path)
        pass

print("{} images get none mask".format(mask_none_num))
print("{} images get none mask with threhold=0.1".format(mask_01_num))
print("Total processed number: {}".format(all_do))
