import torch
import torchvision
import transformers
import diffusers
import random
import numpy as np
import os
import json
import cv2
import torchvision.transforms as T
from PIL import Image
from peft import LoraConfig
from safetensors import safe_open
from omegaconf import OmegaConf
from slugify import slugify
from tqdm import tqdm
import math
import os.path as osp
from scipy.ndimage import distance_transform_edt
import sys, os; sys.path.append(os.getcwd())
from diffusers import FluxFillPipeline, FluxImg2ImgPipeline


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dilate_mask_process(mask, kernel_size=12, iterations=5):
    mask = mask.numpy()
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    
    dilated_mask = torch.Tensor(dilated_mask)
    return dilated_mask


device = torch.device('cuda:0')
weight_dtype = torch.bfloat16

# init fill model
model_root = 'black-forest-labs/FLUX.1-Fill-dev'
pipeline_fill = FluxFillPipeline.from_pretrained(model_root, torch_dtype=weight_dtype).to(device)
pipeline_fill.enable_vae_tiling()

# load lora
remove_lora_ckpt_path = '../checkpoints/omnitry_remove_objects_lora.safetensors' # download from https://huggingface.co/Kunbyte/OmniTry/
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    init_lora_weights="gaussian",
    target_modules=[
        'x_embedder',
        'attn.to_k', 'attn.to_q', 'attn.to_v', 'attn.to_out.0', 
        'attn.add_k_proj', 'attn.add_q_proj', 'attn.add_v_proj', 'attn.to_add_out', 
        'ff.net.0.proj', 'ff.net.2', 'ff_context.net.0.proj', 'ff_context.net.2', 
        'norm1_context.linear', 'norm1.linear', 'norm.linear', 'proj_mlp', 'proj_out'
    ]
)
pipeline_fill.transformer.add_adapter(lora_config)
with safe_open(remove_lora_ckpt_path, framework="pt") as f:
    lora_weights = {}
    for k in f.keys():
        param = f.get_tensor(k) 
        if k.startswith('module.'):
            k = k[len('module.'):]
        lora_weights[k] = param

    msg = pipeline_fill.transformer.load_state_dict(lora_weights, strict=False)    

# init img2img
model_root = 'black-forest-labs/FLUX.1-dev'
pipeline_img2img = FluxImg2ImgPipeline.from_pretrained(model_root, torch_dtype=weight_dtype).to(device)


def remove_garment(image_paths, mask_paths):

    img_conds, dilate_masks, origin_masks = [], [], []
    for image_path, mask_path in zip(image_paths, mask_paths):
        tryon_img = Image.open(image_path)
        mask = Image.open(mask_path)

        max_area = 1024 * 1024
        oH = tryon_img.height
        oW = tryon_img.width
        ratio = math.sqrt(max_area / (oW * oH))
        ratio = min(1, ratio)
        tW, tH = int(oW * ratio) // 16 * 16, int(oH * ratio) // 16 * 16
        transform = T.Compose([
            T.Resize((tH, tW)),
            T.ToTensor(),
        ])

        tryon_img = transform(tryon_img)
        mask = transform(mask)[:1]
        mask = (mask > 0).float()

        # prepare condition
        img_conds.append(tryon_img[None])
        kernel_size = random.randint(1, 15)
        iterations = random.randint(1, 7)
        dilate_mask = dilate_mask_process(mask[0], kernel_size=kernel_size, iterations=iterations)[None, None]
        dilate_masks.append(dilate_mask)
        origin_masks.append(mask[None])

    img_conds = torch.cat(img_conds, dim=0)
    dilate_masks = torch.cat(dilate_masks, dim=0)
    origin_masks = torch.cat(origin_masks, dim=0)
    
    # generate
    result_imgs = pipeline_fill(
        prompt=['a model'] * len(img_conds),
        image=img_conds,
        mask_image=dilate_masks,
        height=img_conds.size(2),
        width=img_conds.size(3),
        guidance_scale=30,
        num_inference_steps=20,
        generator=torch.Generator(device).manual_seed(0)
    ).images
    result_imgs = torch.cat([T.ToTensor()(img)[None] for img in result_imgs], dim=0)

    # img2img refine
    result_imgs_refined = pipeline_img2img(
        prompt=['a model'] * len(img_conds),
        image=result_imgs,
        strength=0.2,
        height=img_conds.size(2),
        width=img_conds.size(3),
        guidance_scale=3.5,
        num_inference_steps=20,
        generator=torch.Generator(device).manual_seed(0)
    ).images
    result_imgs_refined = torch.cat([T.ToTensor()(img)[None] for img in result_imgs_refined], dim=0)

    tryon_imgs = []
    model_imgs = []
    for init_img, result_img, result_img_refined, dilate_mask, origin_mask in zip(img_conds, result_imgs, result_imgs_refined, dilate_masks, origin_masks):

        def sigmoid(x, scale=1.0):
            return 1 / (1 + np.exp(-scale * x))

        origin_mask = np.array(origin_mask[0])
        dilate_mask = np.array(dilate_mask[0])
        blend_mask = origin_mask.copy().astype(np.float32)

        boundary_region = dilate_mask.astype(np.float32) - origin_mask.astype(np.float32)
        dist_from_origin = distance_transform_edt(1 - origin_mask)
        dist_from_dilate = distance_transform_edt(1 - dilate_mask)
        boundary_width = np.max(dist_from_origin[boundary_region > 0])
        normalized_dist = dist_from_origin[boundary_region > 0] / boundary_width
        sigmoid_input = 12 * (1 - normalized_dist) - 6
        blend_mask[boundary_region > 0] = sigmoid(sigmoid_input)
        mask_final = (1 - blend_mask)

        # blending
        tryon_blend_mask = torch.Tensor(mask_final).to(init_img)
        tryon_blend_mask = tryon_blend_mask.unsqueeze(0)
        tryon_img = init_img * (1 - tryon_blend_mask) + result_img_refined * tryon_blend_mask
        tryon_imgs.append(tryon_img)

        # [model] image blending
        model_img = result_img_refined
        model_imgs.append(model_img)

    return tryon_imgs, model_imgs


if __name__ == '__main__':

    # inference
    input_index_file = 'example_ground_objects.json'
    output_index_file = 'example_remove_objects.json'
    
    data = json.load(open(input_index_file))
    outs = []
    for index in tqdm(data):
        
        image_oss_keys, mask_oss_keys, remove_oss_keys = [], [], []
        for i, object_info in enumerate(index['objects']):
            garment_description = object_info['description']
            image_path = index['image_path']
            mask_path = object_info['mask']
                
            new_tryon_path = '.'.join(image_path.split('.')[:-1]) + '_{}_tryon.jpg'.format('_'.join(garment_description.split(' ')))
            remove_path =  '.'.join(image_path.split('.')[:-1]) + '_{}_remove.jpg'.format('_'.join(garment_description.split(' ')))
            index['objects'][i]['tryon'] = new_tryon_path
            index['objects'][i]['remove'] = remove_path

            tryon_imgs, model_imgs = remove_garment([image_path], [mask_path])
            torchvision.utils.save_image(tryon_imgs[0], new_tryon_path)
            torchvision.utils.save_image(model_imgs[0], remove_path)
            outs.append(index)

    # save
    with open(output_index_file, 'w+') as f:
        f.write(json.dumps(outs, indent=4, ensure_ascii=False))