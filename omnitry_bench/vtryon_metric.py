import json
import os
import torch
import torchvision
import lpips
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from transformers import ViTImageProcessor, ViTModel
from torchmetrics.image import StructuralSimilarityIndexMeasure # (SSIM)
from PIL import Image
from huggingface_hub import snapshot_download


device = torch.device('cuda:0')
weight_dtype = torch.bfloat16

# load CLIP model
clip_pt_path = '../checkpoints/clip-vit-base-patch32/'   
snapshot_download(repo_id="openai/clip-vit-base-patch32", local_dir=clip_pt_path)
clip_model = CLIPModel.from_pretrained(clip_pt_path).requires_grad_(False).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_pt_path)

# load DINO model
dino_pt_path = '../checkpoints/dino-vits16/'
snapshot_download(repo_id="facebook/dino-vits16", local_dir=dino_pt_path)
dino_model = ViTModel.from_pretrained(dino_pt_path).requires_grad_(False).to(device)
dino_processor = ViTImageProcessor.from_pretrained(dino_pt_path)

# load LPIPS（default: AlexNet）
lpips_model = lpips.LPIPS(net='alex', version='0.1').to(device).eval()
# load SSIM（PyTorch imp）
ssim_model = StructuralSimilarityIndexMeasure(data_range=1.0).to(device).eval()

## the OmniTryBench index file
benchmark_file = '../OmniTry_Bench/omni_vtryon_benchmark_small_v1.json'
## the Try-on result direction
result_dir = "../evaluation_results/"


result_file = os.path.join(result_dir, f'result.json')
result_detail_file = os.path.join(result_dir, f'result_detail.json')

prompt_pos_dict = {
    # accessary
    "earrings": "on ears",
    "earring": "on ear", 
    "necklace": "around neck",
    "bracelet": "around wrist",
    "ring": "on finger",
    "brooch": "on chest",
    "anklet": "around ankle",
    # head
    "hat": "on head",
    "cap": "on head",
    "hair accessory": "in hair",
    "veil": "over head",
    # face
    "glasses": "on face",
    "sunglasses": "on face",
    "mask": "over face",
    # upper body
    "tie": "around collar",
    "bow tie": "around collar",
    "necktie": "around collar",
    "scarf": "around neck",
    "gloves": "on hands",
    "watch": "on wrist",
    # lower body
    "belt": "around waist",
    "garter": "around thigh",
    "socks": "on feet",
    "stockings": "on legs", 
    # bag
    "bag": "on shoulder or in hand",
    "backpack": "on back",
    "purse": "in hand",
    "clutch": "in hand",
    "wallet": "in pocket",
    # 
    "dress": "on body",
    "gown": "on body",
    "top cloth": "on upper body",
    "coat": "over body",
    "bottom cloth": "on lower body", 
    # shoe
    "shoes": "on feet",
    "shoe": "on foot",
    "flats": "on feet",
    # 
    "costume": "on body",
    "uniform": "on body",
    "swimsuit": "on body",
    "lingerie": "on body",
    "apron": "over clothes",
    "cape": "over shoulders",

    # 
    "watch": "on left wrist with crown upward",  # 
    "shirt": "tucked into waistband",           # 
    "summer uniform": "with collar buttoned",   # 
    "badge": "on right chest 1cm above pocket", # 
    "brooch": "left chest for men, flexible for women",  # 
    "work ID": "centered above left pocket",    # 
    "socks": "dark for men, nude for women",   # 
    "tie clip": "lower half of tie for men",    # 
    "prosecutor emblem": "1cm above left pocket",  # 
    "security armband": "left arm with shoulder strap"  # 
}



to_tensor = T.ToTensor()
# 
to_pil = T.ToPILImage()

def preprocess_lpips_ssim(image_pil):
    """transfor PIL img to the Tensor (LPIPS & SSIM) """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # LPIPS need [-1,1]
    ])
    return transform(image_pil).unsqueeze(0).to(device)

def mask_white(tryon, mask):
    # tryon: (C, H, W)
    # mask: (1, H, W)
    _, orig_H, orig_W = tryon.shape
    
    blank = torch.ones_like(tryon[0])
    for idx in range(tryon.shape[0]):
        tryon[idx] = torch.where(mask[0].gt(0), tryon[idx], blank)
    return tryon

def mask_reverse_white(tryon, mask):
    # tryon: (C, H, W)
    # mask: (1, H, W)
    _, orig_H, orig_W = tryon.shape
    
    blank = torch.ones_like(tryon[0])
    for idx in range(tryon.shape[0]):
        tryon[idx] = torch.where(mask[0].gt(0), blank, tryon[idx])
    return tryon

    
def mask_crop_ori(tryon, mask, inter_size=False, size_img=None):
    # tryon: (C, H, W)
    # mask: (1, H, W)
    _, orig_H, orig_W = tryon.shape

    if mask.gt(0).any():
        y_indices, x_indices = torch.where(mask[0].gt(0)) ## torch.tensor([0.2, 0, 1, -1]).bool() -> tensor([ True, False,  True,  True])
        y_min, y_max = y_indices.min().item(), y_indices.max().item()
        x_min, x_max = x_indices.min().item(), x_indices.max().item()

        h_msk = min(orig_H, (y_max - y_min + 60))
        w_msk = min(orig_W, (x_max - x_min + 60))

        w_msk_4_h = h_msk * orig_W // orig_H
        new_w = min(max(w_msk, w_msk_4_h), orig_W)
        if new_w == orig_W:
            return tryon
            
        new_h = int(new_w * orig_H / orig_W)

        x_mi = max(int((x_min+x_max)/2 - new_w/2), 0)
        y_mi = max(int((y_min+y_max)/2 - new_h/2), 0)

        ## crop
        tryon = tryon[:, y_mi: (y_mi+new_h), x_mi: (x_mi+new_w)]
    else:
        print("mask is none")
    
    if inter_size:
        image_tensor = tryon.unsqueeze(0)
        target_size = (orig_H, orig_W)
        if size_img is not None:
            target_size = (size_img.size(1), size_img.size(2))
        resized_image_tensor = F.interpolate(
            image_tensor, 
            size=target_size, 
            mode='bicubic',  #
            align_corners=False  # 
        )
        tryon = resized_image_tensor.squeeze(0)

    return tryon


if __name__ == '__main__':
    test_data = json.loads(open(benchmark_file, 'r'))

    result_out = {}
    result_detail_out = []

    result_crop_imgs = []
    garment_imgs = []
    garment_crop_imgs =[]

    result_mask_none_num = 0
    skip_infos = []

    garment_class= ""
    class_name, big_class = "", ""
    dino_i_s, clip_i_s, clip_t_s, dino_i_bk_s, lpips_s, ssim_s, cate_n = -1, -1, -1, -1, -1, -1, -1
    dino_i_sum, clip_i_sum, clip_t_sum, dino_i_bk_sum, lpips_sum, ssim_sum, data_n = 0, 0, 0, 0, 0, 0, 0
    
    for d in test_data:
        idx = d['id']
        model_path = d['person']['img_path']
        garment_path = d['object']['img_path']

        if d['class_name'].split('_')[0] != big_class: # class_name
            if cate_n != -1 and cate_n != 0:
                dino_i_avg = dino_i_s /float(cate_n)
                clip_i_avg = clip_i_s /float(cate_n)
                clip_t_avg = clip_t_s /float(cate_n)
                dino_i_bk_avg = dino_i_bk_s/float(cate_n)
                lpips_avg = lpips_s /float(cate_n)
                ssim_avg = ssim_s /float(cate_n)
                class_clip = {}
                class_clip["dino_i_avg"] = dino_i_avg
                class_clip["clip_i_avg"] = clip_i_avg
                class_clip["clip_t_avg"] = clip_t_avg
                class_clip["dino_i_bk_avg"] = dino_i_bk_avg
                class_clip["lpips_avg"] = lpips_avg
                class_clip["ssim_avg"] = ssim_avg
                result_out[big_class] = {} # result_out[class_name] = {}
                result_out[big_class] = class_clip # result_out[class_name] = class_clip

                dino_i_sum += dino_i_avg
                clip_i_sum += clip_i_avg
                clip_t_sum += clip_t_avg
                dino_i_bk_sum += dino_i_bk_avg
                lpips_sum += lpips_avg
                ssim_sum += ssim_avg
                data_n += 1 # 
            dino_i_s, clip_i_s, clip_t_s, dino_i_bk_s, lpips_s, ssim_s, cate_n = 0, 0, 0, 0, 0, 0, 0

        garment_class = d['garment_class']
        class_name = d['class_name']
        big_class = d['class_name'].split('_')[0]

        result_tryon_path = os.path.join(result_dir, f'{idx}.jpg')
        result_mask_path = os.path.join(result_dir, f'{idx}_mask.jpg')
        if not os.path.exists(result_tryon_path):
            result_tryon_path = os.path.join(result_dir, f'{class_name}_{idx}.jpg')
            result_mask_path = os.path.join(result_dir, f'{class_name}_{idx}_mask.jpg')
        
        try:
            model_img = Image.open(model_path)
            tryon_img = Image.open(result_tryon_path)
            mask_img = Image.open(result_mask_path)
            garment_img = Image.open(garment_path)
            garment_imgs.append(garment_img)

            result_detail = {}
            result_detail['id'] = d['id']
            result_detail['person'] = d['person']
            result_detail['object'] = d['object']
            if "gt" in d.keys():
                result_detail['gt'] = d['gt']
            result_detail['garment_class'] = d['garment_class']
            result_detail['class_name'] = d['class_name']
            result_detail['gen_tryon'] = {'img_path': result_tryon_path}
            
            result_crop_path = os.path.join(result_dir, f'{idx}_crop_white.jpg')
            if not os.path.exists(result_crop_path):
                result_crop_path = os.path.join(result_dir, f'{class_name}_{idx}_crop_white.jpg')
            mask_tensor = to_tensor(mask_img)
            if mask_tensor.eq(0).all():
                result_mask_none_num += 1
            if not os.path.exists(result_crop_path):
                tryon_tensor = to_tensor(tryon_img)   ## T.functional.to_tensor(tryon_image)
                result_crop_tensor = mask_white(tryon_tensor, mask_tensor)
                result_crop_tensor = mask_crop_ori(result_crop_tensor, mask_tensor, inter_size=True, size_img=mask_tensor)
                if result_crop_tensor.shape[0] != 3:
                    print(f"result_crop_tensor  find is not 'RGB' : {class_name} {idx}")
                    result_crop_tensor = result_crop_tensor[0:1].repeat(3, 1, 1)
                torchvision.utils.save_image(result_crop_tensor, result_crop_path)
            result_crop = Image.open(result_crop_path)
            result_crop_imgs.append(result_crop)

            garment_mask_path = '.'.join(garment_path.split('.')[:-1]) + '_mask.jpg'
            garment_mask_img = Image.open(garment_mask_path)
            garment_crop_path = '.'.join(garment_path.split('.')[:-1]) + '_crop_white.jpg'
            if not os.path.exists(garment_crop_path):
                garment_tensor = to_tensor(garment_img)   ## T.functional.to_tensor(tryon_image)
                garment_mask_tensor = to_tensor(garment_mask_img) 
                garment_crop_tensor = mask_white(garment_tensor, garment_mask_tensor)
                garment_crop_tensor = mask_crop_ori(garment_crop_tensor, garment_mask_tensor, inter_size=True, size_img=garment_mask_tensor)
                if garment_crop_tensor.shape[0] != 3:
                    print(f"garment_crop_tensor find is not 'RGB' : {class_name} {idx}")
                    garment_crop_tensor = garment_crop_tensor[0:1].repeat(3, 1, 1)
                torchvision.utils.save_image(garment_crop_tensor, garment_crop_path)
            garment_crop = Image.open(garment_crop_path)
            garment_crop_imgs.append(garment_crop)

            # DINO image features
            with torch.no_grad():
                inputs_gen = dino_processor(images=result_crop, return_tensors="pt", padding=True).to(device) 
                inputs_garment = dino_processor(images=garment_crop, return_tensors="pt", padding=True).to(device) 
                
                # normalization
                image_features_gen = dino_model(**inputs_gen).last_hidden_state[:, 0]
                image_features_garment = dino_model(**inputs_garment).last_hidden_state[:, 0]
            
            dino_i_t = F.cosine_similarity(image_features_gen, image_features_garment, dim=-1).item()
            print(f"DINO-I  dino_i_t similarity: {dino_i_t:.4f}")
            # L2 normalization
            image_features_gen = image_features_gen / image_features_gen.norm(p=2, dim=-1, keepdim=True)
            image_features_garment = image_features_garment / image_features_garment.norm(p=2, dim=-1, keepdim=True)
            # cosine_similarity
            dino_i = F.cosine_similarity(image_features_gen, image_features_garment).item()
            print(f"DINO-I similarity: {dino_i:.4f}")
            result_detail["DINO-I"] = dino_i

            # CLIP image features（no text prompt）
            with torch.no_grad():
                inputs_gen = clip_processor(images=result_crop, return_tensors="pt", padding=True).to(device) 
                inputs_garment = clip_processor(images=garment_crop, return_tensors="pt", padding=True).to(device) 
                
                # normalization
                image_features_gen = clip_model.get_image_features(**inputs_gen)
                image_features_garment = clip_model.get_image_features(**inputs_garment)  

            # L2 normalization
            image_features_gen = image_features_gen / image_features_gen.norm(p=2, dim=-1, keepdim=True)
            image_features_garment = image_features_garment / image_features_garment.norm(p=2, dim=-1, keepdim=True)
            
            # cosine_similarity
            clip_i = torch.matmul(image_features_gen, image_features_garment.T).item()
            print(f"CLIP-I similarity: {clip_i:.4f}")
            result_detail["CLIP-I"] = clip_i

            # CLIP-T (image and text)
            text_prompt = [f"{garment_class} {prompt_pos_dict[garment_class]} of the model"]
            if "gt" in d.keys() and "caption" in d['gt'].keys():
                text_prompt = [d['gt']['caption']]
            with torch.no_grad():
                inputs = clip_processor(
                    text=text_prompt, 
                    images=tryon_img, 
                    return_tensors="pt", 
                    padding=True
                ).to(device)
                if inputs['input_ids'].size()[-1] >= 77:
                    text_prompt = [d['gt']['caption'].split('. ')[0]]
                    inputs = clip_processor(
                        text=text_prompt, 
                        images=tryon_img, 
                        return_tensors="pt", 
                        padding=True
                    ).to(device)
                outputs_ct = clip_model(**inputs)

            # image-text similarity
            logits_per_image = outputs_ct.logits_per_image
            clip_t = logits_per_image.item()
            print(f"CLIP-T similarity: {clip_t:.4f}")
            result_detail["CLIP-T"] = clip_t

            tryon_tensor = to_tensor(tryon_img)   ## T.functional.to_tensor(tryon_image)
            mask_tensor = to_tensor(mask_img)
            result_residue_path = os.path.join(result_dir, f'{class_name}_{idx}_residue_white.jpg')
            if not os.path.exists(result_residue_path):
                result_residue_tensor = mask_reverse_white(tryon_tensor, mask_tensor)
                torchvision.utils.save_image(result_residue_tensor, result_residue_path)
                result_residue = Image.open(result_residue_path)
            else:
                result_residue = Image.open(result_residue_path)
                result_residue_tensor = to_tensor(result_residue)
            
            model_residue_path = os.path.join(result_dir, f'{class_name}_{idx}_model_residue.jpg')
            if not os.path.exists(model_residue_path):
                transform = T.Compose([
                    T.Resize((tryon_img.height, tryon_img.width)),
                    T.ToTensor(),
                    # T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
                model_tensor = transform(model_img)
                model_residue_tensor = mask_reverse_white(model_tensor, mask_tensor)
                # model_residue = to_pil(model_residue_tensor)
                torchvision.utils.save_image(model_residue_tensor, model_residue_path)
            model_residue = Image.open(model_residue_path)
            model_residue_tensor = to_tensor(model_residue)

            # background DINo
            with torch.no_grad():
                inputs_gen = dino_processor(images=result_residue, return_tensors="pt", padding=True).to(device) 
                inputs_garment = dino_processor(images=model_residue, return_tensors="pt", padding=True).to(device) 
                
                # normalization
                image_features_gen = dino_model(**inputs_gen).last_hidden_state[:, 0]
                image_features_garment = dino_model(**inputs_garment).last_hidden_state[:, 0]
            
            dino_i_bk_t = F.cosine_similarity(image_features_gen, image_features_garment, dim=-1).item()
            print(f"DINO-I back  dino_i_t similarity: {dino_i_bk_t:.4f}")

            # L2 similarity
            image_features_gen = image_features_gen / image_features_gen.norm(p=2, dim=-1, keepdim=True)
            image_features_garment = image_features_garment / image_features_garment.norm(p=2, dim=-1, keepdim=True)
            
            # cosine similarity
            dino_i_bk = F.cosine_similarity(image_features_gen, image_features_garment).item()
            print(f"DINO-I back similarity: {dino_i_bk:.4f}")
            result_detail["DINO-I-bk"] = dino_i_bk

            # background LPIPS
            # preprocessing images
            result_residue_ls_tensor = preprocess_lpips_ssim(result_residue)
            model_residue_ls_tensor = preprocess_lpips_ssim(model_residue)
            with torch.no_grad():
                lpips_value = lpips_model(model_residue_ls_tensor, result_residue_ls_tensor).item()
            print(f"LPIPS: {lpips_value:.4f}")
            result_detail["LPIPS"] = lpips_value

            # background SSIM（PyTorch imp）
            with torch.no_grad():
                ssim_value = ssim_model(model_residue_tensor.unsqueeze(0), result_residue_tensor.unsqueeze(0)).item()
            print(f"SSIM: {ssim_value:.4f}")
            result_detail["SSIM"] = ssim_value
            
            dino_i_s += dino_i
            clip_i_s += clip_i
            clip_t_s += clip_t
            dino_i_bk_s += dino_i_bk
            lpips_s += lpips_value
            ssim_s += ssim_value
            cate_n += 1
            result_detail_out.append(result_detail)
            
        except Exception as e:
            print(e)
            print(idx)
            print(class_name)
            print(model_path)
            print(garment_path)
            print("skip")
            skip_one = {"error e: ":str(e), "id":idx, "class_name":class_name, "model_path":model_path, "garment_path":garment_path}
            skip_infos.append(skip_one)
            pass

    if cate_n > 0:
        dino_i_avg = dino_i_s /float(cate_n)
        clip_i_avg = clip_i_s /float(cate_n)
        clip_t_avg = clip_t_s /float(cate_n)
        lpips_avg = lpips_s /float(cate_n)
        dino_i_bk_avg = dino_i_bk_s /float(cate_n)
        ssim_avg = ssim_s /float(cate_n)
        class_clip = {}
        class_clip["dino_i_avg"] = dino_i_avg
        class_clip["clip_i_avg"] = clip_i_avg
        class_clip["clip_t_avg"] = clip_t_avg
        class_clip["dino_i_bk_avg"] = dino_i_bk_avg
        class_clip["lpips_avg"] = lpips_avg
        class_clip["ssim_avg"] = ssim_avg
        result_out[big_class] = {} # result_out[class_name] = {}
        result_out[big_class] = class_clip # result_out[class_name] = class_clip

    if len(skip_infos) > 0:
        result_out["skip_infos"] = skip_infos
        print(skip_infos)
    
    ## last class
    dino_i_sum += dino_i_avg
    clip_i_sum += clip_i_avg
    clip_t_sum += clip_t_avg
    dino_i_bk_sum += dino_i_bk_avg
    lpips_sum += lpips_avg
    ssim_sum += ssim_avg
    data_n += 1 

    dino_i_avg = dino_i_sum / float(data_n)
    clip_i_avg = clip_i_sum / float(data_n)
    clip_t_avg = clip_t_sum / float(data_n)
    dino_i_bk_avg = dino_i_bk_sum / float(data_n)
    lpips_avg = lpips_sum / float(data_n)
    ssim_avg = ssim_sum / float(data_n)
    print("all class num ：{}".format(data_n))
    print(f"dino_i_avg: {dino_i_avg:.4f}")
    print(f"clip_i_avg: {clip_i_avg:.4f}")
    print(f"clip_t_avg: {clip_t_avg:.4f}")
    print(f"dino_i_bk_avg: {dino_i_bk_avg:.4f}")
    print(f"lpips_avg: {lpips_avg:.4f}")
    print(f"ssim_avg: {ssim_avg:.4f}")
    result_out["dino_i_avg"] = dino_i_avg
    result_out["clip_i_avg"] = clip_i_avg
    result_out["clip_t_avg"] = clip_t_avg
    result_out["dino_i_bk_avg"] = dino_i_bk_avg
    result_out["lpips_avg"] = lpips_avg
    result_out["ssim_avg"] = ssim_avg

    print("result num whers object mask is none: {}".format(result_mask_none_num))
    result_out["grounding_accuracy"] = (len(test_data) - result_mask_none_num) / float(len(test_data))

    os.makedirs(os.path.dirname(result_detail_file), exist_ok=True) 
    with open(result_detail_file, 'w', encoding='utf-8') as result_detail_file_w:
        json.dump(result_detail_out, result_detail_file_w, ensure_ascii=False, indent=4)
    print(f'The detailed result file is saved in {result_detail_file}')

    with open(result_file, 'w', encoding='utf-8') as result_file_w:
        json.dump(result_out, result_file_w, ensure_ascii=False, indent=4)
    print(f'The summary result file is saved in {result_file}')
