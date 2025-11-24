import io
import base64
import random
import math
import torch
import copy
import numpy as np
import torchvision.transforms as T
import logging
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from peft import LoraConfig
from safetensors import safe_open
from omegaconf import OmegaConf
import os
os.environ["GRADIO_TEMP_DIR"] = ".gradio"

# Custom imports
from omnitry.models.transformer_flux import FluxTransformer2DModel
from omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OmniTry API")

device = torch.device('cuda:0')
weight_dtype = torch.bfloat16
args = OmegaConf.load('configs/omnitry_v1_unified.yaml')
pipeline = None
transformer = None

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate(person_image, object_image, object_class, steps=20, guidance_scale=30, seed=-1):
    logger.info(f"Starting generate: class={object_class}, steps={steps}, guidance={guidance_scale}")
    
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    seed_everything(seed)

    # Original dynamic resize (quality-preserving)
    max_area = 1024 * 1024
    oW = person_image.width
    oH = person_image.height
    ratio = math.sqrt(max_area / (oW * oH))
    ratio = min(1, ratio)
    tW, tH = int(oW * ratio) // 16 * 16, int(oH * ratio) // 16 * 16
    transform = T.Compose([T.Resize((tH, tW)), T.ToTensor()])
    person_tensor = transform(person_image)
    logger.info(f"Person tensor shape: {person_tensor.shape}, mean: {person_tensor.mean().item():.3f}")

    # Original object resize and padding
    ratio = min(tW / object_image.width, tH / object_image.height)
    transform = T.Compose([
        T.Resize((int(object_image.height * ratio), int(object_image.width * ratio))),
        T.ToTensor(),
    ])
    object_image_padded = torch.ones_like(person_tensor)
    object_tensor = transform(object_image)
    new_h, new_w = object_tensor.shape[1], object_tensor.shape[2]
    min_x = (tW - new_w) // 2
    min_y = (tH - new_h) // 2
    object_image_padded[:, min_y: min_y + new_h, min_x: min_x + new_w] = object_tensor
    logger.info(f"Object padded shape: {object_image_padded.shape}, mean: {object_image_padded.mean().item():.3f}")

    # Prompts & conditions
    prompts = [args.object_map[object_class]] * 2
    logger.info(f"Prompts: {prompts}")
    img_cond = torch.stack([person_tensor, object_image_padded]).to(dtype=weight_dtype, device=device)
    mask = torch.zeros_like(img_cond).to(img_cond)
    logger.info(f"img_cond shape: {img_cond.shape}, mask shape: {mask.shape}")

    try:
        with torch.no_grad():
            output = pipeline(
                prompt=prompts,
                height=tH, width=tW,
                img_cond=img_cond, mask=mask,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=torch.Generator(device).manual_seed(seed),
            )
            img = output.images[0]
        
        # Fixed: Normalize to float before mean()
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array.astype(np.float32) / 255.0)  # 0-1 range
        img_mean = img_tensor.mean().item()
        logger.info(f"Generated img shape: {img.size}, tensor mean: {img_mean:.3f}")
        
        # Check for black (using NumPy for speed)
        if np.mean(img_array) < 1:  # Threshold: all-black ~0
            raise ValueError("Generated image is all black/zero—check inputs or model")
        
        return img
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise
    
# load_model function (unchanged from previous)
@app.on_event("startup")
async def load_model():
    global pipeline, transformer
    transformer = FluxTransformer2DModel.from_pretrained(
        f'{args.model_root}/transformer'
    ).requires_grad_(False).to(dtype=weight_dtype)

    pipeline = FluxFillPipeline.from_pretrained(
        args.model_root, transformer=transformer.eval(), torch_dtype=weight_dtype
    )

    pipeline.enable_model_cpu_offload()
    pipeline.vae.enable_tiling()

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=[
            'x_embedder', 'attn.to_k', 'attn.to_q', 'attn.to_v', 'attn.to_out.0',
            'attn.add_k_proj', 'attn.add_q_proj', 'attn.add_v_proj', 'attn.to_add_out',
            'ff.net.0.proj', 'ff.net.2', 'ff_context.net.0.proj', 'ff_context.net.2',
            'norm1_context.linear', 'norm1.linear', 'norm.linear', 'proj_mlp', 'proj_out'
        ]
    )
    transformer.add_adapter(lora_config, adapter_name='vtryon_lora')
    transformer.add_adapter(lora_config, adapter_name='garment_lora')

    with safe_open(args.lora_path, framework="pt") as f:
        lora_weights = {k: f.get_tensor(k) for k in f.keys()}
        transformer.load_state_dict(lora_weights, strict=False)

    def create_hacked_forward(module):
        def lora_forward(self, active_adapter, x, *args, **kwargs):
            result = self.base_layer(x, *args, **kwargs)
            if active_adapter is not None:
                torch_result_dtype = result.dtype
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                result = result + lora_B(lora_A(dropout(x))) * scaling
            return result

        def hacked_lora_forward(self, x, *args, **kwargs):
            return torch.cat((
                lora_forward(self, 'vtryon_lora', x[:1], *args, **kwargs),
                lora_forward(self, 'garment_lora', x[1:], *args, **kwargs),
            ), dim=0)

        return hacked_lora_forward.__get__(module, type(module))

    for n, m in transformer.named_modules():
        if hasattr(m, 'forward') and 'lora' in str(type(m)):
            m.forward = create_hacked_forward(m)

    logger.info("Model loaded successfully")

@app.post("/tryon")
async def try_on(
    person_image: UploadFile = File(...),
    object_image: UploadFile = File(...),
    object_class: str = Query(...),
    steps: int = Query(20, ge=5, le=50),  # Higher default for quality
    guidance_scale: float = Query(30, ge=1.0, le=50.0),  # Higher for adherence
    seed: int = Query(-1)
):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        person_pil = Image.open(io.BytesIO(await person_image.read())).convert("RGB")
        object_pil = Image.open(io.BytesIO(await object_image.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    if object_class not in args.object_map:
        raise HTTPException(status_code=400, detail=f"Invalid class: {object_class}. Options: {list(args.object_map.keys())}")

    try:
        output_img = generate(person_image=person_pil, object_image=object_pil, object_class=object_class,
                              steps=steps, guidance_scale=guidance_scale, seed=seed)
        
        buffered = io.BytesIO()
        output_img.save(buffered, format="PNG", optimize=False)  # No optimize for max quality
        buffered.seek(0)
        img_bytes = buffered.read()
        
        logger.info(f"Output PNG size: {len(img_bytes)} bytes")
        if len(img_bytes) < 10000:  # Adjusted threshold for larger files
            raise ValueError("Output PNG too small—generation failed")
        
        img_str = base64.b64encode(img_bytes).decode()

        return {
            "success": True,
            "image_base64": f"data:image/png;base64,{img_str}",
            "width": output_img.width,
            "height": output_img.height
        }
    except ValueError as ve:
        logger.warning(f"Value error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": pipeline is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_omnitry:app",
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=600,  # 10 min
        limit_max_requests=1000,
        log_level="info"
    )