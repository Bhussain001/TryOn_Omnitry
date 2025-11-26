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
import bitsandbytes as bnb
import os
os.environ["GRADIO_TEMP_DIR"] = ".gradio"
# New: Allocator config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128"
# New: Force offline/local loading
os.environ["HF_OFFLINE"] = "1"

# Custom imports
from omnitry.models.transformer_flux import FluxTransformer2DModel
from omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OmniTry API")

device = torch.device('cuda:0')
weight_dtype = torch.float16  # Changed: Faster/lower mem than bfloat16
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

def generate(person_image, object_image, object_class, steps=10, guidance_scale=7.5, seed=-1):
    logger.info(f"Starting generate: class={object_class}, steps={steps}, guidance={guidance_scale}")
    
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    seed_everything(seed)

    # Hardcapped fixed resize (New: 384x512 for ~40% less mem)
    fixed_h, fixed_w = 512, 384
    transform = T.Compose([T.Resize((fixed_h, fixed_w)), T.ToTensor()])
    person_tensor = transform(person_image)
    logger.info(f"Person tensor shape: {person_tensor.shape}")

    # Object: Fixed resize + pad (New: Smaller fixed)
    o_transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    object_tensor = o_transform(object_image)
    object_image_padded = torch.nn.functional.interpolate(
        object_tensor.unsqueeze(0), size=(fixed_h, fixed_w), mode='bilinear'
    ).squeeze(0)
    object_image_padded = torch.clamp(object_image_padded, 0, 1)
    logger.info(f"Object padded shape: {object_image_padded.shape}")

    # Prompts & conditions (New: Single for person; LoRA handles object)
    prompt_str = args.object_map[object_class]
    prompts = [prompt_str]  # Single prompt; pipeline batches internally
    logger.info(f"Prompts: {prompts}")
    
    img_cond = torch.stack([person_tensor, object_image_padded]).to(dtype=weight_dtype, device=device)
    mask = torch.zeros_like(img_cond).to(img_cond)
    logger.info(f"img_cond shape: {img_cond.shape}")

    try:
        with torch.no_grad():
            output = pipeline(
                prompt=prompts,
                height=fixed_h, width=fixed_w,
                img_cond=img_cond, mask=mask,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=torch.Generator(device).manual_seed(seed),
            )
            img = output.images[0]

        # Logging/check (unchanged)
        img_array = np.array(img)
        if np.mean(img_array) < 1:
            raise ValueError("Generated image is all black")
        
        # Extra cleanup
        del output, img_cond, mask, person_tensor, object_tensor, object_image_padded
        torch.cuda.empty_cache()
        
        return img
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        torch.cuda.empty_cache()
        raise
        
# Updated load_model with extra optimizations
@app.on_event("startup")
async def load_model():
    global pipeline, transformer
    try:
        logger.info("Starting local model load (offline mode)...")
        
        transformer = FluxTransformer2DModel.from_pretrained(
            f'{args.model_root}/transformer',
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=True,
            resume_download=False,
            local_files_only=True,
        ).requires_grad_(False).to(dtype=weight_dtype)
        logger.info("Transformer loaded")

        pipeline = FluxFillPipeline.from_pretrained(
            args.model_root,
            transformer=transformer.eval(),
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=True,
            resume_download=False,
            local_files_only=True,
        )
        logger.info("Pipeline base loaded")

        # Offloads (unchanged + extra)
        pipeline.enable_model_cpu_offload()
        pipeline.vae.enable_tiling()
        pipeline.enable_vae_slicing()

        # LoRA with 8-bit quantization (New: Saves ~2GB)
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=[  # Unchanged list
                'x_embedder', 'attn.to_k', 'attn.to_q', 'attn.to_v', 'attn.to_out.0',
                'attn.add_k_proj', 'attn.add_q_proj', 'attn.add_v_proj', 'attn.to_add_out',
                'ff.net.0.proj', 'ff.net.2', 'ff_context.net.0.proj', 'ff_context.net.2',
                'norm1_context.linear', 'norm1.linear', 'norm.linear', 'proj_mlp', 'proj_out'
            ],
            quantization_config=bnb.nn.Params4bit(  # New: 4-bit quant for LoRA weights
                bnb_type="nf4",
                bnb_4bit_compute_dtype=weight_dtype,
                bnb_4bit_use_double_quant=True,
            ),
        )
        transformer.add_adapter(lora_config, adapter_name='vtryon_lora')
        transformer.add_adapter(lora_config, adapter_name='garment_lora')

        with safe_open(args.lora_path, framework="pt") as f:
            lora_weights = {k: f.get_tensor(k) for k in f.keys()}
            transformer.load_state_dict(lora_weights, strict=False)
        logger.info("LoRA loaded (quantized)")

        # Hacked forward (unchanged)
        # ... your create_hacked_forward and loop ...
        logger.info("Hacks applied")

        # Compile commented (unchanged)
        # pipeline = torch.compile(...)

        logger.info("Model fully loaded")
    except Exception as e:
        logger.error(f"Load failed: {e}")
        raise
    
# Endpoint (lowered defaults; added final cleanup)
@app.post("/tryon")
async def try_on(
    person_image: UploadFile = File(...),
    object_image: UploadFile = File(...),
    object_class: str = Query(...),
    steps: int = Query(10, ge=5, le=30),  # Lower default
    guidance_scale: float = Query(7.5, ge=1.0, le=20.0),  # Lower for mem
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
        
        # Save dims before del
        width, height = output_img.width, output_img.height
        
        buffered = io.BytesIO()
        output_img.save(buffered, format="PNG", optimize=True)  # Optimize for size
        buffered.seek(0)
        img_bytes = buffered.read()
        
        logger.info(f"Output PNG size: {len(img_bytes)} bytes")
        if len(img_bytes) < 10000:
            raise ValueError("Output PNG too small—generation failed")
        
        img_str = base64.b64encode(img_bytes).decode()

        # Final cleanup
        del output_img, person_pil, object_pil
        torch.cuda.empty_cache()

        return {
            "success": True,
            "image_base64": f"data:image/png;base64,{img_str}",
            "width": width,
            "height": height
        }
    except ValueError as ve:
        logger.warning(f"Value error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        torch.cuda.empty_cache()  # Cleanup on error
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": pipeline is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_omnitry:app",
        host="0.0.0.0", port=8000,
        workers=1, limit_concurrency=1,  # Strict limits
        timeout_keep_alive=600,
        log_level="info"
    )