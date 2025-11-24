import io
import base64
import random
import math
import torch
import copy
import numpy as np
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from diffusers import FluxPipeline  # Fallback if needed; actual uses custom
from peft import LoraConfig
from safetensors import safe_open
from omegaconf import OmegaConf
import os
os.environ["GRADIO_TEMP_DIR"] = ".gradio"  # Retained for compat

# Custom imports from repo
from omnitry.models.transformer_flux import FluxTransformer2DModel
from omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline

app = FastAPI(title="OmniTry API", description="Fast Virtual Try-On for Mobile")

# Globals (optimized)
device = torch.device('cuda:0')
weight_dtype = torch.float16  # Faster than bfloat16; adjust if needed
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
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    seed_everything(seed)

    # Optimized fixed resize (faster than dynamic; common for try-on)
    fixed_size = (512, 768)
    transform = T.Compose([T.Resize(fixed_size), T.ToTensor()])
    person_image = transform(person_image)

    # Simplified object resize/pad
    o_transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])  # Fixed for speed
    object_image = o_transform(object_image)
    object_image_padded = torch.nn.functional.interpolate(
        object_image.unsqueeze(0), size=fixed_size, mode='bilinear', align_corners=False
    ).squeeze(0)
    object_image_padded = torch.clamp(object_image_padded, 0, 1)  # Normalize

    # Prompts & conditions
    prompts = [args.object_map[object_class]] * 2
    img_cond = torch.stack([person_image, object_image_padded]).to(dtype=weight_dtype, device=device)
    mask = torch.zeros_like(img_cond).to(img_cond)

    with torch.no_grad():
        img = pipeline(
            prompt=prompts,
            height=fixed_size[1], width=fixed_size[0],
            img_cond=img_cond, mask=mask,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=torch.Generator(device).manual_seed(seed),
        ).images[0]

    return img

@app.on_event("startup")
async def load_model():
    global pipeline, transformer
    # Exact loading from demo
    transformer = FluxTransformer2DModel.from_pretrained(
        f'{args.model_root}/transformer'
    ).requires_grad_(False).to(dtype=weight_dtype)

    pipeline = FluxFillPipeline.from_pretrained(
        args.model_root, transformer=transformer.eval(), torch_dtype=weight_dtype
    )

    # VRAM optimizations
    pipeline.enable_model_cpu_offload()
    pipeline.vae.enable_tiling()

    # LoRA setup (exact)
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

    # Hacked LoRA forward (exact)
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
        if hasattr(m, 'forward') and 'lora' in str(type(m)):  # Simplified check
            m.forward = create_hacked_forward(m)

@app.post("/tryon")
async def try_on(
    person_image: UploadFile = File(..., description="Person image (JPEG/PNG)"),
    object_image: UploadFile = File(..., description="Object/garment image"),
    object_class: str = Query(..., description="Class e.g., 'top clothes'"),
    steps: int = Query(10, ge=5, le=30),
    guidance_scale: float = Query(7.5, ge=1.0, le=50.0),
    seed: int = Query(-1)
):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Load/validate images
    try:
        person_pil = Image.open(io.BytesIO(await person_image.read())).convert("RGB")
        object_pil = Image.open(io.BytesIO(await object_image.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    if object_class not in args.object_map:
        raise HTTPException(status_code=400, detail=f"Invalid class: {object_class}. Use: {list(args.object_map.keys())}")

    try:
        output_img = generate(person_image=person_pil, object_image=object_pil, object_class=object_class,
                              steps=steps, guidance_scale=guidance_scale, seed=seed)
        
        # Base64 encode for mobile
        buffered = io.BytesIO()
        output_img.save(buffered, format="PNG", optimize=True)  # Optimize for size/speed
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {
            "success": True,
            "image_base64": f"data:image/png;base64,{img_str}",
            "width": output_img.width,
            "height": output_img.height
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": pipeline is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")