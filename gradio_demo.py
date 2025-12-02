import os
# --- Global Environment Configuration ---
# CRITICAL: Blocks all network requests to ensure local-only loading
os.environ["HF_OFFLINE"] = "1" 
os.environ["HF_HUB_DISABLE_DOWNLOAD_PROGRESS"] = "1"

# CRITICAL: Advanced PyTorch Memory Configuration to fight fragmentation/stalls
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128"

import gradio as gr
import torch
import diffusers
import transformers
import copy
import random
import numpy as np
import torchvision.transforms as T
import math
import peft
from peft import LoraConfig
from safetensors import safe_open
from omegaconf import OmegaConf
from accelerate import init_empty_weights, load_checkpoint_and_dispatch  # ADD: For low-mem loading
os.environ["GRADIO_TEMP_DIR"] = ".gradio"

from omnitry.models.transformer_flux import FluxTransformer2DModel
from omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline

# --- Configuration ---
device = torch.device('cuda:0')
weight_dtype = torch.float16  # Or torch.bfloat16 for Ampere+ GPUs
args = OmegaConf.load('configs/omnitry_v1_unified.yaml')

print("Loading Transformer with Accelerate low-mem dispatch...")
# Step 1: Init empty model (0 VRAM/RAM usage)
with init_empty_weights():
    transformer = FluxTransformer2DModel.from_pretrained(
        f'{args.model_root}/transformer',
        low_cpu_mem_usage=True,
        local_files_only=True
    )

# Step 2: Dispatch shards sequentially (low peak mem)
device_map = "auto"  # Balances GPU/CPU; or "sequential" for strict offload
transformer = load_checkpoint_and_dispatch(
    transformer,
    checkpoint=f'{args.model_root}/transformer',
    device_map=device_map,
    dtype=weight_dtype,
    offload_folder="offload_tmp",  # Temp CPU cache (auto-deletes)
    max_memory={0: "20GiB", "cpu": "32GiB"},  # Cap GPU; spill to CPU (adjust RAM if needed)
    low_cpu_mem_usage=True
)
transformer.requires_grad_(False)
transformer.eval()

print("Loading Pipeline...")
pipeline = FluxFillPipeline.from_pretrained(
    args.model_root, 
    transformer=transformer,  # Inherits dispatched state
    torch_dtype=weight_dtype,
    low_cpu_mem_usage=True,       
    local_files_only=True         
)

# Enable offloads EARLY (now model is dispatched)
pipeline.enable_sequential_cpu_offload(device_map=device_map)
pipeline.enable_model_cpu_offload()  # Offload inactive components
pipeline.vae.enable_tiling(tile_size=512)  # Chunk VAE (saves 2-4GB)
pipeline.enable_vae_slicing()

print("Inserting LoRA...")
# insert LoRA
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

# LoRA dispatch: Reload with Accelerate to move adapters (handles offload)
transformer = load_checkpoint_and_dispatch(
    transformer,
    checkpoint=f'{args.model_root}/transformer',  # Reuse base (adapters added in-place)
    device_map=device_map,
    dtype=weight_dtype,
    offload_folder="offload_tmp",
    max_memory={0: "20GiB", "cpu": "32GiB"},
    low_cpu_mem_usage=True
)

# hack lora forward
def create_hacked_forward(module):

    def lora_forward(self, active_adapter, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        if active_adapter is not None:
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            # Ensure x is on the correct device and dtype for LoRA
            x = x.to(device=lora_A.weight.device, dtype=lora_A.weight.dtype)
            result = result + lora_B(lora_A(dropout(x))) * scaling
        return result
    
    def hacked_lora_forward(self, x, *args, **kwargs):
        # Pre-align x to base device
        x = x.to(device=module.base_layer.weight.device, dtype=module.base_layer.weight.dtype)
        return torch.cat((
            lora_forward(self, 'vtryon_lora', x[:1], *args, **kwargs),
            lora_forward(self, 'garment_lora', x[1:], *args, **kwargs),
        ), dim=0)
    
    return hacked_lora_forward.__get__(module, type(module))

for n, m in transformer.named_modules():
    if isinstance(m, peft.tuners.lora.layer.Linear):
        m.forward = create_hacked_forward(m)
print("Model fully configured.")

# --- Helper Functions ---
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate(person_image, object_image, object_class, steps=20, guidance_scale=30, seed=-1, progress=gr.Progress(track_tqdm=True)):
    # set seed
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    seed_everything(seed)

    # resize model (LOWER RES FOR TESTING: 512x512 max to save VRAM)
    max_area = 512 * 512  # TEMP: Reduce for OOM testing; revert to 1024*1024 later
    oW = person_image.width
    oH = person_image.height

    ratio = math.sqrt(max_area / (oW * oH))
    ratio = min(1, ratio)
    tW, tH = int(oW * ratio) // 16 * 16, int(oH * ratio) // 16 * 16
    transform = T.Compose([
        T.Resize((tH, tW)),
        T.ToTensor(),
    ])
    person_image = transform(person_image)

    # resize and padding garment
    ratio = min(tW / object_image.width, tH / object_image.height)
    transform = T.Compose([
        T.Resize((int(object_image.height * ratio), int(object_image.width * ratio))),
        T.ToTensor(),
    ])
    object_image_padded = torch.ones_like(person_image)
    object_image = transform(object_image)
    new_h, new_w = object_image.shape[1], object_image.shape[2]
    min_x = (tW - new_w) // 2
    min_y = (tH - new_h) // 2
    object_image_padded[:, min_y: min_y + new_h, min_x: min_x + new_w] = object_image

    # prepare prompts & conditions
    prompts = [args.object_map[object_class]] * 2
    img_cond = torch.stack([person_image, object_image_padded]).to(dtype=weight_dtype, device=device) 
    mask = torch.zeros_like(img_cond).to(img_cond)

    with torch.no_grad():
        img = pipeline(
            prompt=prompts,
            height=tH,
            width=tW,    
            img_cond=img_cond,
            mask=mask,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=torch.Generator(device).manual_seed(seed),
        ).images[0]

    # Explicit cleanup
    del img_cond, mask, person_image, object_image_padded
    torch.cuda.empty_cache()

    return img

# --- Gradio Interface ---
if __name__ == '__main__':
    
    print("Launching Gradio Interface...")
    
    iface = gr.Interface(
        fn=generate,
        inputs=[
            gr.Image(type="pil", label="Person Image"),
            gr.Image(type="pil", label="Object Image"),
            gr.Dropdown(choices=list(args.object_map.keys()), label="Object Class"),
            gr.Slider(minimum=1, maximum=50, value=20, label="Steps"),
            gr.Slider(minimum=1, maximum=50, value=30, step=0.1, label="Guidance Scale"),
            gr.Number(value=-1, label="Seed"),
        ],
        outputs=gr.Image(type="pil", label="Output"),
        title="OmniTry Try-On API",
        description="Upload images and class for virtual try-on. API at /gradio_api/api/tryon.",
        api_name="tryon"
    )

    # Launch with queue for mobile concurrency
    iface.queue().launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        debug=True
    )