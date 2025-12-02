import os
# --- Global Environment Configuration ---
# CRITICAL: Blocks all network requests to ensure local-only loading
os.environ["HF_OFFLINE"] = "1" 
os.environ["HF_HUB_DISABLE_DOWNLOAD_PROGRESS"] = "1"

# CRITICAL: Advanced PyTorch Memory Configuration to fight fragmentation/stalls
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128"

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
os.environ["GRADIO_TEMP_DIR"] = ".gradio"

# Speedup: Enable TF32 for matmuls (bf16/fp16 boost)
torch.backends.cuda.matmul.allow_tf32 = True

from omnitry.models.transformer_flux import FluxTransformer2DModel
from omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline

# --- Configuration ---
device = torch.device('cuda:0')
weight_dtype = torch.bfloat16 # bfloat16 for stability
args = OmegaConf.load('configs/omnitry_v1_unified.yaml')

# init model & pipeline (with local_files_only=True)
print("Loading Transformer...")
transformer = FluxTransformer2DModel.from_pretrained(
    f'{args.model_root}/transformer',
    low_cpu_mem_usage=True,
    local_files_only=True
).requires_grad_(False).to(dtype=weight_dtype)

print("Loading Pipeline...")
pipeline = FluxFillPipeline.from_pretrained(
    args.model_root, 
    transformer=transformer.eval(), 
    torch_dtype=weight_dtype,
    low_cpu_mem_usage=True,
    local_files_only=True
)

# Speedup: Compile transformer for faster inference (first run slower)
pipeline.transformer = torch.compile(pipeline.transformer, mode='reduce-overhead')

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

# Materialize meta LoRA parameters on CPU to enable proper weight loading (non-meta)
for n, m in transformer.named_modules():
    if isinstance(m, peft.tuners.lora.layer.Linear):
        for adapter in ['vtryon_lora', 'garment_lora']:
            if adapter in m.lora_A:  # Check if adapter exists for this layer
                for lora_layer in [m.lora_A, m.lora_B]:
                    param = lora_layer[adapter].weight  # Target the weight param specifically
                    if param.is_meta:
                        with torch.no_grad():
                            # Create real CPU tensor (safetensors loads to CPU anyway)
                            param.data = torch.zeros(param.shape, dtype=param.dtype, device='cpu')
                    # Also handle bias if present (rare for LoRA, but complete)
                    if hasattr(lora_layer[adapter], 'bias') and lora_layer[adapter].bias is not None:
                        bias = lora_layer[adapter].bias
                        if bias.is_meta:
                            with torch.no_grad():
                                bias.data = torch.zeros(bias.shape, dtype=bias.dtype, device='cpu')

with safe_open(args.lora_path, framework="pt") as f:
    lora_weights = {k: f.get_tensor(k) for k in f.keys()}
    transformer.load_state_dict(lora_weights, strict=False)

# hack lora forward
def create_hacked_forward(module):

    def lora_forward(self, active_adapter, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        if active_adapter is not None:
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
    if isinstance(m, peft.tuners.lora.layer.Linear):
        m.forward = create_hacked_forward(m)
print("Model fully configured.")

# Speedup: xFormers for efficient attention (install if needed: pip install xformers)
pipeline.enable_xformers_memory_efficient_attention()

# VRAM saving: Use MODEL offload (faster than sequential); comment out for full GPU if >20GB VRAM
pipeline.enable_model_cpu_offload()  # Faster swaps; disable for max speed
pipeline.vae.enable_tiling()
pipeline.enable_vae_slicing()

# --- Helper Functions ---
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate(person_image, object_image, object_class, steps=12, guidance_scale=7.0, seed=-1, progress=gr.Progress(track_tqdm=True)):
    # set seed
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    seed_everything(seed)

    # Pre-cleanup for mem
    torch.cuda.empty_cache()

    # resize model: Lower res cap for speed
    max_area = 512 * 768  # Reduced from 1024*1024 (~60% faster)
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
            gr.Slider(minimum=8, maximum=30, value=12, label="Steps"),  # Tighter range for speed
            gr.Slider(minimum=1.0, maximum=12.0, value=7.0, step=0.5, label="Guidance Scale"),  # Safer max
            gr.Number(value=-1, label="Seed"),
        ],
        outputs=gr.Image(type="pil", label="Output"),
        title="OmniTry Try-On API",
        description="Upload images and class for virtual try-on. API at /run/tryon.",
        api_name="tryon"
    )

    # Launch with queue for mobile concurrency
    iface.queue().launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        debug=True
    )