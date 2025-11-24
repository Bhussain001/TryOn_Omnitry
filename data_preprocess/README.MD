# Data Pre-process of OmniTry

The data prepartion of OmniTry consists of three parts:
1. Listing objects in image (using MLLM, stage-1 only)
2. Grounding the masks of objects (using Grounding-DINO and SAM)
3. Removing the objects (using FLUX-Fill with fine-tuned removal LoRA)

## Step 1: Listing Objects
Prepare the index file as exampled in `example_raw.json`.
```
[
    {
        "image_path": "file:///path_to_your_tryon_image.jpg",
        "object_path": "file:///path_to_your_object_image.jpg"  // this is optional for stage-1
    }
]
```

Then run
```
python infer_list_objects.py
```

## Step 2: Grounding Object Masks
Prepare the models and environments as shown in [OmniTry-Bench](../omnitry_bench/README.MD). Then run
```
python infer_ground_objects.py
```
which will generate the mask image files in the same place of original try-on images.

## Step 3: Removing Objects
The traceless erasing is implemented in this step, which includes a fine-tuned FLUX-Fill (with LoRA) for removal, together with image-to-image translation (with FLUX) and mask-based blending. 

Firstly, download the [removal lora](https://huggingface.co/Kunbyte/OmniTry/blob/main/omnitry_remove_objects_lora.safetensors), then run
```
python infer_remove_objects.py
```
