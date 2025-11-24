import cv2
import numpy as np
import supervision as sv
import json
import torch
import torchvision
from tqdm import tqdm
import random
import sys
import os
import os.path as osp
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


# init VL model
query_prompt = \
    '''
    Please output in a list format the descriptions of items worn or held by people in the image, paying attention to the following notes:
    1. Consider all possible try-on and takeable items, but note to exclude clothes, shoes, and parts of the human body itself.
    2. Output format: ['object1_desc', 'object2_desc', ...], note that item descriptions should be in the form of interaction method + item information itself, such as: wearing/holding/carrying/using/trying on a XXX; if the item's position is unconventional, specify it, such as holding XXX in front of eyes.
    3. Output in English;
    4. When no such items exist, output an empty list [].
    '''

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-7B-Instruct',
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained(model_root)


if __name__ == '__main__':

    input_index_file = 'example_raw.json'
    output_index_file = 'example_list_objects.json'

    data = json.load(open(input_index_file))
    outs = []
    for d in tqdm(data):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": d['image_path'],
                    },
                    {
                        "type": "text", 
                        "text": query_prompt
                    },
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # organize output
        objects = eval(output_text[0])
        if len(objects) == 0:
            continue
        objects = list(set(objects))[:5]

        d['objects'] = objects
        outs.append(d)

    # save
    with open(output_index_file, 'w+') as f:
        f.write(json.dumps(outs, indent=4, ensure_ascii=False))