#
# Modified from LLaVA/predict.py
# Please see ACKNOWLEDGEMENTS for details about LICENSE
#
import os
import argparse

import time
import torch
from PIL import Image

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


from llava.mm_utils import (
    tokenizer_image_token, process_images, get_model_name_from_path
)
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)

model = None
tokenizer = None
image_processor = None


def predict(args):
    global model, tokenizer, image_processor
    print("predicting...")
    # Remove generation config from model folder
    # to read generation parameters from args
    model_path = os.path.expanduser(args.model_path)
    generation_config = None
    if os.path.exists(os.path.join(model_path, 'generation_config.json')):
        generation_config = os.path.join(model_path, '.generation_config.json')
        os.rename(os.path.join(model_path, 'generation_config.json'),
                  generation_config)

    # Load model
    if model is None:
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, args.model_base, model_name, device="mps")

    # Construct prompt
    qs = args.prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
            DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Set the pad token id for generation
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Tokenize prompt
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(torch.device("mps"))

    # Load and preprocess image
    image = Image.open(args.image_file).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]

    # Run inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half(),
            image_sizes=[image.size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=256,
            use_cache=True)

        outputs = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)[0].strip()
        print(outputs)

    # Restore generation config
    if generation_config is not None:
        os.rename(generation_config, os.path.join(
            model_path, 'generation_config.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="./checkpoints/llava-fastvithd_0.5b_stage3")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str,
                        default=None, help="location of image file")
    parser.add_argument("--prompt", type=str,
                        default="Describe the image.", help="Prompt for VLM.")
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    # 1) select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[INFO] running on {device}")

    # 2) load model + tokenizer + image processor once
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name, device=device
    )

    # 3) optionally compile (PyTorch ≥2.0)
    if hasattr(torch, "compile"):
        print("[INFO] compiling model for inference")
        model = torch.compile(model)

    # 4) build & cache prompt
    qs = args.prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
            DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).to(device).unsqueeze(0)

    # 5) preprocess & cache image
    img = Image.open(args.image_file).convert("RGB")
    img_tensor = process_images([img], image_processor, model.config)[0]
    img_tensor = img_tensor.to(device).half()
    img_sizes = [img.size]

    # 6) override pad token in memory (no disk rename)
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # 7) run inference args.repeat times
    for i in range(1, args.repeat + 1):
        t0 = time.time()
        with torch.inference_mode():
            out_ids = model.generate(
                input_ids,
                images=img_tensor.unsqueeze(0),
                image_sizes=img_sizes,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        text = tokenizer.batch_decode(
            out_ids, skip_special_tokens=True)[0].strip()
        elapsed = time.time() - t0
        print(f"[run {i}/{args.repeat}] {text}")
        print(f"  → time: {elapsed:.3f}s\n")


if __name__ == "__main__":
    main()
