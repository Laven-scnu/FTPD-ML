"""Generate captions with InternVL, optionally loading a LoRA checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from modelscope import AutoModel, AutoTokenizer
from peft import LoraConfig, PeftModel
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file: str, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(image) for image in images])


def load_records(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run InternVL caption generation on FTPD-CL JSON records.")
    parser.add_argument("--model-path", required=True, help="Local model path or ModelScope/Hugging Face model id.")
    parser.add_argument("--input-json", required=True, type=Path, help="JSON file with records containing images and caption fields.")
    parser.add_argument("--output-csv", default=Path("outputs/internvl_captions.csv"), type=Path)
    parser.add_argument("--lora-checkpoint", default=None, help="Optional LoRA checkpoint path.")
    parser.add_argument("--prompt", default="<image>\nDescribe this image: ")
    parser.add_argument("--max-new-tokens", default=1024, type=int)
    parser.add_argument("--max-tiles", default=12, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-flash-attn", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input_json)
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=not args.no_flash_attn,
        trust_remote_code=True,
    ).eval().to(device)
    if args.lora_checkpoint:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = PeftModel.from_pretrained(model, args.lora_checkpoint, config=lora_config).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open(mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "reference", "prediction"])
        writer.writeheader()

        for item in tqdm(records, desc="Generating captions"):
            image_path = item["images"][0]
            reference = item["caption"] if isinstance(item.get("caption"), str) else " ".join(item.get("caption", []))
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            try:
                pixel_values = load_image(image_path, max_num=args.max_tiles).to(dtype).to(device)
                prediction = model.chat(tokenizer, pixel_values, args.prompt, generation_config)
                writer.writerow({"image_path": image_path, "reference": reference, "prediction": prediction})
            except Exception as exc:
                print(f"Failed to process {image_path}: {exc}")


if __name__ == "__main__":
    main()
