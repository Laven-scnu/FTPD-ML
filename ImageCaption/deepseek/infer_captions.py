"""Generate captions with DeepSeek-VL, optionally loading a LoRA checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images
from peft import LoraConfig, PeftModel
from tqdm import tqdm


DEFAULT_LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def load_records(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepSeek-VL caption generation on FTPD-CL JSON records.")
    parser.add_argument("--model-path", required=True, help="Local model path or ModelScope/Hugging Face model id.")
    parser.add_argument("--input-json", required=True, type=Path, help="JSON file with records containing images and caption fields.")
    parser.add_argument("--output-csv", default=Path("outputs/deepseek_captions.csv"), type=Path)
    parser.add_argument("--lora-checkpoint", default=None, help="Optional LoRA checkpoint path.")
    parser.add_argument("--prompt", default="<image_placeholder>Describe this image:")
    parser.add_argument("--max-new-tokens", default=512, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input_json)
    device = torch.device(args.device)

    processor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    model = MultiModalityCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    model = model.to(torch.float16 if device.type == "cuda" else torch.float32).to(device)
    if args.lora_checkpoint:
        model = PeftModel.from_pretrained(model, args.lora_checkpoint, config=DEFAULT_LORA_CONFIG)
    model = model.eval()

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
                conversation = [
                    {"role": "User", "content": args.prompt, "images": [image_path]},
                    {"role": "Assistant", "content": ""},
                ]
                pil_images = load_pil_images(conversation)
                prepare_inputs = processor(conversations=conversation, images=pil_images, force_batchify=True).to(model.device)
                for key, value in vars(prepare_inputs).items():
                    if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
                        setattr(prepare_inputs, key, value.to(torch.float16))

                inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
                outputs = model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
                prediction = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                writer.writerow({"image_path": image_path, "reference": reference, "prediction": prediction})
            except Exception as exc:
                print(f"Failed to process {image_path}: {exc}")


if __name__ == "__main__":
    main()
