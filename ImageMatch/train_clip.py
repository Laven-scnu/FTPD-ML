"""Fine-tune CLIP on image-caption pairs.

Input format:
    dataset_root/
      sample_000001/
        figure.png
        caption.txt
      sample_000002/
        figure.jpg
        caption.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import clip
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}


class ImageTextDataset(Dataset):
    def __init__(self, image_paths: list[Path], text_paths: list[Path], preprocess, device: torch.device):
        self.image_paths = image_paths
        self.text_paths = text_paths
        self.preprocess = preprocess
        self.device = device

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.preprocess(image)
        text = self.text_paths[idx].read_text(encoding="utf-8", errors="ignore").strip()
        text_tokens = clip.tokenize([text], truncate=True).squeeze(0)
        return image, text_tokens


def get_image_and_caption_paths(root_folder: Path) -> tuple[list[Path], list[Path]]:
    image_paths: list[Path] = []
    text_paths: list[Path] = []

    for sample_dir in sorted(p for p in root_folder.iterdir() if p.is_dir()):
        image_path = next((p for p in sorted(sample_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS), None)
        caption_path = sample_dir / "caption.txt"
        if image_path and caption_path.exists():
            image_paths.append(image_path)
            text_paths.append(caption_path)

    return image_paths, text_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune OpenAI CLIP on FTPD-CL image-caption pairs.")
    parser.add_argument("--data-root", required=True, type=Path, help="Root directory containing image/caption sample folders.")
    parser.add_argument("--output-dir", default=Path("outputs/clip"), type=Path, help="Directory for checkpoints.")
    parser.add_argument("--model-name", default="clip-finetune", help="Checkpoint filename prefix.")
    parser.add_argument("--clip-backbone", default="ViT-B/32", help="CLIP backbone name.")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=1e-6, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--max-samples", default=None, type=int, help="Optional cap for quick experiments.")
    parser.add_argument("--checkpoint-every", default=4, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net, preprocess = clip.load(args.clip_backbone, device=device, jit=False)
    net.train()

    image_paths, text_paths = get_image_and_caption_paths(args.data_root)
    if args.max_samples:
        image_paths = image_paths[: args.max_samples]
        text_paths = text_paths[: args.max_samples]
    if not image_paths:
        raise ValueError(f"No image-caption pairs found under {args.data_root}")

    logger.info(f"Found {len(image_paths)} image-caption pairs")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ImageTextDataset(image_paths, text_paths, preprocess, device)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    use_amp = device.type == "cuda"
    for epoch in range(args.epochs):
        total_loss = 0.0
        for images, label_tokens in train_loader:
            images = images.to(device)
            label_tokens = label_tokens.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits_per_image, logits_per_text = net(images, label_tokens)
                ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
                loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                clip.model.convert_weights(net)
            total_loss += loss.item()

        epoch_loss = total_loss / max(len(train_loader), 1)
        logger.info(f"train epoch={epoch} loss={epoch_loss:.6f}")
        torch.save(net.state_dict(), args.output_dir / f"{args.model_name}_epoch_{epoch}.pth")

        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "network": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                args.output_dir / f"{args.model_name}_checkpoint.pth",
            )
        scheduler.step()


if __name__ == "__main__":
    main()
