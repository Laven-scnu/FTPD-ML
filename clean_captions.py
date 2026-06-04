"""Clean caption.txt files produced by figure extraction."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from tqdm import tqdm


CLEAN_PATTERN = re.compile(
    r"\[\d+\]"  # inline references like [12]
    r"|\$.*?\$"  # inline formulas
    r"|[-]{2,}"  # repeated dashes
    r"|[^\w\s\u4e00-\u9fa5.,;!?，。；！？:：'\"()\[\]{}<>/\\-]"  # uncommon symbols
    r"|[ \t]{2,}",
    re.UNICODE,
)


def clean_caption(text: str) -> str:
    cleaned = re.sub(CLEAN_PATTERN, " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def iter_caption_files(root_dir: Path):
    for sample_dir in sorted(path for path in root_dir.iterdir() if path.is_dir()):
        caption_file = sample_dir / "caption.txt"
        if caption_file.exists():
            yield caption_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean caption.txt files under a figure dataset directory.")
    parser.add_argument("--root-dir", required=True, type=Path, help="Root directory containing figure sample folders.")
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix for writing cleaned captions to a new file, for example '_clean'. Defaults to overwriting caption.txt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {args.root_dir}")

    caption_files = list(iter_caption_files(args.root_dir))
    for caption_file in tqdm(caption_files, desc="Cleaning captions", unit="file"):
        raw_text = caption_file.read_text(encoding="utf-8", errors="ignore").strip()
        cleaned_text = clean_caption(raw_text)
        output_file = caption_file if not args.suffix else caption_file.with_name(f"{caption_file.stem}{args.suffix}{caption_file.suffix}")
        output_file.write_text(cleaned_text + "\n", encoding="utf-8")

    print(f"Cleaned {len(caption_files)} caption files under {args.root_dir}")


if __name__ == "__main__":
    main()
