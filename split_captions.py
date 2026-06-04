"""Create single, multiple, and first-sentence caption variants."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


CAPTION_PREFIX_PATTERN = re.compile(
    r"^(?:\*{0,2}(?:FIGURE|Figure|FIG\.|Fig\.)\s*\d+\.?\s*)(.*)$",
    re.IGNORECASE,
)
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?])\s")


def strip_caption_prefix(text: str) -> str | None:
    match = CAPTION_PREFIX_PATTERN.match(text.strip())
    if not match:
        return None
    return match.group(1).lstrip("*").strip()


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(text) if sentence.strip()]


def process_caption_file(caption_file: Path) -> bool:
    content = caption_file.read_text(encoding="utf-8", errors="ignore").strip()
    stripped = strip_caption_prefix(content)
    if not stripped:
        return False

    sentences = split_sentences(stripped)
    if not sentences:
        return False

    if len(sentences) == 1:
        (caption_file.parent / "single_caption.txt").write_text(sentences[0] + "\n", encoding="utf-8")
    else:
        (caption_file.parent / "multiple_caption.txt").write_text(" ".join(sentences) + "\n", encoding="utf-8")

    (caption_file.parent / "first_caption.txt").write_text(sentences[0] + "\n", encoding="utf-8")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split caption.txt into first/single/multiple variants.")
    parser.add_argument("--root-dir", required=True, type=Path, help="Root directory containing figure sample folders.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {args.root_dir}")

    processed = 0
    for sample_dir in sorted(path for path in args.root_dir.iterdir() if path.is_dir()):
        caption_file = sample_dir / "caption.txt"
        if caption_file.exists() and process_caption_file(caption_file):
            processed += 1

    print(f"Processed {processed} caption files under {args.root_dir}")


if __name__ == "__main__":
    main()
