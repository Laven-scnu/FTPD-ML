"""Extract figure images, captions, and textual mentions from markdown files.

The expected input is a directory of markdown files generated from papers. Each
paper may contain markdown image links followed by figure captions. For every
matched figure this script creates:

- the copied image file
- caption.txt
- mention.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


IMAGE_PATTERN = re.compile(r"!\[(.*?)\]\((.*?)\)")
TITLE_PATTERN = re.compile(
    r"^\*?\s*(Figure\s*\d+|Fig\.\s*\d+|FIGURE\s*\d+|FIG\.\s*\d+)(.*)$"
)
FIGURE_NUMBER_PATTERN = re.compile(r"(Figure\s*\d+|Fig\.\s*\d+|FIGURE\s*\d+|FIG\.\s*\d+)")


def _figure_aliases(figure_number: str) -> list[str]:
    number_match = re.search(r"\d+", figure_number)
    if not number_match:
        return [figure_number]
    number = number_match.group()
    return list(dict.fromkeys([figure_number, f"Fig. {number}", f"Fig {number}", f"Figure {number}", f"FIG. {number}", f"FIGURE {number}"]))


def extract_figures_from_markdown(markdown_file: Path, output_root: Path) -> list[dict[str, object]]:
    markdown_file = markdown_file.resolve()
    output_root = output_root.resolve()
    paper_output_dir = output_root / markdown_file.stem
    content = markdown_file.read_text(encoding="utf-8", errors="ignore")
    paragraphs = re.split(r"\n+", content)
    extracted: list[dict[str, object]] = []

    for i, paragraph in enumerate(paragraphs):
        image_match = IMAGE_PATTERN.match(paragraph.strip())
        if not image_match:
            continue

        image_name = image_match.group(1)
        raw_image_path = image_match.group(2)
        image_path = (markdown_file.parent / raw_image_path).resolve()

        title = None
        for next_paragraph in paragraphs[i + 1 :]:
            next_paragraph = next_paragraph.strip()
            if not next_paragraph:
                continue
            title_match = TITLE_PATTERN.match(next_paragraph)
            if title_match:
                title = title_match.group().strip()
            break

        if not title:
            continue

        figure_number_match = FIGURE_NUMBER_PATTERN.search(title)
        figure_number = figure_number_match.group() if figure_number_match else "Unknown"
        aliases = _figure_aliases(figure_number)
        mentions = [p.strip() for p in paragraphs if any(alias in p for alias in aliases)]

        figure_dir = paper_output_dir / figure_number.replace("/", "_")
        figure_dir.mkdir(parents=True, exist_ok=True)

        copied_image_path = None
        if image_path.exists():
            copied_image_path = figure_dir / f"{figure_number}{image_path.suffix}"
            shutil.copy2(image_path, copied_image_path)

        (figure_dir / "caption.txt").write_text(title + "\n", encoding="utf-8")
        (figure_dir / "mention.txt").write_text("\n".join(mentions) + "\n", encoding="utf-8")

        extracted.append(
            {
                "markdown_file": str(markdown_file),
                "image_name": image_name,
                "source_image": str(image_path),
                "copied_image": str(copied_image_path) if copied_image_path else None,
                "figure_number": figure_number,
                "caption": title,
                "mentions": mentions,
            }
        )

    return extracted


def extract_directory(input_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    all_records: list[dict[str, object]] = []
    for markdown_file in sorted(input_dir.rglob("*.md")):
        all_records.extend(extract_figures_from_markdown(markdown_file, output_dir))
    return all_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract figures, captions, and mentions from markdown papers.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing markdown files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for extracted figure folders.")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional JSONL file for extraction metadata.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    records = extract_directory(args.input_dir, args.output_dir)
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        with args.metadata.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Extracted {len(records)} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
