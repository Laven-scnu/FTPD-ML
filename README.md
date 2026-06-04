# FTPD-CL Code

This directory contains the current public-facing code for figure-text extraction, data cleaning, image-text matching, and captioning experiments used in the FTPD-CL project.

## Current Structure

```text
code/
|-- README.md
|-- requirements.txt
|-- FigureProcessing/
|   |-- extract_figures.py
|   |-- clean_captions.py
|   |-- split_captions.py
|   `-- flatten_figure_folders.py
|-- ImageMatch/
|   `-- train_clip.py
`-- ImageCaption/
    |-- deepseek/
    |   |-- finetune.sh
    |   `-- infer_captions.py
    |-- InternVL/
    |   |-- finetune.sh
    |   `-- infer_captions.py
    `-- BLIP/
```

## Main Workflow

- `FigureProcessing/extract_figures.py`: extract figure images, captions, and figure mentions from markdown files.
- `FigureProcessing/clean_captions.py`: clean noisy `caption.txt` files with a reusable regex-based pipeline.
- `FigureProcessing/split_captions.py`: generate `single_caption.txt`, `multiple_caption.txt`, and `first_caption.txt`.
- `FigureProcessing/flatten_figure_folders.py`: flatten `paper/figure` folders into a single figure-level dataset directory.
- `ImageMatch/train_clip.py`: fine-tune CLIP on image-caption pairs.
- `ImageCaption/deepseek/`: DeepSeek-VL LoRA fine-tuning and inference scripts.
- `ImageCaption/InternVL/`: InternVL LoRA fine-tuning and inference scripts.

## Auxiliary Material

- `ImageCaption/BLIP/`: a third-party BLIP code copy kept for reference. For a cleaner public release, it is usually better to reference the upstream repository instead of vendoring the full codebase unless you intentionally keep it with proper license attribution.

## Environment

Python 3.9+ is recommended. Install the core dependencies as needed:

```bash
pip install -r requirements.txt
```

Large model dependencies differ by experiment. DeepSeek-VL, InternVL, ModelScope, Swift, PEFT, and CUDA versions should be installed according to the model providers' official instructions.

## Data Layout

The CLIP fine-tuning script expects one folder per figure:

```text
dataset_root/
  sample_000001/
    figure.png
    caption.txt
    mention.txt
  sample_000002/
    figure.jpg
    caption.txt
```

Captioning scripts expect JSON records like:

```json
[
  {
    "images": ["dataset_root/sample_000001/figure.png"],
    "caption": "Figure caption text."
  }
]
```

## Figure Extraction and Cleaning

Extract figures and captions from markdown files:

```bash
python FigureProcessing/extract_figures.py \
  --input-dir data/markdown_papers \
  --output-dir data/extracted_figures \
  --metadata outputs/extraction_metadata.jsonl
```

The script searches markdown image links followed by captions beginning with `Figure`, `Fig.`, `FIGURE`, or `FIG.`. It writes copied images, `caption.txt`, and `mention.txt` files under the output directory.

Clean extracted captions:

```bash
python FigureProcessing/clean_captions.py \
  --root-dir data/extracted_figures
```

Create caption variants for downstream experiments:

```bash
python FigureProcessing/split_captions.py \
  --root-dir data/extracted_figures
```

Flatten `paper/figure` folders into a single figure-level dataset:

```bash
python FigureProcessing/flatten_figure_folders.py \
  --source-dir data/extracted_figures \
  --target-dir data/flat_figures
```

## CLIP Image-Text Matching

Fine-tune CLIP:

```bash
python ImageMatch/train_clip.py \
  --data-root data/extracted_figures \
  --output-dir outputs/clip \
  --batch-size 16 \
  --epochs 20
```

For a smoke test:

```bash
python ImageMatch/train_clip.py \
  --data-root data/extracted_figures \
  --output-dir outputs/clip_test \
  --max-samples 128 \
  --epochs 1
```

## Caption Model Fine-Tuning

DeepSeek-VL with Swift:

```bash
MODEL_PATH=/path/to/deepseek-vl-7b-chat \
DATASET_PATH=data/train.jsonl \
OUTPUT_DIR=outputs/deepseek_lora \
bash ImageCaption/deepseek/finetune.sh
```

InternVL with Swift:

```bash
MODEL_PATH=/path/to/InternVL2_5 \
DATASET_PATH=data/train.jsonl \
OUTPUT_DIR=outputs/internvl_lora \
bash ImageCaption/InternVL/finetune.sh
```

## Caption Generation

DeepSeek-VL inference:

```bash
python ImageCaption/deepseek/infer_captions.py \
  --model-path /path/to/deepseek-vl-7b-chat \
  --lora-checkpoint outputs/deepseek_lora/checkpoint-xxx \
  --input-json data/test.json \
  --output-csv outputs/deepseek_predictions.csv
```

InternVL inference:

```bash
python ImageCaption/InternVL/infer_captions.py \
  --model-path /path/to/InternVL2_5 \
  --lora-checkpoint outputs/internvl_lora/checkpoint-xxx \
  --input-json data/test.json \
  --output-csv outputs/internvl_predictions.csv
```

Omit `--lora-checkpoint` to run the base model.

## Notes for GitHub Release

- Do not commit PDFs, extracted images, trained checkpoints, API keys, or local logs.
- Replace absolute paths with command-line arguments or environment variables.
- Add citations and license notices for third-party models and code.
- If the dataset itself is released separately, include only small example records in this repository.
