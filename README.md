# FTPD-ML

Public code for figure-text extraction, article metadata processing, image-text matching, and image captioning experiments in the FTPD-ML project.

## Repository Structure

```text
FTPD-CL-main/
|-- ArticleProcess/
|-- FigureProcessing/
|-- ImageMatch/
|-- ImageCaption/
|-- requirements.txt
`-- README.md
```

## Installation

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The model-specific captioning experiments may require additional packages and CUDA versions. Install those dependencies according to the model project's official instructions.

## Article Processing

`ArticleProcess` contains two independent DOI enrichment tools:

- `query_crossref.py` detects retraction-related Crossref metadata.
- `query_unpaywall.py` queries open-access status and license metadata.

Both tools accept an input CSV through `--input`, write an output CSV through `--output`, preserve duplicate input rows, and use a SQLite checkpoint by default.

### Input CSV

The only required column is `doi`. `markdown_name` is optional and is copied to the output.

```csv
doi,markdown_name
10.1038/s41586-020-2649-2,paper_0001.md
10.1126/science.aar2131,paper_0002.md
```

### Crossref Retraction Check

```bash
python ArticleProcess/query_crossref.py \
  --input data/doi_tasks.csv \
  --output outputs/crossref_retraction.csv \
  --email you@example.org
```

Optional controls include `--workers`, `--timeout`, `--retries`, `--backoff`, `--progress-db`, and `--retry-errors`.

Output columns:

```text
doi,markdown_name,is_retracted,retraction_reason,error
```

The detector checks Crossref update records, title prefixes, Crossmark assertions, article type, and retraction-related relations. It is a metadata heuristic and should be manually verified before making publication or legal decisions.

### Unpaywall OA and License Check

Unpaywall requires a contact email. Pass it explicitly or set `UNPAYWALL_EMAIL`.

```bash
# Linux/macOS:
export UNPAYWALL_EMAIL=you@example.org

# Windows PowerShell:
# $env:UNPAYWALL_EMAIL = "you@example.org"

python ArticleProcess/query_unpaywall.py \
  --input data/doi_tasks.csv \
  --output outputs/unpaywall.csv \
  --workers 4
```

Output columns:

```text
doi,markdown_name,is_oa,oa_status,oa_url,license_raw,license_category,error
```

`license_category` is normalized to values such as `CC0`, `CC-BY`, `CC-BY-NC`, `CC-BY-NC-SA`, `CC-BY-ND`, `UNKNOWN`, or `OTHER`.

### Checkpoint and Resume

If `--progress-db` is omitted, the tool creates a SQLite file beside the output, for example `outputs/crossref_retraction.sqlite`. Re-run the same command to continue. Use `--retry-errors` to query rows that previously ended with an error.

## Figure Processing

Extract figures and captions from Markdown:

```bash
python FigureProcessing/extract_figures.py \
  --input-dir data/markdown_papers \
  --output-dir data/extracted_figures \
  --metadata outputs/extraction_metadata.jsonl
```

Clean and split captions:

```bash
python FigureProcessing/clean_captions.py --root-dir data/extracted_figures
python FigureProcessing/split_captions.py --root-dir data/extracted_figures
```

Flatten figure folders when needed:

```bash
python FigureProcessing/flatten_figure_folders.py \
  --source-dir data/extracted_figures \
  --target-dir data/flat_figures
```

## Image-Text Matching and Captioning

The CLIP training script expects one folder per figure:

```text
dataset_root/
`-- sample_000001/
    |-- figure.png
    |-- caption.txt
    `-- mention.txt
```

```bash
python ImageMatch/train_clip.py \
  --data-root data/extracted_figures \
  --output-dir outputs/clip \
  --epochs 20
```

DeepSeek-VL and InternVL scripts are under `ImageCaption/`. Their model, LoRA, Swift, and CUDA dependencies are experiment-specific.

## Reproducibility and Data Policy

Do not commit PDFs, extracted image collections, trained checkpoints, API keys, local SQLite databases, or logs. Use a small example CSV for documentation and keep research data in a separate location.

Crossref and Unpaywall are external services. Follow their current terms, rate limits, and attribution requirements. Third-party code and model directories retain their own licenses; check those licenses before redistribution.
