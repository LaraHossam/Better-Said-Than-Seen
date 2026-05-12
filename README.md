# Better Said Than Seen

A dermatology benchmark evaluating whether large vision-language models (LVLMs) reason better from **clinical vignettes** (structured text descriptions) than from **raw images** — and how the two modalities compare when combined.

---

## Repository Structure

```
Better-Said-Than-Seen/
├── DermDetail/
│   ├── Dataset/                        # DermDetail JSONL splits (P1–P7, 8 prompt versions)
│   ├── synthetic_data.ipynb            # Generate synthetic vignette variants
│   └── format_layers_from_results.ipynb
│
├── Vignettes/
│   ├── test_vignettes_*.jsonl              # Pre-generated vignettes per vignette model
│   ├── openrouter_vignette_generation.py   # Generate vignettes from images via LLM
│   ├── openrouter_vignette_inference.py    # Inference on vignettes (text-only)
│   ├── openrouter_vignette_inference_mm.py # Inference on vignettes + images
│   ├── hf_vignette_inference.py            # HuggingFace model inference
│   └── Logs/                               # Inference outputs (auto-created)
│
├── Inference/
│   ├── OpenRouter/
│   │   ├── inference.py                # Main DermDetail inference runner
│   │   ├── postprocess_and_predict.py  # Format normalization (called automatically)
│   │   ├── image_utils.py
│   │   ├── data_loader.py
│   │   ├── prompts.py
│   │   ├── evaluate.py
│   │   └── scripts/                    # Format conversion utilities
│   ├── medgemma.py                     # MedGemma local inference
│   ├── medmo.py                        # MedMo local inference
│   └── Logs/                           # Inference outputs (auto-created)
│
├── ICDLens/
│   ├── generate_codes.py   # Extract ICD-11 codes from model responses
│   ├── eval.py             # Hierarchical ICD-11 evaluation (HDP / HDR / HDF1)
│   └── Logs/               # Coded predictions and eval results (auto-created)
│
├── Attention Analysis/
│   └── medgemma_with_attention.py
│
├── run_derm_detail_eval.sh # End-to-end DermDetail pipeline 
├── run_vignette_eval.sh    # End-to-end vignette pipeline 
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Core dependencies for inference and evaluation: `openai`, `requests`, `tqdm`, `python-dotenv`, `Pillow`, `transformers`, `torch`.

### 2. Environment variables

Create a `.env` file in the repo root — it is never committed:

```bash
# OpenRouter (used for most model inference)
OPENROUTER_API_KEY=sk-or-...

# DeepInfra (optional alternative provider)
DEEPINFRA_TOKEN=...

# WHO ICD-11 API (required for evaluation)
ICD11_CLIENT_ID=...
ICD11_CLIENT_SECRET=...
```

WHO ICD-11 credentials: <https://icd.who.int/icdapi>

### 3. Place images

Encounter images are **not included** in this repository. Download and place them at:

```
Better-Said-Than-Seen/Data/images_train/
```

If your images are stored elsewhere, update the `IMAGES_ROOT` variable near the top of `run_derm_detail_eval.sh`.

---

## Running the Pipelines

### DermDetail — Image-based evaluation

Interactive menu that lets you pick any combination of models and prompt versions:

```bash
./run_derm_detail_eval.sh
```

The script runs three steps automatically for each selection:

1. **Inference** — calls the model on DermDetail images + prompt
2. **Code extraction** — maps free-text responses to ICD-11 codes
3. **Evaluation** — computes hierarchical diagnostic metrics


### Vignette — Text-based evaluation

```bash
./run_vignette_eval.sh --model <openrouter_model_id> [options]
```

**Examples:**

```bash
# Claude reading Gemini-generated vignettes, images excluded
./run_vignette_eval.sh \
  --model          anthropic/claude-haiku-4.5 \
  --vignette-model google_gemini-2.5-flash-lite \
  --vignette-mode  no-image-findings
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | OpenRouter model ID to evaluate |
| `--vignette-model` | `qwen2.5-vl-32b-instruct` | Which pre-generated vignette set to use |
| `--provider` | `openrouter` | `openrouter` or `deepinfra` |
| `--vignette-mode` | `with-image-findings` | `with-image-findings` or `no-image-findings` |

The script runs three steps:

1. **Vignette inference** — model reads the vignette text (and optionally image findings)
2. **Code extraction** — maps responses to ICD-11 codes
3. **Evaluation** — computes hierarchical diagnostic metrics

Pre-generated vignette files live in `Vignettes/`. To generate new ones for a different model:

```bash
python Vignettes/openrouter_vignette_generation.py \
  --input-jsonl DermDetail/Dataset/DermDetail_P1.jsonl \
  --model       google/gemini-2.5-flash-lite-preview-09-2025
```

---

## Running Individual Steps

### Inference only

```bash
python Inference/OpenRouter/inference.py \
  --jsonl               DermDetail/Dataset/DermDetail_P1.jsonl \
  --images-root         Data/images_train \
  --model               openai/gpt-4.1 \
  --output              Inference/Logs \
  --image-mode          single-call \
  --derm-detail-version v1
```

Image modes: `single-call` · `single-call-shuffled` · `text-only` · `images-only`

Output is written to `Inference/Logs/<model>_DermDetail_<version>.jsonl` and the `.converted.jsonl` is created automatically.

### ICD-11 code extraction

```bash
python ICDLens/generate_codes.py \
  --input  Inference/Logs/openai_gpt-4.1_DermDetail_v1.converted.jsonl \
  --outdir ICDLens/Logs
```

### Evaluation

```bash
python ICDLens/eval.py \
  --gt   ICDLens/Logs/train_2.filtered_best_coded.jsonl \
  --pred ICDLens/Logs/openai_gpt-4.1_DermDetail_v1.converted_best_coded.jsonl
```

---

## Evaluation Metrics

Computed by `ICDLens/eval.py` using the WHO ICD-11 hierarchy:

| Metric | Description |
|--------|-------------|
| **HP** | Hierarchical Precision |
| **HR** | Hierarchical Recall |
| **HF1** | Harmonic mean of HDP and HDR |
---
