#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────
# PATHS  (all relative to repo root)
# ─────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGES_ROOT="${REPO_ROOT}/Data/images_train"
OUTPUT_DIR="${REPO_ROOT}/Inference/Logs"
DATASET_DIR="${REPO_ROOT}/DermDetail/Dataset"
ICDLENS_DIR="${REPO_ROOT}/ICDLens"
ICDLENS_LOGS="${ICDLENS_DIR}/Logs"
GT_FILE="${ICDLENS_LOGS}/train_2.filtered_best_coded.jsonl"
SCRIPTS_DIR="${REPO_ROOT}/Inference/OpenRouter"
# ─────────────────────────────────────────────

resolve_model() {
  case $1 in
    1)  echo "openai/gpt-4.1" ;;
    2)  echo "openai/gpt-4o" ;;
    3)  echo "anthropic/claude-haiku-4.5" ;;
    4)  echo "google/gemini-2.5-flash-lite-preview-09-2025" ;;
    5)  echo "google/gemini-3-pro-preview" ;;
    6)  echo "qwen/qwen2.5-vl-32b-instruct" ;;
    7)  echo "qwen/qwen3-vl-8b-instruct" ;;
    8)  echo "x-ai/grok-2-vision-1212" ;;
    9)  echo "meta-llama/Llama-3.2-11B-Vision-Instruct" ;;
    10) echo "meta-llama/Llama-3.2-90B-Vision-Instruct" ;;
    11) echo "mistralai/Mistral-Small-3.2-24B-Instruct-2506" ;;
    12) echo "google/gemma-3-4b-it" ;;
    *)  echo "" ;;
  esac
}

resolve_version() {
  case $1 in
    1) echo "v1:train_2.jsonl.v1.jsonl" ;;
    2) echo "v2:train_2.jsonl.v2.jsonl" ;;
    3) echo "v3:train_2.jsonl.v3.jsonl" ;;
    4) echo "v4:train_2.jsonl.v4.jsonl" ;;
    5) echo "v5:train_2.jsonl.v5_correct.jsonl" ;;
    6) echo "v6:train_2.jsonl.v6.jsonl" ;;
    7) echo "v7_close:train_2.jsonl.v5.jsonl_close_v7.jsonl" ;;
    8) echo "v8_far:train_2.jsonl.v5.jsonl_far_v8.jsonl" ;;
    *) echo "" ;;
  esac
}

# ── Run mode selection ────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║        DermDetail Inference Runner       ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "Run mode:"
echo "  [1] Single — pick one model + one version"
echo "  [2] Batch  — pick multiple models + versions"
echo ""
read -p "Enter mode [1-2]: " RUN_MODE

# ── Model menu ────────────────────────────────
echo ""
echo "Available models:"
echo "  [1]  openai/gpt-4.1"
echo "  [2]  openai/gpt-4o"
echo "  [3]  anthropic/claude-haiku-4.5"
echo "  [4]  google/gemini-2.5-flash-lite-preview-09-2025"
echo "  [5]  google/gemini-3-pro-preview"
echo "  [6]  qwen/qwen2.5-vl-32b-instruct"
echo "  [7]  qwen/qwen3-vl-8b-instruct"
echo "  [8]  x-ai/grok-2-vision-1212"
echo "  [9]  meta-llama/Llama-3.2-11B-Vision-Instruct"
echo "  [10] meta-llama/Llama-3.2-90B-Vision-Instruct"
echo "  [11] mistralai/Mistral-Small-3.2-24B-Instruct-2506"
echo "  [12] google/gemma-3-4b-it"
echo ""

if [[ "$RUN_MODE" == "1" ]]; then
  read -p "Enter model number [1-12]: " SELECTED_MODELS
else
  read -p "Enter model numbers separated by spaces (e.g. 1 3 6): " SELECTED_MODELS
fi

# ── Version menu ──────────────────────────────
echo ""
echo "Available versions:"
echo "  [1] v1  — minimal"
echo "  [2] v2  — short"
echo "  [3] v3  — medium"
echo "  [4] v4  — detailed"
echo "  [5] v5  — v5_correct"
echo "  [6] v6  — longest"
echo "  [7] v7  — close variant"
echo "  [8] v8  — far variant"
echo ""

if [[ "$RUN_MODE" == "1" ]]; then
  read -p "Enter version number [1-8]: " SELECTED_VERSIONS
else
  read -p "Enter version numbers separated by spaces (e.g. 1 2 3): " SELECTED_VERSIONS
fi

# ── Build jobs temp file ──────────────────────
JOBS_FILE=$(mktemp /tmp/derm_jobs.XXXXXX)
TOTAL=0

for mc in $SELECTED_MODELS; do
  for vc in $SELECTED_VERSIONS; do
    MODEL=$(resolve_model "$mc")
    VERSION_ENTRY=$(resolve_version "$vc")
    if [[ -z "$MODEL" || -z "$VERSION_ENTRY" ]]; then
      echo "⚠️  Skipping invalid selection: model=$mc version=$vc"
      continue
    fi
    echo "${MODEL}|${VERSION_ENTRY}" >> "$JOBS_FILE"
    TOTAL=$((TOTAL + 1))
  done
done

# ── Confirm ───────────────────────────────────
echo ""
echo "─────────────────────────────────────────"
echo "  Jobs queued: $TOTAL"
while IFS='|' read -r m ve; do
  v="${ve%%:*}"
  echo "    • $m  ×  $v"
done < "$JOBS_FILE"
echo "─────────────────────────────────────────"
read -p "Proceed? [y/N]: " CONFIRM
if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
  rm -f "$JOBS_FILE"
  echo "Aborted."; exit 0
fi

mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_LOG="${OUTPUT_DIR}/batch_results_${TIMESTAMP}.txt"
echo "DermDetail Batch Run — $(date)" > "$BATCH_LOG"
echo "========================================" >> "$BATCH_LOG"

PASSED=0
FAILED=0

# ── Run all jobs ──────────────────────────────
while IFS='|' read -r MODEL VERSION_ENTRY; do
  VERSION="${VERSION_ENTRY%%:*}"
  JSONL_FILE="${VERSION_ENTRY##*:}"

  JSONL_PATH="${DATASET_DIR}/${JSONL_FILE}"
  SAFE_MODEL=$(echo "$MODEL" | tr '/' '_')
  CONVERTED_OUTPUT="${OUTPUT_DIR}/${SAFE_MODEL}_DermDetail_${VERSION}.converted.jsonl"
  CODED_OUTPUT="${ICDLENS_LOGS}/${SAFE_MODEL}_DermDetail_${VERSION}.converted_best_coded.jsonl"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  ▶ Model:   $MODEL"
  echo "  ▶ Version: $VERSION"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if python "${SCRIPTS_DIR}/inference.py" \
      --jsonl               "$JSONL_PATH" \
      --images-root         "$IMAGES_ROOT" \
      --model               "$MODEL" \
      --output              "$OUTPUT_DIR" \
      --image-mode          single-call \
      --derm-detail-version "$VERSION" \
    && python "${ICDLENS_DIR}/generate_codes.py" \
      --input  "$CONVERTED_OUTPUT" \
      --outdir "$ICDLENS_LOGS" \
    && EVAL_OUT=$(python "${ICDLENS_DIR}/eval.py" \
      --gt   "$GT_FILE" \
      --pred "$CODED_OUTPUT" 2>&1); then

    echo "$EVAL_OUT"
    {
      echo ""
      echo "Model:   $MODEL"
      echo "Version: $VERSION"
      echo "--- eval output ---"
      echo "$EVAL_OUT"
      echo "-------------------"
    } >> "$BATCH_LOG"
    echo "✅ Done: $MODEL × $VERSION"
    PASSED=$((PASSED + 1))

  else
    echo "❌ FAILED: $MODEL × $VERSION" | tee -a "$BATCH_LOG"
    FAILED=$((FAILED + 1))
  fi

done < "$JOBS_FILE"

rm -f "$JOBS_FILE"

# ── Final summary ─────────────────────────────
{
  echo ""
  echo "========================================"
  echo "SUMMARY: $PASSED passed, $FAILED failed out of $TOTAL jobs"
  echo "Finished: $(date)"
} >> "$BATCH_LOG"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║              Batch complete ✅           ║"
echo "╚══════════════════════════════════════════╝"
echo "  Total:  $TOTAL jobs"
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"
echo "  Log:    $BATCH_LOG"