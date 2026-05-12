#!/bin/bash
# Usage: ./run_vignette_eval.sh --model <model> [--vignette-model <vignette_model>]
# Example:
#   ./run_vignette_eval.sh --model qwen/qwen2.5-vl-32b-instruct

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=""
VIGNETTE_MODEL="qwen2.5-vl-32b-instruct"
PROVIDER="openrouter"
VIGNETTE_MODE="with-image-findings"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)           MODEL="$2";          shift 2 ;;
    --vignette-model)  VIGNETTE_MODEL="$2"; shift 2 ;;
    --provider)        PROVIDER="$2";       shift 2 ;;
    --vignette-mode)   VIGNETTE_MODE="$2";  shift 2 ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 --model <model> [--vignette-model <vignette_model>]"
      exit 1
      ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "Error: --model is required."
  echo "Usage: $0 --model <model> [--vignette-model <vignette_model>]"
  exit 1
fi

MODEL_SLUG="${MODEL//\//_}"
VIGNETTE_FILENAME="${MODEL_SLUG}_${PROVIDER}_inference_vignettes_plus_images_by_${VIGNETTE_MODEL}.jsonl"
BEST_CODED_FILENAME="${MODEL_SLUG}_${PROVIDER}_inference_vignettes_plus_images_by_${VIGNETTE_MODEL}_best_coded.jsonl"

INPUT_JSONL="${REPO_ROOT}/Vignettes/test_vignettes_${VIGNETTE_MODEL}.jsonl"
VIGNETTE_OUTPUT="${REPO_ROOT}/Vignettes/Logs/${VIGNETTE_MODEL}/${VIGNETTE_FILENAME}"
ICD_PRED="${REPO_ROOT}/ICDLens/Logs/${BEST_CODED_FILENAME}"
ICD_GT="${REPO_ROOT}/ICDLens/Logs/test_best_coded.jsonl"

echo "============================================="
echo "  Vignette Evaluation Pipeline"
echo "============================================="
echo "  Model:          $MODEL"
echo "  Vignette model: $VIGNETTE_MODEL"
echo "  Provider:       $PROVIDER"
echo "---------------------------------------------"
echo "  Step 1 input:   $INPUT_JSONL"
echo "  Step 1 output:  $VIGNETTE_OUTPUT"
echo "  Step 2 output:  $ICD_PRED"
echo "  Step 3 GT:      $ICD_GT"
echo "============================================="
echo ""

echo "[1/3] Running vignette inference..."
python "${REPO_ROOT}/Vignettes/openrouter_vignette_inference.py" \
  --input-jsonl    "$INPUT_JSONL" \
  --provider       "$PROVIDER" \
  --model          "$MODEL" \
  --vignette-mode  "$VIGNETTE_MODE" \
  --vignette-model "$VIGNETTE_MODEL" \
  --print-prompt
echo "[1/3] Done. Output: $VIGNETTE_OUTPUT"
echo ""

echo "[2/3] Generating ICD codes..."
python "${REPO_ROOT}/ICDLens/generate_codes.py" \
  --input   "$VIGNETTE_OUTPUT" \
  --outdir  "${REPO_ROOT}/ICDLens/Logs"
echo "[2/3] Done. Output: $ICD_PRED"
echo ""

echo "[3/3] Running evaluation..."
python "${REPO_ROOT}/ICDLens/eval.py" \
  --gt   "$ICD_GT" \
  --pred "$ICD_PRED"
echo "[3/3] Done."
echo ""
echo "Pipeline complete!"
