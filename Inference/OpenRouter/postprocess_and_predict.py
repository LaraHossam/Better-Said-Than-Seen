#!/usr/bin/env python3
"""
Postprocess and Predict Pipeline

This script:
1. Converts a model output JSONL to the compact "make-gold" format.
2. Calls the derma-gold CLI bash script (run_derma_gold.sh) in PREDICT mode.
3. Saves the predicted labels in the same folder, ending with "_predicted.jsonl".

Usage:
    from postprocess_and_predict import run_postprocess_and_predict
    run_postprocess_and_predict(input_path="/path/to/input.jsonl", output_file="/path/to/model_output.jsonl")
"""

import os
import subprocess
from scripts.normalize_format import convert_file  
from evaluate import run_evaluation

def run_postprocess_and_predict(input_path: str):
    base, ext = os.path.splitext(input_path)
    output_file =  f"{base}.predicted.jsonl"
    print("🔄 Converting output file to compact JSONL format...")

    # 1️⃣ Convert to the make-gold pipeline format
    converted_path = convert_file(input_path)
    print(f"✅ Converted file saved to: {converted_path}")

    return output_file


if __name__ == "__main__":
    # Example standalone usage
    # You can test directly by running:
    # python postprocess_and_predict.py /path/to/input.jsonl /path/to/output.jsonl
    import sys
    if len(sys.argv) < 2:
        print("Usage: python postprocess_and_predict.py <input_path>")
        sys.exit(1)
    run_postprocess_and_predict(sys.argv[1])
