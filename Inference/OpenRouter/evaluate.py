#!/usr/bin/env python3
"""
evaluate.py

Automatically runs derma-gold CLI (evaluate mode) using the bash script.

Usage:
    python evaluate.py /path/to/predicted.jsonl
"""

import os
import subprocess
import sys


def run_evaluation(predicted_path: str):
    """
    Run the derma-gold evaluation using the bash script.
    Compares model predictions against the gold label file.
    """

    # --- Configuration ---
    bash_path = "/home/USER/Documents/DermAgent-Chat/derma-gold-labels/run_derma_gold.sh"
    gold_path = "/home/USER/Documents/DermAgent-Chat/derma-gold-labels/data/gold_labels_dermo/train_gold_labels_v1.jsonl"

    # --- Validate paths ---
    if not os.path.exists(predicted_path):
        raise FileNotFoundError(f"❌ Predicted file not found: {predicted_path}")
    if not os.path.exists(gold_path):
        raise FileNotFoundError(f"❌ Gold labels file not found: {gold_path}")

    print("📊 Running evaluation...")
    print(f"Gold labels:     {gold_path}")
    print(f"Predicted labels: {predicted_path}")

    # Prepare simulated terminal inputs for the bash script:
    # Mode 2 → Evaluate
    user_input = f"2\n{gold_path}\n{predicted_path}\n"

    # Run the bash script
    subprocess.run(["bash", bash_path], input=user_input, text=True, check=True)

    print("✅ Evaluation completed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py /path/to/predicted.jsonl")
        sys.exit(1)

    predicted_path = sys.argv[1]
    run_evaluation(predicted_path)
