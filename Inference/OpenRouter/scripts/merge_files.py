import json
import sys
from pathlib import Path

def merge_jsonl_files(output_path, *input_paths):
    """Merge multiple JSONL files into one output file."""
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for path in input_paths:
            path = Path(path)
            if not path.exists():
                print(f"⚠️ Skipping missing file: {path}")
                continue
            with open(path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    line = line.strip()
                    if line:  # skip empty lines
                        try:
                            json.loads(line)  # validate JSON
                            outfile.write(line + '\n')
                        except json.JSONDecodeError:
                            print(f"⚠️ Skipping invalid JSON line in {path}")
    print(f"✅ Merged {len(input_paths)} files into: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_jsonl.py output.jsonl input1.jsonl input2.jsonl ...")
        sys.exit(1)

    output_file = sys.argv[1]
    input_files = sys.argv[2:]
    merge_jsonl_files(output_file, *input_files)
