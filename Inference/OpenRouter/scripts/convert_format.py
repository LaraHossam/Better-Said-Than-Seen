# import json
# import os

# # === CONFIG ===
# INPUT_FILE = "/home/lara.hassan/Documents/DermAgent-Chat/dermagent/outputs/meta-llama_Llama-3.2-11B-Vision-Instruct_ENC00001-ENC00150.jsonl"

# def get_output_path(input_path, suffix="-to-gold", output_dir=None):
#     """
#     Generate a dynamic output file path based on the input file name.
#     If output_dir is given, saves to that directory instead.
#     """
#     base_name = os.path.basename(input_path)
#     name, ext = os.path.splitext(base_name)
#     new_name = f"{name}{suffix}{ext}"

#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#         return os.path.join(output_dir, new_name)
#     else:
#         return os.path.join(os.path.dirname(input_path), new_name)


# def convert_record(record):
#     """
#     Convert a single record from the original format to the target English-only DermAgent format.
#     """
#     image_filename = os.path.basename(record.get("image_id", ""))

#     converted = {
#         "encounter_id": record.get("encounter_id"),
#         "author_id": AUTHOR_ID,
#         "image_ids": [image_filename],
#         "responses": [
#             {
#                 "author_id": record.get("model_id"),
#                 "content_en": (
#                     f"Diagnosis: {record.get('diagnosis', 'N/A')}. "
#                     f"Confidence: {record.get('confidence', 'N/A')}. "
#                     f"Reasoning: {record.get('reasoning', 'N/A')}"
#                 ),
#             }
#         ],
#     }

#     return converted


# def main():
#     output_dir = os.path.join(os.path.dirname(INPUT_FILE), "../processed_outputs")
#     output_path = get_output_path(INPUT_FILE, suffix="-to-gold", output_dir=output_dir)

#     with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
#         for line in fin:
#             if not line.strip():
#                 continue
#             record = json.loads(line)
#             converted = convert_record(record)
#             fout.write(json.dumps(converted, ensure_ascii=False) + "\n")

#     print(f"✅ Conversion complete! Saved to {output_path}")


# if __name__ == "__main__":
#     main()
