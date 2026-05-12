import json

input_file = "/home/USER/Documents/DermAgent-Chat/dermagent/outputs/deepinfra/mistralai_Mistral-Small-3.2-24B-Instruct-2506/mistralai_Mistral-Small-3.2-24B-Instruct-2506_ENC00001-ENC00840.predicted.jsonl"
output_file = input_file.replace(".jsonl", "_dedup.jsonl")

seen = set()
total = 0
written = 0
skipped = 0

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for i, line in enumerate(infile, 1):
        line = line.strip()
        if not line:
            continue  # skip blank lines
        total += 1

        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"⚠️ Skipping invalid JSON on line {i}: {e}")
            skipped += 1
            continue

        encounter_id = record.get("encounter_id")
        if not encounter_id:
            print(f"⚠️ No encounter_id found on line {i}")
            skipped += 1
            continue

        if encounter_id not in seen:
            seen.add(encounter_id)
            outfile.write(json.dumps(record) + "\n")
            written += 1

print("✅ Done!")
print(f"📊 Total lines read: {total}")
print(f"📦 Unique encounter_ids written: {written}")
print(f"🚫 Duplicates or invalid skipped: {total - written}")
print(f"📝 Output saved to: {output_file}")
