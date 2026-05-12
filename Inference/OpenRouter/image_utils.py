import base64
import mimetypes
import os
from typing import List

def guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "image/jpeg"

def encode_image_file_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    mime = guess_mime(path)
    return f"data:{mime};base64,{b64}"

def build_image_parts(image_paths: List[str]) -> List[dict]:
    parts = []
    for p in image_paths:
        if not os.path.exists(p):
            continue
        data_uri = encode_image_file_to_data_uri(p)
        parts.append({
            "type": "image_url",
            "image_url": {"url": data_uri}
        })
    return parts
