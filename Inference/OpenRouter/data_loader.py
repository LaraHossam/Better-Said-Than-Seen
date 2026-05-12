import json
import os
from typing import Dict, Iterable, List, Optional
from dataclasses import dataclass 
@dataclass
class Response:
    author_id: str
    text: str
    lang: str  # "en" | "zh" | "es"
@dataclass
class Encounter:
    encounter_id: str
    author_id: str
    image_ids: List[str]
    query_title: Dict[str, str]
    query_content: Dict[str, str]
    responses: List[Response]

def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def resolve_image_paths(image_ids: List[str], images_root: str) -> List[str]:
    paths = [os.path.join(images_root, img_id) for img_id in image_ids]
    return paths

class DermJSONLDataset:
    """
    Iterates over encounters in your JSONL file.
    Fields we actually pass to the LLM:
      - encounter_id
      - query_title_en
      - query_content_en
      - image_paths 
    """
    def __init__(self, jsonl_path: str, images_root: str, max_images: Optional[int] = None):
        self.jsonl_path = jsonl_path
        self.images_root = images_root
        self.max_images = max_images

    def __iter__(self) -> Iterable[Dict]:
        for obj in read_jsonl(self.jsonl_path):
            image_ids = obj.get("image_ids", [])
            if self.max_images:
                image_ids = image_ids[: self.max_images]
            yield {
                "encounter_id": obj.get("encounter_id"),
                "query_title_en": obj.get("query_title_en") or "",
                "query_content_en": obj.get("query_content_en") or "",
                "image_paths": resolve_image_paths(image_ids, self.images_root),
                "gold_labels": obj.get("gold_labels", []),
            }
