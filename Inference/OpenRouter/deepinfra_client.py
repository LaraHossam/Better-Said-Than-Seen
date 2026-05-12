import os
import json
import re
from typing import List, Optional, Tuple
from openai import OpenAI

# DEEPPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"  
DEEPPINFRA_BASE_URL = "https://openrouter.ai/api/v1"  

def get_api_key() -> Optional[str]:
    return os.getenv("DEEPINFRA_TOKEN") or os.getenv("DEEPINFRA_API_KEY")

class DeepInfraClient:
    """
    Thin wrapper around the OpenAI-compatible API on DeepInfra for:
    - multimodal chat.completions (vision models)
    - text-only chat.completions
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = DEEPPINFRA_BASE_URL):
        api_key = api_key or get_api_key()
        if not api_key:
            raise RuntimeError("Missing DEEPINFRA_TOKEN (or DEEPINFRA_API_KEY).")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat_multimodal(
        self,
        model: str,
        system_prompt: str,
        image_parts,
        user_text: str,
        temperature: float = 0.1,
        max_output_tokens: int = 512,
        mock: bool = False,
        print_bool: bool = False,
    ):
        """
        Run a multimodal chat request on DeepInfra/OpenAI-compatible API.
        If image_parts is empty/None, falls back to TEXT-ONLY format automatically.
        """

        has_images = bool(image_parts)  # False for None or []

        if has_images:
            # Multimodal: user content is a list of parts (images + text)
            user_content = [*image_parts, {"type": "text", "text": user_text}]
        else:
            # Text-only: user content must be a plain string for many providers
            user_content = user_text

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # --- MOCK MODE -------------------------------------------------
        if mock:
            print("🧪 Mock mode enabled — not calling API.")
            if print_bool:
                import json
                print("Messages sent to model:")
                print(json.dumps(messages, indent=2))

            fake_response = {
                "id": "mock-123",
                "model": model,
                "usage": {
                    "prompt_tokens": 125,
                    "completion_tokens": 42,
                    "total_tokens": 167
                },
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "This is a mock model reply ✅ — looks like a benign skin lesion."
                        }
                    }
                ],
                "latency_ms": 183,
            }
            return fake_response

        # --- REAL API CALL ---------------------------------------------
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )

        if print_bool:
            print("DeepInfra client response:", resp)

        return resp


    @staticmethod
    def extract_json(s: str) -> dict:
        """
        Robustly pull the first JSON object from a model response.
        """
        # Find the first {...} block
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in model response.")
        snippet = m.group(0)
        return json.loads(snippet)
