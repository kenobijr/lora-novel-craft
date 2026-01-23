"""
Store config data for all data_prep files centrally
"""

from dataclasses import dataclass
from transformers import AutoTokenizer


@dataclass
class SceneConfig:
    max_scene_size: int = 3000
    min_paragraph_size: int = 75
    debug_mode: bool = True
    debug_dir: str = "./data/debug/"


# tokenizor singleton qwen
_tokenizer = None


def get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Thinking-2507")
    return _tokenizer
