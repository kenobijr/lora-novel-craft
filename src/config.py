"""
Store config data for all data_prep files centrally
"""

from transformers import AutoTokenizer
from pydantic import BaseModel
from typing import List, Optional


# scene config params
class SceneConfig(BaseModel):
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

# book json structure including scenes


class BookMeta(BaseModel):
    book_id: str
    title: str
    author: str
    word_count: int = 0
    total_chapters: int = 0
    total_scenes: int = 0
    world_context: Optional[str] = None


class Scene(BaseModel):
    scene_id: int
    chapter_index: int
    chapter_title: Optional[str] = None
    instruction: Optional[str] = None
    text: str
    running_summary: Optional[str] = None


class Book(BaseModel):
    meta: BookMeta
    scenes: List[Scene]


# scene splitting llm response format
class SceneBoundary(BaseModel):
    """ single scene boundary marker """
    final_token_sum: str
    end_paragraph: int


class ScenePartitioning(BaseModel):
    """ llm response schema for scene partitioning """
    scenes: List[SceneBoundary]
