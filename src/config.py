"""
Store config data for all data_prep files centrally
"""

from transformers import AutoTokenizer
from pydantic import BaseModel
from typing import List, Optional

# ------------------ BOOK / SCENE CREATION LOGIC ------------------


class SceneConfig(BaseModel):
    """ scene creation config params """
    max_scene_size: int = 3000
    min_paragraph_size: int = 75
    prompt_system: str = "./prompts/scene_splitting/systemmessage.md"
    prompt_input_format: str = "./prompts/scene_splitting/input_format.md"
    prompt_instruction: str = "./prompts/scene_splitting/instruction.md"
    debug_mode: bool = True
    debug_dir: str = "./data/debug/scene_creation"


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


# ------------------ RUNNING SUMMARY CREATION LOGIC ------------------

class SummaryConfig(BaseModel):
    """ running summary creation config params """
    max_tokens: int = 400  # refers to final formatted str
    max_words: int = 200  # refers to raw json dict values summed up (no keys / signs / ...)
    max_retry: int = 2
    prompt_system: str = "./prompts/summary_creation/systemmessage.md"
    prompt_input_format: str = "./prompts/summary_creation/input_format.md"
    prompt_instruction_narrative: str = "./prompts/summary_creation/instruction_narrative.md"
    prompt_instruction_reference: str = "./prompts/summary_creation/instruction_reference.md"
    debug_dir: str = "./data/debug/summary_creation"


class RunningSummary(BaseModel):
    """ llm response schema for running summary creation """
    scene_end_state: str  # where/how does this scene physically end
    emotional_beat: str  # dominant feeling as scene closes
    immediate_tension: str  # unresolved micro-conflict carrying into next scene
    global_events: str  # cumulative compressed history + new scene merged
    unresolved_threads: str  # 3-5 active plot threads // semicolon separated
    world_state: str  # current situation/location/stakes // semicolon separated
    active_characters: str  # characters in focus with 2-4 word context each
    global_shift: str  # what changed - new knowledge/relationships/dangers


def get_root_summary_narrative() -> RunningSummary:
    """ narrative root summary as pydantic obj """
    return RunningSummary(
        scene_end_state="[INITIALIZATION] Story not yet begun.",
        emotional_beat="[INITIALIZATION] Story not yet begun.",
        immediate_tension="[INITIALIZATION] Story not yet begun.",
        global_events="[INITIALIZATION] Story not yet begun.",
        unresolved_threads="None",
        world_state="[INITIALIZATION] Story not yet begun.",
        active_characters="None",
        global_shift="[INITIALIZATION] Story not yet begun.",
    )


def get_root_summary_reference() -> RunningSummary:
    """ reference root summary as pydantic obj - placeholder for future implementation """
    # TODO: implement reference summary logic when special reference content is added
    return get_root_summary_narrative()

# ------------------ TOKENIZER ------------------


# tokenizor singleton qwen
_tokenizer = None


def get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Thinking-2507")
    return _tokenizer
