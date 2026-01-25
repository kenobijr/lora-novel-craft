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
    debug_dir: str = "./data/debug/"


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
    max_summary_size: int = 400
    prompt_system: str = "./prompts/summary_creation/systemmessage.md"
    prompt_input_format: str = "./prompts/summary_creation/input_format.md"
    prompt_instruction_narrative: str = "./prompts/summary_creation/instruction_narrative.md"
    prompt_instruction_reference: str = "./prompts/summary_creation/instruction_reference.md"


class RunningSummary(BaseModel):
    """ llm response schema for running summary creation """
    SCENE_END_STATE: str  # where/how does this scene physically end
    EMOTIONAL_BEAT: str  # dominant feeling as scene closes
    IMMEDIATE_TENSION: str  # unresolved micro-conflict carrying into next scene
    GLOBAL_EVENTS: str  # cumulative compressed history + new scene merged
    UNRESOLVED_THREADS: str  # 3-5 active plot threads // semicolon separated
    WORLD_STATE: str  # current situation/location/stakes // semicolon separated
    ACTIVE_CHARACTERS: str  # characters in focus with 2-4 word context each
    GLOBAL_SHIFT: str  # what changed - new knowledge/relationships/dangers


def get_root_summary_narrative() -> str:
    """ narrative root summariy as pydantic obj """
    return RunningSummary(
        SCENE_END_STATE="[INITIALIZATION] Story not yet begun.",
        EMOTIONAL_BEAT="[INITIALIZATION] Story not yet begun.",
        IMMEDIATE_TENSION="[INITIALIZATION] Story not yet begun.",
        GLOBAL_EVENTS="[INITIALIZATION] Story not yet begun.",
        UNRESOLVED_THREADS="None",
        WORLD_STATE="[INITIALIZATION] Story not yet begun.",
        ACTIVE_CHARACTERS="None",
        GLOBAL_SHIFT="[INITIALIZATION] Story not yet begun.",
    )


def get_root_summary_reference() -> RunningSummary:
    """ reference root summariy in ready formatted str """

    return RunningSummary(
        local_momentum=LocalMomentum(
            scene_end_state="[INITIALIZATION] Document not yet begun.",
            emotional_beat="[INITIALIZATION] Document not yet begun.",
            immediate_tension="[INITIALIZATION] Document not yet begun."
        ),
        global_state=GlobalState(
            events="The document begins.",
            unresolved_threads=[],
            state="[INITIALIZATION] Document not yet begun.",
            active=[],
            shift="[INITIALIZATION] Document not yet begun."
        )
    )

# ------------------ TOKENIZER ------------------


# tokenizor singleton qwen
_tokenizer = None


def get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Thinking-2507")
    return _tokenizer
