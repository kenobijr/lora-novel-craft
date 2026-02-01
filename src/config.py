"""
Store config data for all data_prep files centrally
"""

import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
from pydantic import BaseModel
from typing import List, Optional

# ------------------ GLOBALS ------------------

load_dotenv()

API_KEY = os.getenv("OPEN_ROUTER_KEY")
if not API_KEY:
    raise ValueError("OPEN_ROUTER_KEY not found in environment")

TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Thinking-2507")

# ------------------ BOOK / SCENE CREATION LOGIC ------------------


class SceneConfig(BaseModel):
    """ scene creation config params """
    operation_name: str = "semantic_scene"
    max_scene_size: int = 3000
    min_paragraph_size: int = 75
    prompt_system: str = "./prompts/scene_creation/systemmessage.md"
    prompt_input_format: str = "./prompts/scene_creation/input_format.md"
    prompt_instruction: str = "./prompts/scene_creation/instruction.md"
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
    operation_name: str = "running_summary"
    max_tokens: int = 400  # token range (final formatted str)
    max_words: int = 200  # word range (raw json dict values summed up (no keys / signs / ...)
    max_words_buffer: int = 40  # allowed overshoot for word range
    max_compress_attempts: int = 3
    prompt_system: str = "./prompts/summary_creation/systemmessage.md"
    prompt_input_format: str = "./prompts/summary_creation/input_format.md"
    prompt_instruction_narrative: str = "./prompts/summary_creation/instruction_narrative.md"
    prompt_instruction_reference: str = "./prompts/summary_creation/instruction_reference.md"
    debug_dir: str = "./data/debug/summary_creation"
    api_base_url: str = "https://openrouter.ai/api/v1"
    api_max_retries: int = 3


class SummaryStats(BaseModel):
    """ track summary creation stats through the process to create final report at end """
    created: int = 0
    compressed: int = 0
    compress_runs: int = 0
    compressed_successfully: int = 0
    too_large: int = 0
    total_words: int = 0
    total_tokens: int = 0


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
    """ reference root summary as pydantic obj for vocab, foreword, ..."""
    return RunningSummary(
        scene_end_state="[REFERENCE] Establishing foundational material.",
        emotional_beat="[REFERENCE] Setting narrative baseline.",
        immediate_tension="[REFERENCE] No plot tension - knowledge establishment.",
        global_events="[REFERENCE] No narrative events yet.",
        unresolved_threads="None - reference material precedes plot.",
        world_state="[REFERENCE] Pre-narrative context establishment.",
        active_characters="None introduced in narrative yet.",
        global_shift="[REFERENCE] Building foundational knowledge base.",
    )

# ------------------ INSTRUCTION TUNING LOGIC ------------------


class InstructionConfig(BaseModel):
    operation_name: str = "instruction_tuning"
    max_tokens: int = 150  # token range (final formatted str)
    max_words: int = 80
    # prompts to use for create instruction llm calls
    prompt_system: str = "./prompts/instruction_creation/systemmessage.md"
    prompt_input_format: str = "./prompts/instruction_creation/input_format.md"
    prompt_instruction: str = "./prompts/instruction_creation/instruction.md"
    # inference systemmessage to be added as metadata to llm calls
    inference_systemmessage: str = "./prompts/inference/systemmessage.md"
    # url
    api_base_url: str = "https://openrouter.ai/api/v1"
    api_max_retries: int = 3
    # debug
    debug_dir: str = "./data/debug/instruction_creation"


class InstructionStats(BaseModel):
    """ track instruction creation stats through the process to create final report at end """
    created: int = 0
    # compressed: int = 0
    # compress_runs: int = 0
    # compressed_successfully: int = 0
    total_words: int = 0
    total_tokens: int = 0
    too_large: int = 0


class SceneInstruction(BaseModel):
    scene_goal: str  # Primary event/revelation/decision that must occur
    characters_present: str  # Character 1; Character 2, ....
    emotional_beat: str  # The dominant emotion of the scene
    constraints: str  # Location, time pressure, secrets in play, physical limitations
