"""
Store config data for all data_prep files centrally
"""

import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
from pydantic import BaseModel
from typing import List, Optional
import logging
from openai import OpenAI

# ------------------ GLOBALS ------------------

load_dotenv()

API_KEY = os.getenv("OPEN_ROUTER_KEY")
if not API_KEY:
    raise ValueError("OPEN_ROUTER_KEY not found in environment")

TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Thinking-2507")

# ------------------ MODELS -------------------
# openrouter id
MODEL_REGISTRY = {
    "google/gemini-2.0-flash-lite-001": {},
    "google/gemini-2.5-flash": {},
    "google/gemini-2.5-pro": {},
    "qwen/qwen-2.5-72b-instruct": {"extra_body": {"provider": {"only": ["DeepInfra"]}}},
}

# ------------------ BOOK CREATION LOGIC ------------------


class BookConfig(BaseModel):
    # operation
    operation_name: str = "book_creation"
    output_dir: str = "./data/json/base"
    debug_dir: str = "./data/debug/book"
    # prompts
    prompt_system: str = "./prompts/world_context/systemmessage.md"
    prompt_instruction: str = "./prompts/world_context/instruction.md"
    # llm / api
    llm: str = "google/gemini-2.5-pro"
    api_base_url: str = "https://openrouter.ai/api/v1"
    api_max_retries: int = 3  # # openai sdk param
    query_retry: int = 2  # retry on response errors


class BookMeta(BaseModel):
    book_id: str
    title: str
    author: str
    word_count: int = 0
    total_chapters: int = 0
    total_scenes: int = 0
    world_context: Optional[str] = None


class Scene(BaseModel):
    scene_id: int = 0
    chapter_index: int
    chapter_title: Optional[str] = None
    instruction: Optional[str] = None
    text: str
    running_summary: Optional[str] = None


class Book(BaseModel):
    meta: BookMeta
    chapters: List[str] = []  # intermediate state
    scenes: List[Scene] = []


class WorldContext(BaseModel):
    """ llm response schema for world context creation """
    tone_style: str  # Era/Genre / Atmosphere / Prose Voice / Sensory Anchors
    world_rules: str  # Tech/Magic Level / Social Order / Key Constraint
    protagonist_conditions: str  # protagonist's position in society, key constraints, motivation
    factions: str  # Ruling Power / Resistance / Third Force
    locations: str  # Location A / Location B / The Outside
    narrative_engine: str  # Central Conflict / Stakes / Thematic Core


# ------------------ SCENE CREATION LOGIC ------------------


class SceneConfig(BaseModel):
    # operation
    operation_name: str = "semantic_scene"
    scene_max_tokens: int = 2900  # max token restraint for target semantic scenes
    chunk_min_tokens: int = 75  # min token restraint any text chunk must fullfil
    atomic_scene_max_tokens: int = 1500  # max token restraint for llm cut atomic scenes
    debug_dir: str = "./data/debug/scene"
    # prompts
    prompt_system: str = "./prompts/scene/systemmessage.md"
    prompt_instruction: str = "./prompts/scene/instruction.md"
    # llm / api
    llm: str = "qwen/qwen-2.5-72b-instruct"
    api_base_url: str = "https://openrouter.ai/api/v1"
    api_max_retries: int = 3  # # openai sdk param
    query_retry: int = 2  # retry on response errors


class SceneStats(BaseModel):
    """ track semantic scene creation stats through the process to create final report at end """
    chunk_amount: int = 0  # total amount of all text chunks across all chapters
    chunk_tokens: int = 0  # total sum of tokens of all text chunks
    atomic_amount: int = 0  # total amount of all llm cut atomic scenes across all chapters
    atomic_tokens: int = 0  # total sum of tokens of all atomic scenes
    semantic_amount: int = 0  # total amount of all semantic scenes across all chapters
    semantic_tokens: int = 0  # total sum of tokens of all semantic scenes
    original_word_count: int = 0  # save for stats before updating it at book meta after processing
    invalid_partitioning: int = 0  # can occur 1x at chapter-wise llm queries


# scene splitting llm response format
class SceneBoundary(BaseModel):
    """ single scene boundary marker """
    final_token_sum: str
    chunk_boundary: int


class ScenePartitioning(BaseModel):
    """ llm response schema for scene partitioning """
    scenes: List[SceneBoundary]


# ------------------ RUNNING SUMMARY CREATION LOGIC ------------------

class SummaryConfig(BaseModel):
    # operation
    operation_name: str = "running_summary"
    max_tokens: int = 400  # token range (final formatted str)
    max_words: int = 200  # word range (raw json dict values summed up (no keys / signs / ...)
    max_words_buffer: int = 40  # allowed overshoot for word range
    debug_dir: str = "./data/debug/summary"
    # prompts
    prompt_system: str = "./prompts/summary/systemmessage.md"
    prompt_instruction: str = "./prompts/summary/instruction_narrative.md"
    prompt_instruction_reference: str = "./prompts/summary/instruction_reference.md"
    # llm / api
    llm: str = "google/gemini-2.0-flash-lite-001"
    max_compress_attempts: int = 3
    query_retry: int = 2  # retry on response errors
    api_base_url: str = "https://openrouter.ai/api/v1"
    api_max_retries: int = 3  # # openai sdk param


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
    # operation
    operation_name: str = "instruction_tuning"
    max_tokens: int = 100  # token range (final formatted str)
    max_words: int = 80
    debug_dir: str = "./data/debug/instruction"
    # prompts used to setup create instruction llm calls
    prompt_system: str = "./prompts/instruction/systemmessage.md"
    prompt_instruction: str = "./prompts/instruction/instruction.md"
    # inference systemmessage to be added as context at create instruction llm calls
    inference_systemmessage: str = "./prompts/inference/systemmessage.md"
    # llm / api
    llm: str = "google/gemini-2.0-flash-lite-001"
    api_base_url: str = "https://openrouter.ai/api/v1"
    api_max_retries: int = 3  # openai sdk param
    query_retry: int = 2  # retry on response errors


class InstructionStats(BaseModel):
    created: int = 0
    total_words: int = 0
    total_tokens: int = 0
    too_large: int = 0


class SceneInstruction(BaseModel):
    scene_goal: str  # Primary event/revelation/decision that must occur
    characters_present: str  # Character 1; Character 2, ....
    emotional_beat: str  # The dominant emotion of the scene
    constraints: str  # Location, time pressure, secrets in play, physical limitations

# ---------------- DATA COMPILING LOGIC -----------------


class CompilerConfig(BaseModel):
    # operation
    operation_name: str = "compiling"
    output_dir: str = "./data/json/final"
    max_tokens: int = 4096  # token limit (final formatted str) per row
    debug_dir: str = "./data/debug/compiling"
    # prompts for compress running summary llm calls
    prompt_system: str = "./prompts/compress/systemmessage.md"
    prompt_instruction: str = "./prompts/compress/instruction.md"
    # inference systemmessage to be added as base to each row
    inference_systemmessage: str = "./prompts/inference/systemmessage.md"
    # llm / api
    llm: str = "google/gemini-2.0-flash-lite-001"
    api_base_url: str = "https://openrouter.ai/api/v1"
    api_max_retries: int = 3  # openai sdk param
    query_retry: int = 2  # retry on response errors


class CompileStats(BaseModel):
    compiled: int = 0
    total_words: int = 0
    total_tokens: int = 0
    too_large: int = 0


# ------------------ BASE LLM CLASS --------------------


class BaseLLM:
    """
    - provides base common init logic / params that all llm classes share to inherit from
    - save task-specific confic class; use these params for setup of other attributes
    - load prompts for systemmessage & instruction
    - save logger
    - setup llm openai client
    """
    def __init__(self, config, logger: logging.Logger):
        self.cfg = config
        with open(self.cfg.prompt_system, mode="r", encoding="utf-8") as f:
            self.prompt_system = f.read()
        with open(self.cfg.prompt_instruction, mode="r", encoding="utf-8") as f:
            self.prompt_instruction = f.read()
        self.logger = logger
        self.client = OpenAI(
            base_url=self.cfg.api_base_url,
            api_key=API_KEY,
            max_retries=self.cfg.api_max_retries
        )


# ------------------ MD CLEANER LOGIC ------------------

class CleanerConfig(BaseModel):
    operation_name: str = "md_cleaning"
    output_dir: str = "./data/md/final/text"
    debug_dir: str = "./data/debug/cleaning"
