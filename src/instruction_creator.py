import os
import sys
import json
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from src.utils import parse_scene_range
from src.config import get_tokenizer, InstructionConfig, Book
from typing import Tuple

# llm model = openrouter id
LLM = "google/gemini-2.0-flash-lite-001"
# "google/gemini-2.5-pro"
# "qwen/qwen-2.5-72b-instruct"
# "google/gemini-2.0-flash-lite-001"

# load api key
load_dotenv()
api_key = os.getenv("OPEN_ROUTER_KEY")
if not api_key:
    raise ValueError("could not load API key...")

# load tokenizer
tokenizer = get_tokenizer()
if not tokenizer:
    raise ValueError("could not load tokenizer...")


class InstructionCreatorLLM:
    def __init__(self, config: InstructionConfig, world_context: str):
        self.cfg = config
        self.wc = world_context
        # load prompts to use at llm call
        with open(self.cfg.prompt_system, mode="r", encoding="utf-8") as f:
            self.prompt_system = f.read()
        with open(self.cfg.prompt_input_format, mode="r", encoding="utf-8") as f:
            self.prompt_input = f.read()
        with open(self.cfg.prompt_instruction, mode="r", encoding="utf-8") as f:
            self.prompt_instruction = f.read()
        # load inference systemmessage to add it as metadata content to prompt
        with open(self.cfg.inference_systemmessage, mode="r", encoding="utf-8") as f:
            self.inference_systemmessage = f.read()
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.cfg.api_base_url,
            max_retries=self.cfg.api_max_retries,
        )
        # init stats obj
        # init logger


class InstructionProcessor:
    def __init__(self, book_json_path: str, config=None):
        self.cfg = config if config is not None else InstructionConfig()
        self.book_json_path = book_json_path
        with open(book_json_path, mode="r", encoding="utf-8") as f:
            self.book_content = Book(**json.load(f))
        self.llm = InstructionCreatorLLM(
            self.cfg,
            self.book_content.meta.world_context
        )
        # stats object
        # logger

    def _process_scenes(self, scene_range):
        for i in range(scene_range[0], scene_range[1]):
            print(self.book_content.scenes[i].scene_id)

    def run(self, scene_range: Tuple[int, int]):
        len_scenes = len(self.book_content.scenes)
        if scene_range is None:
            scene_range = (0, len_scenes)
        else:
            if scene_range[0] < 0:
                raise ValueError("start must be >= 0")
            if scene_range[1] > len_scenes:
                raise ValueError(f"end must be <= {len_scenes}")
            if scene_range[0] >= scene_range[1]:
                raise ValueError("start must be < end")
        self._process_scenes(scene_range)


def main():
    """
    cli entry point for instruction creation on book json
    - default: instruction is created for each scene in book json scene list
    - provide optional scene range arg to create instructions for only certain range of scenes
    - "Usage: python instruction_creator.py <input_book.json> <start,end>"
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "book_path",
        help="path to book json file",
    )
    parser.add_argument(
        "scene_range",
        nargs="?",
        type=parse_scene_range,
        help="optional range as start,end (e.g. 0,10)",
    )
    args = parser.parse_args()
    p = InstructionProcessor(args.book_path)
    p.run(args.scene_range)


if __name__ == "__main__":
    main()
