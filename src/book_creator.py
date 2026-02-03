"""
Convert .md book into base .json book file, split text along chapters and create World Context file
- input .md file:
    - all chapter headers must be like "# Chapter 1" or "Chapter 1: Some Title" as content anchors
- book_id arg:
    - must contain the name of the author followed by the title of the book (sep. by "-")
    - all smallcaps and with _, e.g.: "iron_heel_london" for The Iron Heel by Jack London
- reference .md file (optional):
    - provide additional reference content to be used for world context creation
- world context creation:
    - query llm to compress narrative (+ optional ref material) into "world constitution"
    - json response schema enforcement
"""

import argparse
from src.config import BookConfig, Book, BookMeta, API_KEY, TOKENIZER, WorldContext
from src.utils import init_logger
import logging
import os
import re
import json
from openai import OpenAI
from typing import Dict

# llm model = openrouter id
LLM = "google/gemini-2.5-flash"
# "google/gemini-2.5-pro"
# "qwen/qwen-2.5-72b-instruct"
# "google/gemini-2.0-flash-lite-001"
# "google/gemini-2.5-flash"


class WorldContextLLM:
    def __init__(
        self,
        config: BookConfig,
        book_content: str,
        book_reference: str | None,
        logger: logging.Logger,
    ):
        self.cfg = config
        self.book_content = book_content
        self.book_reference = book_reference
        # load prompts
        with open(self.cfg.prompt_system, mode="r", encoding="utf-8") as f:
            self.prompt_system = f.read()
        with open(self.cfg.prompt_input_format, mode="r", encoding="utf-8") as f:
            self.prompt_input = f.read()
        with open(self.cfg.prompt_instruction, mode="r", encoding="utf-8") as f:
            self.prompt_instruction = f.read()
        self.logger = logger
        self.client = OpenAI(
            base_url=self.cfg.api_base_url,
            api_key=API_KEY,
            max_retries=self.cfg.api_max_retries
        )

    def _construct_prompt(self):
        """ add reference material content to prompt if available """
        ref_block = ""
        if self.book_reference is not None:
            ref_block = f"""
<reference_material>
{self.book_reference}
</reference_material>
"""

        return f"""
<system>
{self.prompt_system}
</system>

<input_description>
{self.prompt_input}
</input_description>

<novel>
{self.book_content}
</novel>
{ref_block}
<instruction>
{self.prompt_instruction}
</instruction>
"""

    def create_world_context(self):
        prompt = self._construct_prompt()

        for attempt in range(self.cfg.json_parse_retries):
            # log full prompt to logfile before llm query
            self.logger.debug(
                f"\n=== WORLD CONTEXT CREATION: PROMPT START ===\n"
                f"{prompt}\n"
                f"=== WORLD CONTEXT CREATION: PROMPT END ==="
            )
            response = self.client.chat.completions.create(
                model=LLM,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "world_context_creation",
                        "strict": True,
                        "schema": WorldContext.model_json_schema()
                    }
                },
                # # only needed for qwen3!!!
                # extra_body={
                #     "provider": {
                #         "only": ["DeepInfra"]
                #     }
                # }
            )
            # grab content in raw json for logging
            result_content = response.choices[0].message.content
            # log llm response before parsing / formatting
            self.logger.debug(
                f"\n=== WORLD CONTEXT CREATION: RESPONSE START ===\n"
                f"{result_content}\n"
                f"=== WORLD CONTEXT CREATION: RESPONSE END ==="
            )
            # catch json deserialize error
            try:
                # parse into python dict rep & count words
                result = json.loads(result_content)
                break
            # at this certain error try 1x again with same prompt & log warning; next time: crash it
            except json.JSONDecodeError:
                if attempt == 0:
                    self.logger.warning("Invalid JSON response at world context creation")
                    continue
                raise
        # count words from LLM response (dict values) & update stats / logs
        total_words = sum(len(str(v).split()) for v in result.values())
        self.logger.info(f"World Context: LLM response amount words: {total_words}")
        return result


class BookProcessor:
    def __init__(self, input_book_path: str, book_id: str, input_ref_path: str, config=None):
        self.cfg = config if config is not None else BookConfig()
        self.input_book_path = input_book_path
        # unique identifier for book
        self.book_id = book_id
        with open(input_book_path, mode="r", encoding="utf-8") as f:
            self.book_content = f.read()
        # optional ref material path -> if available, pass to llm wc creation
        self.book_reference = None
        if input_ref_path is not None:
            with open(input_ref_path, mode="r", encoding="utf-8") as f:
                self.book_reference = f.read()
        # output file path to save json
        self.book_name = os.path.basename(self.input_book_path).removesuffix(".md")
        self.book_json_path = os.path.join(self.cfg.output_dir, f"{self.book_name}.json")
        self.logger = init_logger(self.cfg.operation_name, self.cfg.debug_dir, self.book_name)
        self.book = None
        # setup llm
        self.llm = WorldContextLLM(
            self.cfg,
            self.book_content,
            self.book_reference,  # str | None
            self.logger,
        )

    @staticmethod
    def _format_world_context(response: Dict) -> str:
        """ bring llm response into target str format """
        lines = ["# Novel World Context\n"]
        for key, value in response.items():
            # transform snake_case key to markdown header: scene_end_state -> ## SCENE END STATE
            header = "## " + key.upper().replace("_", " ")
            lines.append(f"{header}\n{value}")
        return "\n\n".join(lines)

    def _process_world_context(self):
        """ steer world context creation with llm call and save to json """
        try:
            wc = self.llm.create_world_context()
        except Exception:
            self.logger.exception("Failed process creating world context...")
            raise
        # bring response into target .md format, count tokens and log
        wc = self._format_world_context(wc)
        amount_tokens = len(TOKENIZER.encode(wc))
        self.logger.info(f"World Context: Total amount tokens in final format: {amount_tokens}")
        # save & write to json
        self.book.meta.world_context = wc
        with open(self.book_json_path, mode="w", encoding="utf-8") as f:
            json.dump(self.book.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
        self.logger.info(f"World context saved to file: {self.book_json_path}")
        self.logger.info("---------------------------------------------")

    def _create_book(self):
        """
        - construct book object with bookmeta and author, title and book_id
        - all other attributes updated after chapter processing
        """
        author = self.book_name.split("-")[0].replace("_", " ")
        title = self.book_name.split("-")[1].replace("_", " ")
        self.book = Book(
            meta=BookMeta(
                book_id=self.book_id,
                title=title,
                author=author,
            )
        )

    def _process_chapters(self):
        """
        - split up book into chapters along these chapter anchors: # Chapter 1
        - use re (?=...) lookahead splits before the match
        """
        if "# Chapter" not in self.book_content:
            raise ValueError(f"No '# Chapter' anchors found in {self.input_book_path}")
        # ^ + multiline flag: match only valid at start of newline, but every newline due to flag
        chapters = re.split(r"(?=^# Chapter)", self.book_content, flags=re.MULTILINE)
        # save cleaned str list (remove empty string at 1st pos from split before # Chapter 1)
        self.book.chapters = [i for i in chapters if i.strip()]
        # safe book meta data depending on processed chapter text
        self.book.meta.total_chapters = len(self.book.chapters)
        self.book.meta.word_count = sum(len(chapter.split()) for chapter in self.book.chapters)
        self.logger.info(f"Parsed {self.book.meta.total_chapters} chapters to json book file")
        self.logger.info(f"Total words: {self.book.meta.word_count}")
        # write to json
        with open(self.book_json_path, mode="w", encoding="utf-8") as f:
            json.dump(self.book.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    def run(self):
        self.logger.info(f"Starting process book: {self.book_name} ...")
        self._create_book()
        self.logger.info("Processing chapters ...")
        self.logger.info("---------------------------------------------")
        self._process_chapters()
        self.logger.info("---------------------------------------------")
        self.logger.info(f"Book JSON file written to: {self.book_json_path}")
        self.logger.info("---------------------------------------------")
        self.logger.info("Creating world context ...")
        self._process_world_context()
        self.logger.info("------Operation completed successfully-------")


def main():
    """
    cli entry point for converting .md novels into .json with narrative split into chapters
    - Usage: python book_creator.py <input_book_path.md> <iron_heel_london> <input_ref_path.md>
    - book_id: all smallcaps and with _, e.g.: "iron_heel_london" for The Iron Heel by Jack London
    - input_ref_path: optional, if meaningful reference content available
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "input_book_path",
        help="path to input book .md file",
    )
    parser.add_argument(
        "book_id",
        help="unique identifier for book; create some author title combination"
    )
    parser.add_argument(
        "input_ref_path",
        nargs="?",
        help="path to input ref material .md file",
    )
    args = parser.parse_args()
    bp = BookProcessor(args.input_book_path, args.book_id, args.input_ref_path)
    bp.run()


if __name__ == "__main__":
    main()
