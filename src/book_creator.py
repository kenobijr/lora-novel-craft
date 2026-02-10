"""
Convert .md book into base .json book file, split text along chapters and create World Context file.
- input .md file:
    - input book file must come in syntax "author_name-some_title.md" to process name + title
    - all chapter headers must be like "# Chapter 1" or "Chapter 1: Some Title" as content anchors
- reference .md file (optional):
    - provide additional reference content to be used for world context creation
"""

import argparse
from src.config import (
    BaseLLM, BookConfig, Book, BookMeta, WorldContext, TOKENIZER, MODEL_REGISTRY, BookStats
)
from src.utils import init_logger, format_llm_response
from typing import Dict
import logging
import os
import re
import json


class WorldContextLLM(BaseLLM):
    """
    - handles llm related logic: model / api / key / connections / ...
    - loading & saving base prompts, logger & openai client done via base class from config.py
    - constructing prompt with task-specific content & llm queries done in each subclass
    """
    def __init__(
        self,
        config: BookConfig,
        logger: logging.Logger,
        book_content: str,
        book_reference: str | None,
        stats: BookStats,
    ):
        super().__init__(config, logger)
        # novel text as str
        self.book_content = book_content
        # additional novel reference material as str (optional)
        self.book_reference = book_reference
        self.stats = stats

    def _construct_prompt_compress(self, prompt: str, response: Dict, amount_words: int) -> str:
        """ prompt to compress an oversized world context response down to word limit """
        return f"""
<system>
{self.prompt_system}
</system>

<instruction>
CRITICAL: You received following prompt earlier: {prompt}
You generated the following World Context: {json.dumps(response, indent=2)}
It has {amount_words} words (counting JSON values only).
**The maximum is {self.cfg.max_words} words.
You must cut at least {amount_words - self.cfg.max_words} words.**

Compress each field while preserving:
- Core world rules and power structures
- Sensory anchors that ground scene generation
- Faction dynamics and spatial anchors

Cut generic adjectives, redundant qualifiers and vague descriptions first.
Keep the same JSON field structure and format. Do not add or remove fields.
</instruction>
"""

    def _compress_wc(self, prompt: str, response: Dict, amount_words: int) -> Dict:
        """
        - if world context response greater than allowed max words, compress it
        - send original prompt (with novel) + response to llm with compress instruction
        - do max x runs (defined in cfg) using same prompt each time
        - if compressed response < word threshold, return; otherwise return last response
        """
        adapted_prompt = self._construct_prompt_compress(prompt, response, amount_words)
        for run in range(self.cfg.max_compress_attempts):
            for attempt in range(self.cfg.query_retry):
                self.logger.debug(
                    f"\n=== WC COMPRESSION: PROMPT START ===\n"
                    f"{adapted_prompt}\n"
                    f"=== WC COMPRESSION: PROMPT END ==="
                )
                compressed_response = self.client.chat.completions.create(
                    model=self.cfg.llm,
                    messages=[{"role": "user", "content": adapted_prompt}],
                    temperature=0.5,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "world_context_creation",
                            "strict": True,
                            "schema": WorldContext.model_json_schema()
                        }
                    },
                    **MODEL_REGISTRY.get(self.cfg.llm, {}),
                )
                compressed_content = compressed_response.choices[0].message.content
                self.logger.debug(
                    f"\n=== WC COMPRESSION: RESPONSE START ===\n"
                    f"{compressed_content}\n"
                    f"=== WC COMPRESSION: RESPONSE END ==="
                )
                try:
                    compressed_result = json.loads(compressed_content)
                    break
                except json.JSONDecodeError as e:
                    if attempt < self.cfg.query_retry - 1:
                        self.logger.warning(f"Invalid JSON at WC compression: {e}")
                        continue
                    raise
            # count words from LLM response (dict values) & update stats
            total_words = sum(len(str(v).split()) for v in compressed_result.values())
            self.logger.info(f"WC compress run # {run}: LLM response amount words: {total_words}")
            self.stats.compress_runs += 1
            if total_words <= (self.cfg.max_words + self.cfg.max_words_buffer):
                self.logger.info(f"WC compressed successfully after run # {run}")
                self.stats.compressed_successfully = True
                return compressed_result
        self.logger.info(f"WC compression failed; returned last run # {run} as response.")
        return compressed_result

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
        for attempt in range(self.cfg.query_retry):
            # log full prompt to logfile before llm query
            self.logger.debug(
                f"\n=== WORLD CONTEXT CREATION: PROMPT START ===\n"
                f"{prompt}\n"
                f"=== WORLD CONTEXT CREATION: PROMPT END ==="
            )
            response = self.client.chat.completions.create(
                model=self.cfg.llm,
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
                # load additional cfg from registry for certain models
                **MODEL_REGISTRY.get(self.cfg.llm, {})
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
            except json.JSONDecodeError:
                if attempt < self.cfg.query_retry - 1:
                    self.logger.warning("Invalid JSON response at world context creation")
                    continue
                raise

        # count words from LLM response (dict values) & update stats / logs
        total_words = sum(len(str(v).split()) for v in result.values())
        self.logger.info(f"World Context 1st query: LLM response amount words: {total_words}")
        # if llm response not in total words constraint + buffer, compress it
        if total_words > (self.cfg.max_words + self.cfg.max_words_buffer):
            result = self._compress_wc(prompt, result, total_words)
        # count words final response (compressed or not) to save at stats word counter
        self.stats.wc_words = sum(len(str(v).split()) for v in result.values())
        return result


class BookProcessor:
    def __init__(self, input_book_path: str, ref: str | None, config=None):
        self.cfg = config if config is not None else BookConfig()
        self.input_book_path = input_book_path
        with open(input_book_path, mode="r", encoding="utf-8") as f:
            self.book_content = f.read()
        # optional ref material -> if available, pass to llm wc creation
        self.book_reference = ref
        # output file path to save json
        self.book_name = os.path.basename(self.input_book_path).removesuffix(".md")
        self.output_path = os.path.join(self.cfg.output_dir, f"{self.book_name}.json")
        self.logger = init_logger(self.cfg.operation_name, self.cfg.debug_dir, self.book_name)
        # construct book id from book_name: pattern: title last name author
        author, title = self.book_name.split("-", 1)
        last_name = author.split("_")[-1]
        self.book_id = f"{title}_{last_name}".lower()
        # output book obj
        self.book = None
        # stats
        self.stats = BookStats()
        # setup llm
        self.llm = WorldContextLLM(
            self.cfg,
            self.logger,
            self.book_content,
            self.book_reference,  # str | None
            self.stats,
        )

    def _process_world_context(self):
        """ steer world context creation with llm call and save to json """
        try:
            wc = self.llm.create_world_context()
        except Exception:
            self.logger.exception("Failed process creating world context...")
            raise
        # bring response into target .md format
        wc = format_llm_response(wc, "Novel World Context")
        # save & write to json
        self.book.meta.world_context = wc
        with open(self.output_path, mode="w", encoding="utf-8") as f:
            json.dump(self.book.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
        self.logger.info(f"World context saved to file: {self.output_path}")

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
        # write to json
        with open(self.output_path, mode="w", encoding="utf-8") as f:
            json.dump(self.book.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
        self.logger.info(f"Book JSON file written to: {self.output_path}")

    def _create_report(self):
        s = self.stats
        # narrative
        content_words = self.book.meta.word_count
        content_tok = sum(len(TOKENIZER.encode(chapter)) for chapter in self.book.chapters)
        len_chapters = self.book.meta.total_chapters
        avg_w, avg_t = int(content_words / len_chapters), int(content_tok / len_chapters)
        self.logger.info("-------------NARRATIVE CONTENT---------------")
        self.logger.info(f"A total of # {len_chapters} chapters were parsed.")
        self.logger.info(f"Narrative content: {content_words:,} words; {content_tok:,} tokens.")
        self.logger.info(f"Avg word/chapter: {avg_w:,}; Avg tok/chapter: {avg_t:,}")
        # world_context
        wc_tokens = len(TOKENIZER.encode(self.book.meta.world_context))
        self.logger.info("---------------WORLD CONTEXT-----------------")
        self.logger.info(f"World Context: Compression needed: {s.compress_runs} runs")
        self.logger.info(f"Compression successful: {s.compressed_successfully}")
        self.logger.info(f"World Context: Total words raw llm response: {self.stats.wc_words}")
        self.logger.info(f"World Context: Total amount tokens in final format: {wc_tokens}")
        # params
        self.logger.info("--------------OPERATION PARAMS---------------")
        self.logger.info(f"Max words: {self.cfg.max_words}")
        self.logger.info(f"Max words buffer: {self.cfg.max_words_buffer}")
        self.logger.info(f"LLM used: {self.cfg.llm}")

    def run(self):
        """
        - load book .md file, extract narrative and split along chapter anchors
        - create base json file with chapters text, write to json
        - query llm with book text (+ optional ref material) to get world_context, save to json
        """
        self.logger.info(f"Starting process book: {self.book_name} ...")
        self._create_book()
        self.logger.info("Processing chapters ...")
        self.logger.info("---------------------------------------------")
        self._process_chapters()
        # update book meta counters after chapter processing
        self.book.meta.total_chapters = len(self.book.chapters)
        self.book.meta.word_count = sum(len(chapter.split()) for chapter in self.book.chapters)
        self.logger.info("---------------------------------------------")
        self.logger.info("Creating world context ...")
        self._process_world_context()
        self.logger.info("---------------------------------------------")
        self._create_report()
        self.logger.info("------Operation completed successfully-------")
        return self.output_path


def main():
    """
    cli entry point for converting .md novels into .json with narrative split into chapters
    - Usage: python book_creator.py <input_book_path.md> <input_ref_path.md>
    - input_ref_path: optional, if meaningful reference content available
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "input_book_path",
        help="path to input book .md file",
    )
    parser.add_argument(
        "input_ref_path",
        nargs="?",
        help="path to input ref material .md file",
    )
    args = parser.parse_args()
    # if .md ref provided, read & pass as str (same way as in orchestrator)
    ref = None
    if args.input_ref_path is not None:
        with open(args.input_ref_path, mode="r", encoding="utf-8") as f:
            ref = f.read()
    bp = BookProcessor(args.input_book_path, ref)
    bp.run()


if __name__ == "__main__":
    main()
