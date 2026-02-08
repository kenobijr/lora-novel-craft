import argparse
import os
import json
import logging
from src.config import TOKENIZER, CompilerConfig, Book, CompileStats, BaseLLM
from src.utils import init_logger, parse_range
from typing import Tuple


class CompileLLM(BaseLLM):
    def __init__(
        self,
        config,
        logger: logging.Logger,
        stats: CompileStats,
    ):
        super().__init__(config, logger)
        self.stats = stats


class CompileProcessor:
    def __init__(self, book_path: str, config: CompilerConfig = None):
        self.cfg = config if config is not None else CompilerConfig()
        self.book_path = book_path
        with open(book_path, mode="r", encoding="utf-8") as f:
            self.book_content = Book(**json.load(f))
        book_name = os.path.basename(book_path).removesuffix(".json")
        self.logger = init_logger(self.cfg.operation_name, self.cfg.debug_dir, book_name)
        self.stats = CompileStats()
        self.llm = CompileLLM(
            self.cfg,
            self.logger,
            self.book_content.meta.world_context,
            self.stats,
        )

    def _process_scenes(self, scene_range: Tuple[int, int]) -> None:
        ...

    def run(self, scene_range: Tuple[int, int] | None = None) -> None:
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
        self.logger.info(f"Starting process book: {self.book_content.meta.title} ...")
        self.logger.info(f"Processing scenes: start {scene_range[0]} - end {scene_range[1]}...")
        self.logger.info("---------------------------------------------")
        self._process_scenes(scene_range)


def main():
    """
    cli entry point for compiling book .json into target .jsonl train format
    - default: row is created for each scene in book json scene list
    - provide optional scene range arg to create rows for only certain range of scenes
    - "Usage: python dataset_compiler.py <input_book.json> <start,end>"
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "book_path",
        help="path to book json file",
    )
    parser.add_argument(
        "scene_range",
        nargs="?",
        type=parse_range,
        help="optional range as start,end (e.g. 0,10)",
    )
    args = parser.parse_args()
    p = CompileProcessor(args.book_path)
    p.run(args.scene_range)


if __name__ == "__main__":
    main()
