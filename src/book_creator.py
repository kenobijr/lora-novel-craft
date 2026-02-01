"""
Covert .md book into base json book file and splits narrative along chapters.
- input .md file:
    - must be valid .md file
    - must contain text
    - must contain at least one chapter in form of "# Chapter 1" or "Chapter 1: Some Title"
    - all chapter headers must be in this exact format # Chapter 1 as anchors
- book_id arg:
    - must contain the name of the author followed by the title of the book (sep. by "-")
    - all smallcaps and with _, e.g.: "iron_heel_london" for The Iron Heel by Jack London
"""

import argparse
from src.config import BookConfig, Book, BookMeta
from src.utils import init_logger
import os
import re
import json


class BookProcessor:
    def __init__(self, input_book_path: str, book_id: str, config=None):
        self.cfg = config if config is not None else BookConfig()
        self.input_book_path = input_book_path
        # unique identifier for book
        self.book_id = book_id
        with open(input_book_path, mode="r", encoding="utf-8") as f:
            self.narrative = f.read()
        self.book_name = os.path.basename(self.input_book_path).removesuffix(".md")
        # output file path to save json
        self.book_json_path = os.path.join(self.cfg.output_dir, f"{self.book_name}.json")
        self.logger = init_logger(self.cfg.operation_name, self.cfg.debug_dir, self.book_name)
        self.book = None

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
        if "# Chapter" not in self.narrative:
            raise ValueError(f"No '# Chapter' anchors found in {self.input_book_path}")
        # ^ + multiline flag: match only valid at start of newline, but every newline due to flag
        chapters = re.split(r"(?=^# Chapter)", self.narrative, flags=re.MULTILINE)
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
        self.logger.info("-------Operation completed successfully.-------")


def main():
    """
    cli entry point for converting .md novels into .json with narrative split into chapters
    - Usage: python book_creator.py <input_book_path.md> <iron_heel_london>
    - book_id: all smallcaps and with _, e.g.: "iron_heel_london" for The Iron Heel by Jack London
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "input_book_path",
        help="path to input book md file",
    )
    parser.add_argument(
        "book_id",
        help="unique identifier for book; create some author title combination"
    )
    args = parser.parse_args()
    bp = BookProcessor(args.input_book_path, args.book_id)
    bp.run()


if __name__ == "__main__":
    main()
