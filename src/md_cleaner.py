"""
CLI script to clean / format .md book files for further processing
- scope: clean .md book text from common pandoc/epub/html artefacts, which occur typically
- not scope: book-specific patterns / edgecases -> must be handled separately
- ATTENTION: chapter anchors "# Chapter 1" / "# Chapter 1: Title" must be set before cleaning!
- process:
    1. convert html blockquotes & smallcaps into .md format (keep content)
    2. remove artefacts
    3. normalize formatting (whitespace cleanup, linebreaks, ...)
"""

import os
import re
import argparse
from src.config import CleanerConfig
from src.utils import init_logger


cfg = CleanerConfig()


def convert_blockquotes(text: str) -> str:
    """ convert pandoc blockquotes (::: blockquote ... :::) to markdown (> ...) """
    def replacer(match: re.Match) -> str:
        inner_text = match.group(1)
        quoted_lines = [f"> {line}" for line in inner_text.split("\n")]
        return "\n".join(quoted_lines)
    pattern = r'::: blockquote\s*\n(.*?)\n:::'
    return re.sub(pattern, replacer, text, flags=re.DOTALL)


def convert_small_caps(text: str) -> str:
    """
    - convert small caps html tags to markdown bold with uppercase
    - <span class="smcap">text</span> -> **TEXT**
    """
    def replacer(match):
        content = match.group(1)
        return f"**{content.upper()}**"
    pattern = r'<span class="smcap">([^<]+)</span>'
    return re.sub(pattern, replacer, text)


def remove_artefacts(text: str) -> str:
    """ delete diverse html / pandoc / artefacts / ... """
    text = re.sub(r'<span id="[^"]+">([^<]+)</span>', "", text)  # span id markers with content
    text = re.sub(r'</?div[^>]*>', "", text)
    text = re.sub(r'<span class="nothing">.*?</span>', "", text)  # decorative image spans
    text = re.sub(r'<\?pagebreak[^?]*\?>', "", text)  # pagebreak directives
    # remove pages: []{#9780063068452_Chapter_1.xhtml_page_12 .right_1 .pagebreak title="12"}
    text = re.sub(r"\[\]\{.*?\}", "", text)
    text = re.sub(r"\{\..*?\}", "", text)  # css formatting: {.chap_head}
    return text


def normalize_formatting(text: str) -> str:
    """ normalize whitespace - collapse multiple newlines """
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def run_cleaner(input_book_path: str, force: bool) -> None:
    """
    - execute cleaning / formatting of .md file
    - write output file only if not existing yet (or with --force)
    """
    with open(input_book_path, "r", encoding="utf-8") as f:
        book_content = f.read()
    # init logger with book_name and log BEFORE stats
    book_name = os.path.basename(input_book_path).removesuffix(".md")
    logger = init_logger(cfg.operation_name, cfg.debug_dir, book_name)
    logger.info(f"Starting cleaning book: {book_name} ...")
    char_before, words_before = len(book_content), len(book_content.split())
    logger.info(f"BEFORE | Amount chars: {char_before:,}; Amount words: {words_before:,}")
    logger.info("---------------------------------------------")
    # execute cleaning & formatting
    book_content = convert_blockquotes(book_content)  # convert html to .md -> keep content
    book_content = convert_small_caps(book_content)  # convert html to .md -> keep content
    book_content = remove_artefacts(book_content)  # delete diverse html / pandoc / ... artefacts
    book_content = normalize_formatting(book_content)  # general cleanup, multiple linebreaks, ...
    # counting after processing & log AFTER
    char_after, words_after = len(book_content), len(book_content.split())
    char_delta, words_delta = char_before - char_after, words_before - words_after
    logger.info(f"AFTER | Amount chars: {char_after:,}; Amount words: {words_after:,}")
    logger.info(f"Chars removed: {char_delta:,}; Words removed: {words_delta:,}")
    # write processed .md file if not existing yet (or with --force)
    output_book_path = os.path.join(cfg.output_dir, f"{book_name}.md")
    if os.path.exists(output_book_path) and not force:
        raise FileExistsError(
            f"Output file already exists: {output_book_path}\n"
            f"Use --force to overwrite"
        )
    with open(output_book_path, "w", encoding="utf-8") as f:
        f.write(book_content)
    logger.info("---------------------------------------------")
    logger.info(f"Cleaned book .md file written to: {output_book_path}")
    logger.info("------Operation completed successfully-------")


def main():
    """
    cli entry point to clean / format .md book files for further processing
    - usage: python md_cleaner.py <input_book_path.md> --force
    - output file written to output_dir defined in config.py
    - overwrite output file with optional --force if it exists
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "input_book_path",
        help="path to input book md file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite output file if it exists"
    )
    args = parser.parse_args()
    run_cleaner(args.input_book_path, args.force)


if __name__ == "__main__":
    main()
