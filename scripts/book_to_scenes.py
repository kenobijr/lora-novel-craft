"""
Split up book in .md format into "Semantic Scenes" and add them to target book json
  - read in book -> split up into chapters -> loop through & process each chapter
  - preprocess each chapter by splitting it by \n\n into paragraphs and number each one
  - add amount tokens of each paragraph to obj with tiktokenizer
  - take llm response to create scenes with combined paragraphs
  - add scenes to target book json
"""
import sys
import re
from transformers import AutoTokenizer
from typing import List

tokenizer = AutoTokenizer.from_pretrained(
      "Qwen/Qwen3-30B-A3B-Thinking-2507"
      # trust_remote_code=True
)


def process_chapters(chapters: List[str]):
    # split up each chapter into paragraph divided by \n\n
    chapters = [chapter.split("\n\n") for chapter in chapters]
    # filter empty paragraphs within each chapter
    chapters = [[p for p in chapter if p.strip()] for chapter in chapters]

    print(len(chapters))
    print(len(chapters[0]))
    print(chapters[0])
    print("\n-----------------------\n")

    # bring each chapter into target text format with numbered p + token amount
    for chapter in chapters:
        lines = []
        for i, p in enumerate(chapter, start=1):
            tok_p = len(tokenizer.encode(p))
            lines.append(f"\n[P:{i}|Tok:{tok_p}] {p}")
        chapter_str = "\n".join(lines)
        print(chapter_str)
        break


def steer_mapping(input_file: str, output_file: str):
    # read in .md
    with open(input_file, mode="r", encoding="utf-8") as f:
        content = f.read()

    # split up book into chapters; form: # Chapter 1
    # (?=...) lookahead splits before the match
    # ^ + multiline flag: match only valid at start of newline, but every newline due to flag
    chapters = re.split(r"(?=^# Chapter)", content, flags=re.MULTILINE)
    # remove empty string at first pos from split before # Chapter 1
    chapters = [i for i in chapters if i.strip()]

    process_chapters(chapters)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python book_to_scenes.py <input_file.md> <output_file.json")
    else:
        steer_mapping(sys.argv[1], sys.argv[2])
