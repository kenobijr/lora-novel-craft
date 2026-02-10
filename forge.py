"""
- input book path .md with content sep by # Chapter 1, ... anchors
- 0-n optional reference input path; if more than 1 ref input path: 
    - ref .md files must be pre-split to fit scene size
    - ref .md files must be passed in meaningful consecutive order 
"""

import argparse
from typing import Tuple
from src.book_creator import BookProcessor


def construct_ref(input_ref_path: Tuple[str]):
    """ take 1-n input ref .md path and return as concat str """
    ref_content = ""
    for ref in input_ref_path:
        with open(ref, mode="r", encoding="utf-8") as f:
            content = f.read()
        ref_content += f"{content}\n"
    return ref_content


def forge_book(input_book_path: str, *input_ref_path: str):
    # if no input_ref_path provided set param to None; else construct it depending on amount
    if not input_ref_path:
        ref = None
    else:
        ref = construct_ref(input_ref_path)
    # init base book .json & world_context creation -> target dir: json/base
    # b = BookProcessor(input_book_path, ref)
    # b.run()


def main():
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "input_book_path",
        help="path to input book .md file",
    )
    parser.add_argument(
        "input_ref_path",
        nargs="*",
        help="path to input ref material .md file",
    )
    args = parser.parse_args()
    forge_book(args.input_book_path, *args.input_ref_path)

if __name__ == "__main__":
    main()
