"""
Remove all Reference scenes from a given book json files scene list and update scene_id's after it.
- reference scenes content must not go into training dataset like regular narrative scenes
- instead, they can be integrated indirectly into world context and running summaries as context
- after processing running summaries they must be removed
"""

import argparse
import json
from src.config import Book


def remove_ref_scenes(book_path: str):
    """ take input book json; filter out ref scenes; update scene_ids; write back into json """
    print("started process...")
    # read in json
    with open(book_path, mode="r", encoding="utf-8") as f:
        book_content = Book(**json.load(f))
    # print stats before
    print(f"Total amount scenes before: {len(book_content.scenes)}")
    ref_scenes = sum(1 for scene in book_content.scenes if scene.instruction == "special")
    print(f"Total amount ref scenes before: {ref_scenes}")
    # delete all ref scenes
    book_content.scenes = [scene for scene in book_content.scenes if scene.instruction != "special"]
    # update scene_id's
    for i, scene in enumerate(book_content.scenes, start=1):
        scene.scene_id = i
    # update global word counter & scene counter at book meta
    full_text = " ".join([scene.text for scene in book_content.scenes])
    book_content.meta.word_count = len(full_text.split())
    book_content.meta.total_scenes = len(book_content.scenes)
    # write back to json
    with open(book_path, mode="w", encoding="utf-8") as f:
        json.dump(book_content.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
    # print stats after
    print(f"Total amount scenes after: {len(book_content.scenes)}")
    ref_scenes = sum(1 for scene in book_content.scenes if scene.instruction == "special")
    print(f"Total amount ref scenes after: {ref_scenes}")
    print(f"Operation completed successfully. File updated: {book_path}")


def main():
    """
    cli entry point to remove all ref scenes from book json file
    - usage: <target_book.json>
    """
    parser = argparse.ArgumentParser(description=__name__)
    parser.add_argument(
        "book_path",
        help="path to book json file",
    )
    args = parser.parse_args()
    remove_ref_scenes(args.book_path)


if __name__ == "__main__":
    main()
