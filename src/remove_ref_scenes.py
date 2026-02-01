import argparse
import json
from src.config import Book


def remove_ref_scenes(book_path: str):
    with open(book_path, mode="r", encoding="utf-8") as f:
        book_content = Book(**json.load(f))
    # delete all ref scenes
    book_content.scenes = [scene for scene in book_content.scenes if scene.instruction != "special"]
    # update scene_id's
    for i, scene in enumerate(book_content.scenes, start=1):
        scene.scene_id = i
    with open(book_path, mode="w", encoding="utf-8") as f:
        json.dump(book_content.model_dump(mode="json"), f, indent=2, ensure_ascii=False)


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
