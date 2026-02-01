"""
Add special e.g. reference material as additional scenes to json_scenes book file.
- scene text is checked against scene max token threshold
- scene is always prepended at list and scene_ids of all elements updated
"""

import argparse
import json
from src.config import TOKENIZER, SceneConfig, Book, Scene


cfg = SceneConfig()


def add_scene(book_json_path, scene_md_path):
    print("started process...")
    # load book_json into dicts & unpack into pydantic obj
    with open(book_json_path, mode="r", encoding="utf-8") as f:
        book_content = Book(**json.load(f))
    # load scene content
    with open(scene_md_path, mode="r", encoding="utf-8") as f:
        scene_text = f.read()
    # calc & check text against max token size
    tok_amount = len(TOKENIZER.encode(scene_text))
    if tok_amount > cfg.max_tokens:
        raise ValueError(f"Scene text token: {tok_amount} over threshold: {cfg.max_tokens}")
    # create new scene
    new_scene = Scene(
        scene_id=1,
        # reference content: chapter idx always 0, since prepended before narrative content
        chapter_index=0,
        # reference content: title always "Reference"
        chapter_title="Reference",
        # reference content: instruction at this step set "special", to differentiate from None
        instruction="special",
        text=scene_text,
        running_summary=None,
    )
    print(f"Len scenes before adding: {len(book_content.scenes)}")
    # prepend scene to book json scenes list at 1st pos
    book_content.scenes.insert(0, new_scene)
    print(f"Len scenes after adding: {len(book_content.scenes)}")
    # update all scene_id's
    for i, scene in enumerate(book_content.scenes, start=1):
        scene.scene_id = i
    # write to target json
    with open(book_json_path, mode="w", encoding="utf-8") as f:
        json.dump(book_content.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
    print(f"Operation completed successfully. File updated: {book_json_path}")


def main():
    """
    cli entry point to add ref scene manually to existing book json file
    - usage: <target_book.json> <scene_content.md>
    """
    parser = argparse.ArgumentParser(description=__name__)
    parser.add_argument(
        "book_path",
        help="path to book .json",
    )
    parser.add_argument(
        "scene_content",
        help="path to scene content .md"
    )
    args = parser.parse_args()
    add_scene(args.book_path, args.scene_content)


if __name__ == "__main__":
    main()
