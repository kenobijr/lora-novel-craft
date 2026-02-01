"""
Add special e.g. reference material as additional scenes to json_scenes book file.
- check against max tok len with tokenizer
- scene is prepended to scenes list
"""

import argparse
import json
from src.config import TOKENIZER, SceneConfig, Book, Scene


# load cfg
cfg = SceneConfig()


def add_scene(book_json_path, scene_md_path, scene_id):
    print("started process...")
    # load book_json into dicts & unpack into pydantic obj
    with open(book_json_path, mode="r", encoding="utf-8") as f:
        book_content = Book(**json.load(f))
    # load scene content
    with open(scene_md_path, mode="r", encoding="utf-8") as f:
        scene_text = f.read()
    # calc & check text against max token size
    tok_amount = len(TOKENIZER.encode(scene_text))
    assert tok_amount <= cfg.max_scene_size, "Scene text token amount too big..."
    print(f"Read in Reference scene content in valid token amount range: {tok_amount} tokens")
    # map & check scene_id
    scene_id = int(scene_id)
    assert scene_id > 0, "Valid scene_id > 0 needed for scene adding..."
    new_scene = Scene(
        scene_id=scene_id,
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
    # write to target json
    with open(book_json_path, mode="w", encoding="utf-8") as f:
        json.dump(book_content.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
    print(f"Operation completed successfully. File updated: {book_json_path}")


def main():
    """
    cli entry point to add scenes manual at specified spot to existing book json file
    - usage: <target_book.json> <scene_content.md> <scene_id>
    - scene_id attribute the added scene will get in scenes list - check other scenes id's!
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
    parser.add_argument(
        "scene_id",
        help="scene_id attribute for new scene"
    )
    args = parser.parse_args()
    add_scene(args.book_path, args.scene_content, args.scene_id)


if __name__ == "__main__":
    main()
