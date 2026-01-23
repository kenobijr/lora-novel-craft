"""
Add special reference material as additional scene to json_scenes book file.
- check against max tok len with tokenizer
- scene is prepended to scenes list
arguments:
- target book json
- scene_content.md
- scene_id
"""

import sys
import json
from transformers import AutoTokenizer
from src.config import SceneConfig

# init tokenizer & config
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Thinking-2507")
cfg = SceneConfig()


def add_scene(book_json_path, scene_md_path, scene_id):
    print("started process...")
    # load book_json
    with open(book_json_path, mode="r", encoding="utf-8") as f:
        book_content = json.load(f)
    # load scene content
    with open(scene_md_path, mode="r", encoding="utf-8") as f:
        scene_text = f.read()
    # calc & check text against max token size
    tok_amount = len(tokenizer.encode(scene_text))
    assert tok_amount <= cfg.max_scene_size, "Scene text token amount too big..."
    print(f"Read in Reference scene content in valid token amount range: {tok_amount} tokens")
    # map & check scene_id
    scene_id = int(scene_id)
    assert scene_id > 0, "Valid scene_id > 0 needed for scene adding..."
    # build scene obj
    new_scene = {}
    new_scene["scene_id"] = scene_id
    # special content: chapter idx 0; title "Reference"; instruction "special" vs. null
    new_scene["chapter_index"] = 0
    new_scene["chapter_title"] = "Reference"
    new_scene["instruction"] = "special"
    new_scene["text"] = scene_text
    new_scene["recursive_summary"] = None
    print(f"Len scenes before adding: {len(book_content["scenes"])}")
    # prepend scene to book json scenes list at 1st pos
    book_content["scenes"].insert(0, new_scene)
    print(f"Len scenes after adding: {len(book_content["scenes"])}")
    # write to target json
    with open(book_json_path, mode="w", encoding="utf-8") as f:
        json.dump(book_content, f, indent=2, ensure_ascii=False)
    print(f"Operation completed successfully. File updated: {book_json_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: <target_book.json> <scene_content.md> <scene_id>")
    else:
        add_scene(sys.argv[1], sys.argv[2], sys.argv[3])
