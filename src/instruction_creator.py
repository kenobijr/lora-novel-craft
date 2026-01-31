import sys
import json
from src.config import get_tokenizer, InstructionConfig, Book
from typing import Tuple


# load tokenizer
tokenizer = get_tokenizer()
if not tokenizer:
    raise ValueError("could not load tokenizer...")


class InstructionProcessor:
    def __init__(self, book_json_path: str, config=None):
        self.cfg = config if config is not None else InstructionConfig()
        self.book_json_path = book_json_path
        with open(book_json_path, mode="r", encoding="utf-8") as f:
            self.book_content = Book(**json.load(f))

    def _process_scenes(self, scene_range):
        for i in range(scene_range[0], scene_range[1]):
            print(self.book_content.scenes[i].scene_id)

    def run(self, scene_range: Tuple[int, int]):
        len_scenes = len(self.book_content.scenes)
        if scene_range is None:
            scene_range = (0, len_scenes)
        else:
            if scene_range[0] < 0:
                raise ValueError("...")
            if scene_range[1] > len_scenes:
                raise ValueError("...")
            if scene_range[0] >= scene_range[1]:
                raise ValueError("...")
        self._process_scenes(scene_range)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summary_creator.py <input_book.json> 0,10")
        sys.exit(2)
    else:
        scene_range = None
        if len(sys.argv) == 3:
            try:
                parts = sys.argv[2].split(",")
                scene_range = (int(parts[0]), int(parts[1]))
            except (IndexError, ValueError):
                print("Usage: 2nd argument must be range with 2 numbers split by ',': e.g.: 0,5")
                sys.exit(2)

        p = InstructionProcessor(sys.argv[1])
        p.run(scene_range)
