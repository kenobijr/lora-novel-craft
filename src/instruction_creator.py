import sys
import json
from src.config import get_tokenizer, InstructionConfig, Book


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

    def run(self):
        ...


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summary_creator.py <input_book.json>")
        sys.exit(2)
    else:
        p = InstructionProcessor(sys.argv[1])
        p.run()
