"""
Helper script to replace all word occurrences defined in a dict within .md file
- input .md file is replaced by the edited output file
- adapt replacement dict
"""
import sys


# word_a is replaced by word_b
replacement = {
    "word_a": "word_b",
    "word_1": "word_2",
}


def replace_words(md_file: str) -> None:
    assert md_file.endswith(".md"), "Only .md files accepted"
    with open(md_file, mode="r", encoding="utf-8") as f:
        content = f.read()
    counter = 0
    for match, replace in replacement.items():
        if match in content:
            counter += content.count(match)
            content = content.replace(match, replace)
    print(f"Replaced {counter} matches.")
    with open(md_file, mode="w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python replace_words.py <target_file.md>")
    else:
        replace_words(sys.argv[1])
