""" Take .md file as input and print token amount to console. """
import argparse
from src.config import TOKENIZER


def calc_tokens(input_md: str):
    with open(input_md, mode="r", encoding="utf-8") as f:
        content = f.read()
    tok = len(TOKENIZER.encode(content))
    print(f"Read in file: {input_md}")
    print(f"Token amount: {tok}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_md",
        help="path to book json file",
    )
    args = parser.parse_args()
    calc_tokens(args.input_md)


if __name__ == "__main__":
    main()
