"""
CLI script to clean / format .md book files for further processing

pipeline order:
1. process_footnotes() - extract, replace inline refs, remove blocks
2. html_to_markdown() - convert html elements to md format
3. remove_artifacts() - remove pandoc/css noise
4. normalize_formatting() - whitespace cleanup
"""

import os
import re
import argparse


def extract_footnotes(text: str) -> dict[str, str]:
    """
    - extract all footnote blocks into a dictionary
    - coming in this format "href="#4356712297217026545_1164-h-27.htm.html_fn-25.2"
    id="4356712297217026545_1164-h-27.htm.html_fnref-25.2" class="pginternal"><sup>[2]</sup></a>
    <span id="4356712297217026545_1164-h-27.htm.html_fn-25.2"></span>
    <a href="#4356712297217026545_1164-h-27.htm.html_fnref-25.2" class="pginternal">[2]</a>
    # regex to match: <span id="...fn-N"></span> <a href="...fnref-N"...>[N]</a> footnote text EOL
    # supports both integer (1) and decimal (2.1) footnote numbers
    """
    footnotes = {}
    pattern = r'<span id="([^"]*?fn-(\d+(?:\.\d+)?)[^"]*?)"></span>\s*<a[^>]*>\[[\d.]+\]</a>\s*(.+?)$'

    for match in re.finditer(pattern, text, flags=re.MULTILINE):
        footnote_num = match.group(2)
        footnote_text = match.group(3).strip()
        footnotes[footnote_num] = footnote_text

    return footnotes


def replace_inline_footnotes(text: str, footnotes: dict[str, str]) -> str:
    """
    - replace inline footnote references with [Note: text]
    - regex to match: <a href="#...fn-N" id="...fnref-N"...><sup>[N]</sup></a>
    - supports both integer (1) and decimal (2.1) footnote numbers
    - deliver (content) vs. [Note: content] to prevent confusing llm with system syntax
    """
    def replacer(match):
        footnote_num = match.group(1)
        if footnote_num in footnotes:
            return f" ({footnotes[footnote_num]})"
        return match.group(0)  # keep original if not found

    pattern = r'<a href="#[^"]*?fn-(\d+(?:\.\d+)?)"[^>]*><sup>\[[\d.]+\]</sup></a>'
    return re.sub(pattern, replacer, text)


def remove_footnote_blocks(text: str) -> str:
    """
    - remove all remaining footnote block artifacts
    - supports both integer (1) and decimal (2.1) footnote numbers
    """
    pattern = r'<span id="[^"]*?fn-[\d.]+[^"]*?"></span>\s*<a[^>]*>\[[\d.]+\]</a>.*?$\n?'
    return re.sub(pattern, '', text, flags=re.MULTILINE)


def format_blockquotes(match: re.Match) -> str:
    """
    - match multi line pandoc blockquotes "::: blockquote" [...] ":::"
    - add ">" at beginning of all such lines
    """
    # grab the inner text (Group 1)
    inner_text = match.group(1)
    # Add "> " to the start of every line
    quoted_lines = [f"> {line}" for line in inner_text.split('\n')]
    # Join them back together
    return "\n".join(quoted_lines)


def convert_small_caps(text: str) -> str:
    """
    - convert small caps html tags to markdown bold with uppercase
    <span class="smcap">text</span> -> **TEXT**
    """
    def replacer(match):
        content = match.group(1)
        return f"**{content.upper()}**"

    pattern = r'<span class="smcap">([^<]+)</span>'
    return re.sub(pattern, replacer, text)


def clean_inscriptions(text: str) -> str:
    """
    - handles: S<span class="small">YRUP OF</span> -> SYRUP OF
    - logic: find the first letter, then the span, then join them and uppercase everything
    - pattern: matches a letter followed by the span tag containing more letters
    """
    pattern = r'([A-Za-z])<span class="small">(.*?)</span>'

    def merge_and_upper(match):
        first_letter = match.group(1)
        rest_of_text = match.group(2)
        return (first_letter + rest_of_text).upper()
    return re.sub(pattern, merge_and_upper, text)


def process_footnotes(text: str) -> str:
    """ handle footnotes: extract, replace inline refs, remove blocks """
    footnotes = extract_footnotes(text)
    text = replace_inline_footnotes(text, footnotes)
    text = remove_footnote_blocks(text)
    return text


def html_to_markdown(text: str) -> str:
    # process blockquotes (::: -> >)
    # flags=re.DOTALL allows the dot (.) to match newlines
    text = re.sub(r'::: blockquote\s*\n(.*?)\n:::', format_blockquotes, text, flags=re.DOTALL)
    # convert small caps html to markdown bold uppercase
    text = convert_small_caps(text)
    # clean inscriptions
    text = clean_inscriptions(text)
    # remove anchor tags that link to internal notes section (href="#notes.xhtml...")
    text = re.sub(r'<a href="#notes\.xhtml[^"]*"[^>]*>(.*?)</a>', r'\1', text)
    # Remove page marker spans but keep the text inside
    text = re.sub(r'<span id="[^"]+">([^<]+)</span>', r'\1', text)
    # Remove empty span id anchors (footnote/endnote markers)
    text = re.sub(r'<span id="[^"]+"></span>', '', text)
    return text


def remove_artifacts(text: str) -> str:
    # remove decorative image spans
    text = re.sub(r'<span class="nothing">.*?</span>', '', text)
    # Remove pagebreak directives
    text = re.sub(r'<\?pagebreak[^?]*\?>', '', text)
    # remove pages: []{#9780063068452_Chapter_1.xhtml_page_12 .right_1 .pagebreak title="12"}
    text = re.sub(r"\[\]\{.*?\}", "", text)
    # remove css formatting: {.chap_head}
    text = re.sub(r"\{\..*?\}", "", text)
    return text


def normalize_formatting(text: str) -> str:
    # scene breaks: matches divider & any trailing/leading whitespace/newlines & replace with \n\n
    pattern = r"\n+\s*\\?\*[\s\\?\*]+\s*\n+"
    text = re.sub(pattern, "\n\n", text)
    # normalize multiple consecutive line breaks to single line break
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def run_cleaner(file_name: str, input_dir: str, output_dir: str, force: str) -> None:
    """
    - I/O operations specified by CLI arguments
    - execute cleaning / formatting of .md file
    - write output file only if not existing yet (or with --force)
    - print simple before / after stats
    """
    # read input .md file
    input_file = os.path.join(input_dir, file_name)
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()
    # counting before processing
    char_before = len(content)
    token_before = len(content.split())
    # execute cleaning & formatting
    content = process_footnotes(content)
    content = html_to_markdown(content)
    content = remove_artifacts(content)
    content = normalize_formatting(content)
    # counting after processing
    char_after = len(content)
    token_after = len(content.split())
    # write processed .md file if not existing yet (or with --force)
    output_file = os.path.join(output_dir, file_name)
    if os.path.exists(output_file) and not force:
        raise FileExistsError(
            f"Output file already exists: {output_file}\n"
            f"Use --force to overwrite"
        )
    with open(output_file, "w", encoding="utf8") as f:
        f.write(content)
    # print stats
    char_delta = char_before - char_after
    token_delta = token_before - token_after
    print(f"File cleaned and saved as {output_file} successfully.")
    print(f"BEFORE | Amount chars: {char_before:,}; Amount tokens: {token_before:,}")
    print(f"AFTER  | Amount chars: {char_after:,}; Amount tokens: {token_after:,}")
    print(f"Chars removed: {char_delta:,}; Tokens removed: {(token_delta):,}")


def process_args() -> argparse.Namespace:
    """
    process script cli arguments
    - mandatory: input file to process without path
    - optional:
        - input_dir path if it differs from default
        - output_dir path if it differs from default
        - --force if existing output file should be overwritten
    """
    parser = argparse.ArgumentParser(description="convert raw_md into raw_json")
    parser.add_argument(
        "input_file",
        help="Complete file name without path, e.g.: example_book_1.md"
    )
    parser.add_argument(
        "-i", "--input_dir",
        default="./data/md_raw/",
        help="Input Dir (default: .data/md_raw/)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        default="./data/md_clean/",
        help="Output Dir (default: ./data/md_clean/)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it exists"
    )
    return parser.parse_args()


if __name__ == "__main__":
    arg = process_args()
    run_cleaner(arg.input_file, arg.input_dir, arg.output_dir, arg.force)
