"""
- input book path .md with content sep by # Chapter 1, ... anchors
- 0-n optional reference input path; if more than 1 ref input path:
    - ref .md files must be pre-split to fit scene size
    - ref .md files must be passed in meaningful consecutive order
"""

import argparse
from typing import Tuple
from src.book_creator import BookProcessor
from src.scene_creator import SceneProcessor
from src.manual_scene_adder import add_scene
from src.summary_creator import SummaryProcessor
from src.remove_ref_scenes import remove_ref_scenes
from src.instruction_creator import InstructionProcessor
from src.dataset_compiler import CompileProcessor


def construct_ref(input_ref_path: Tuple[str]):
    """ take 1-n input ref .md path and return as concat str """
    ref_content = ""
    for ref in input_ref_path:
        with open(ref, mode="r", encoding="utf-8") as f:
            content = f.read()
        ref_content += f"{content}\n"
    return ref_content


def forge_book(input_book_path: str, *input_ref_path: str):
    print(f"FORGER_start: process book {input_book_path} ...")
    print("------Stage 1: Book & World Context-------")
    # set flag if ref mat provided; default case: no; if yes, custom logic needed at some stages
    has_ref = False
    # if no input_ref_path provided set param to None; else construct it depending on amount
    if not input_ref_path:
        ref = None
    else:
        has_ref = True
        ref = construct_ref(input_ref_path)
    # init base book .json & world_context creation -> target dir: json/base
    b = BookProcessor(input_book_path, ref)
    base_path = b.run()
    print("------Stage 2: Semantic Scene-------")
    # process narrative chapters into semantic scenes; custom logic for ref material
    s = SceneProcessor(base_path)
    scene_path = s.run()
    # if ref provided, prepend this scenes manually to scenes list
    if has_ref:
        for ref_path in input_ref_path:
            add_scene(scene_path, ref_path)
    print("------Stage 3: Running Summary-------")
    # create running summaries along semantic (ref) scenes as timesteps
    sp = SummaryProcessor(scene_path)
    summary_path = sp.run()
    # if ref provided, remove ref scenes here (were compressed to running sum of 1st semantic scene)
    if has_ref:
        remove_ref_scenes(summary_path)
    # create custom instruction along every semantic scene
    print("------Stage 4: Instruction Tuning-------")
    i = InstructionProcessor(summary_path)
    instruction_path = i.run()
    print("------Stage 5: Final .jsonl-------")
    # create target .jsonl output
    c = CompileProcessor(instruction_path)
    final_output_path = c.run()
    print(f"Ops completed: target .jsonl file at {final_output_path}")
    print("FORGER_end: Operation completed successfully-------")


def main():
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "input_book_path",
        help="path to input book .md file",
    )
    parser.add_argument(
        "input_ref_path",
        nargs="*",
        help="path to input ref material .md file",
    )
    args = parser.parse_args()
    forge_book(args.input_book_path, *args.input_ref_path)


if __name__ == "__main__":
    main()
