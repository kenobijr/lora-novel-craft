"""
Create "Running Summaries" for each scene of a book json
"""

import sys
import json
from src.config import Book, Scene, get_tokenizer, SummaryConfig

tokenizer = get_tokenizer()
if not tokenizer:
    raise ValueError("could not load tokenizer...")

# - Parts / Content:
#   - Systemmessage / Instruction
#   - World Context
#     - Fixed across all Semantic Scenes
#     - No references to chapter / scene amount (World Context = The Simulation. (Diegetic))
#   - Narrative Status / Header
#     - Prepended to the Rolling Summary or separate element before
#     - Contains strictly context about: [Progress: XX%]
#     - Current Scene / Total amount scenes %
#   - Rolling summary
#     - Seed content 1st narrative scene: "NARRATIVE INITIALIZATION: The story begins."
#     - Evolving across all Semantic Scenes

# - Algo / Script:
#   - Create Root Summary for scene 1
#   - Take world context + Rolling summary current scene (n) + scene text current scene (n)
#   - Add prompt & query llm to update the current rolling summary with this content best possible
# (within the token constraints) to produce the Rolling summary for the next scene (n + 1)
#   - Output is saved at each scene object


class SummaryProcessor:
    def __init__(self, book_json_path: str, config=None):
        # enable init with argument for testing; normal case create SceneConfig obj from config.py
        self.cfg = config if config is not None else SummaryConfig()
        # load book json & map into pydantic obj
        with open(book_json_path, mode="r", encoding="utf-8") as f:
            self.book_json = Book(**json.load(f))

    def _construct_prompt(self, is_narrative: bool):
        """
        - flag is_narrative true for narrative scenes; false for reference scenes
        - construct prompt with:
        1. systemmessage
        2. world_context
        3.
        """

    def _process_scenes(self):
        """
        - loop through all semantic scenes of book to create running summary for each
        - distinguish scene type: narrative vs. reference -> reference instruction value = "special"
        """
        for scene in self.book_json.scenes:
            # create flag for scene is narrativ type or reference
            is_narrative = True if scene.instruction != "special" else False
            # set root summary at 1st semantic scene
            scene.running_summary = (
                self.cfg.root_summary_narrative
                if is_narrative
                else self.cfg.root_summary_reference
            )
            print(self.book_json.meta.title)
            print(scene.scene_id)
            print(scene.running_summary)
            # construct prompt
            prompt = self._construct_prompt(is_narrative)

            break

    def run(self):
        """ steer the operation and print stats"""
        print("Starting process ...")
        self._process_scenes()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summary_creator.py <input_book.md>")
    else:
        sp = SummaryProcessor(sys.argv[1])
        sp.run()
