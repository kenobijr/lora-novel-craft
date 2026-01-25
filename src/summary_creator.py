"""
Create "Running Summaries" for each scene of a book json
"""

import sys
import json
import os
from src.config import (
    Book, Scene, get_tokenizer, SummaryConfig, RunningSummary,
    get_root_summary_narrative, get_root_summary_reference
)
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict


# llm model = openrouter id
LLM = "qwen/qwen-2.5-72b-instruct"
# "google/gemini-2.5-pro"
# "qwen/qwen-2.5-72b-instruct"
# "google/gemini-2.0-flash-lite-001"

# load api key
load_dotenv()
api_key = os.getenv("OPEN_ROUTER_KEY")
if not api_key:
    raise ValueError("could not load API key...")

# load tokenizer
tokenizer = get_tokenizer()
if not tokenizer:
    raise ValueError("could not load tokenizer...")


class SummaryCreatorLLM:
    def __init__(self, config: SummaryConfig, world_context: str):
        self.cfg = config
        # world context from book json needed for each llm call
        self.wc = world_context
        # load prompts
        with open(self.cfg.prompt_system, mode="r", encoding="utf-8") as f:
            self.prompt_system = f.read()
        with open(self.cfg.prompt_input_format, mode="r", encoding="utf-8") as f:
            self.prompt_input = f.read()
        with open(self.cfg.prompt_instruction_narrative, mode="r", encoding="utf-8") as f:
            self.prompt_instruction_nar = f.read()
        with open(self.cfg.prompt_instruction_reference, mode="r", encoding="utf-8") as f:
            self.prompt_instruction_ref = f.read()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=3  # standard SDK feature: try 3 times before giving up for certain errors
        )

    def _construct_prompt(self, scene: Scene, novel_progress: int, is_narrative: bool) -> str:
        """Construct prompt with system, world_context, rolling summary, scene text, instruction."""
        prompt_instruction = (
            self.prompt_instruction_nar
            if is_narrative
            else self.prompt_instruction_ref
        )
        return f"""
<system>
{self.prompt_system}
</system>

<input_description>
{self.prompt_input}
</input_description>

<world_context>
{self.wc}
</world_context>

<current_rolling_summary>
NOVEL_PROGRESS: {novel_progress}%
{scene.running_summary}
</current_rolling_summary>

<scene_text>
{scene.text}
</scene_text>

<instruction>
{prompt_instruction}
</instruction>
"""

    def get_llm_running_summary(
            self,
            scene: Scene,
            novel_progress: int,
            is_narrative: bool
    ) -> dict:
        """
        - prompt llm to create updated running summary for scene
        - return validated dict with local_momentum & global_state
        """
        prompt = self._construct_prompt(scene, novel_progress, is_narrative)
        response = self.client.chat.completions.create(
            model=LLM,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "running_summary",
                    "strict": True,
                    "schema": RunningSummary.model_json_schema()
                }
            },
            # fits only on qwen3!!!
            extra_body={                                         
                "provider": {                                         
                    "only": ["DeepInfra"]                                          
                }                                                             
            }
        )
        result = json.loads(response.choices[0].message.content)
        if not result:
            raise ValueError(f"No api result for scene: {scene.scene_id}")
        # DEBUG
        print(prompt)
        print(result)
        return result


class SummaryProcessor:
    def __init__(self, book_json_path: str, config=None):
        # enable init with argument for testing; normal case create SceneConfig obj from config.py
        self.cfg = config if config is not None else SummaryConfig()
        # load book json & map into pydantic obj
        with open(book_json_path, mode="r", encoding="utf-8") as f:
            self.book_json = Book(**json.load(f))
        self.llm = SummaryCreatorLLM(self.cfg, self.book_json.meta.world_context)

    def _format_running_summary(self, summary_dict: Dict) -> str:
        """ take running summary as python dict rep and map into target str output format"""
        lines = [f"{key}: {value}" for key, value in summary_dict.items()]
        return "\n".join(lines)

    def _set_root_summary(self) -> None:
        """
        - set root summary at first scene manually
        - pydanctic obj -> py dict rep -> str to mirror llm response flow and use same logic
        - distinguish between narrative vs. reference root type
        """
        first_scene = self.book_json.scenes[0]
        # get root narrative or reference summary depending on scene instruciton attribute
        is_narrative = True if first_scene.instruction != "special" else False
        root = get_root_summary_narrative() if is_narrative else get_root_summary_reference()
        root = self._format_running_summary(root.model_dump())
        # safe at 1st scene
        first_scene.running_summary = root

    def _calc_novel_progress(self, scene_id: int) -> int:
        """
        calculate narrative progress percentage.
        progress represents "story completed so far" - the state BEFORE this scene.
        - scene 1: 0% (nothing written yet)
        - scene N: (N-1)/total (scenes 1..N-1 are done)
        """
        total = self.book_json.meta.total_scenes
        return int(((scene_id - 1) / total) * 100)

    def _process_scenes(self):
        """
        - loop through all semantic scenes of book to create running summary for each
        - distinguish scene type: narrative vs. reference -> reference instruction value = "special"
        """
        for scene in self.book_json.scenes:
            print(f"starting processing scene id: {scene.scene_id}")
            # create flag for scene is narrativ type or reference
            is_narrative = True if scene.instruction != "special" else False
            # calc novel progress of scene
            novel_progress = self._calc_novel_progress(scene.scene_id)
            print("Query LLM ...")
            # get updated rolling summary from llm
            updated_summary = self.llm.get_llm_running_summary(scene, novel_progress, is_narrative)
            print("\n=== LLM RESPONSE ===")
            print(self._format_running_summary(updated_summary))
            break

    def run(self):
        """ steer the operation and print stats"""
        print(f"Starting process book: {self.book_json.meta.title} ...")
        print("Setting root summary at 1st sceen manually ...")
        self._set_root_summary()
        print("Start processing scenes ...")
        print("---------------------------------------------")
        self._process_scenes()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summary_creator.py <input_book.md>")
    else:
        sp = SummaryProcessor(sys.argv[1])
        sp.run()
