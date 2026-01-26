"""
Create "Running Summaries" for each scene of a book json
"""

import sys
import json
import os
from datetime import datetime
from src.config import (
    Book, Scene, get_tokenizer, SummaryConfig, RunningSummary,
    get_root_summary_narrative, get_root_summary_reference
)
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Tuple


# llm model = openrouter id
LLM = "google/gemini-2.0-flash-lite-001"
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
    def __init__(self, config: SummaryConfig, world_context: str, book_json_path: str):
        self.cfg = config
        # world context from book json needed for each llm call
        self.wc = world_context
        # save title file for debugging
        self.book_name = os.path.basename(book_json_path).removesuffix(".json")
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

    def _debug_llm_call(self, prompt: str, response: Dict) -> None:
        os.makedirs(self.cfg.debug_dir, exist_ok=True)
        ts = datetime.now().strftime("%H%M%S_%f")
        prompt_path = os.path.join(self.cfg.debug_dir, f"debug_prompt_{self.book_name}_{ts}.md")
        llm_path = os.path.join(self.cfg.debug_dir, f"debug_llm_{self.book_name}_{ts}.json")
        with open(prompt_path, mode="w", encoding="utf-8") as f:
            f.write(prompt)
        with open(llm_path, mode="w", encoding="utf-8") as f:
            json.dump(response, f, indent=2, ensure_ascii=False)

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
            # only needed for qwen3!!!
            # extra_body={                                         
            #     "provider": {                                         
            #         "only": ["DeepInfra"]                                          
            #     }                                                             
            # }
        )
        result = json.loads(response.choices[0].message.content)
        if not result:
            raise ValueError(f"No api result for scene: {scene.scene_id}")
        # DEBUG: create logfiles if debug mode activated
        if self.cfg.debug_mode:
            self._debug_llm_call(prompt, result)
        return result


class SummaryProcessor:
    def __init__(self, book_json_path: str, config=None):
        # enable init with argument for testing; normal case create SceneConfig obj from config.py
        self.cfg = config if config is not None else SummaryConfig()
        # save path of book
        self.book_json_path = book_json_path
        # load book json & map into pydantic obj
        with open(book_json_path, mode="r", encoding="utf-8") as f:
            self.book_json = Book(**json.load(f))
        self.llm = SummaryCreatorLLM(self.cfg, self.book_json.meta.world_context, book_json_path)

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
        # get root narrative or reference summary depending on scene instruction attribute
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

    def _process_scenes(self, scene_range: Tuple):
        """
        - loop through specified scenes range to create running summary for each
        - range uses python semantics: start inclusive, end exclusive
        - e.g. (0, 3) processes scenes 0, 1, 2 -> scene 3 receives final summary
        - distinguish scene type: narrative vs. reference -> reference instruction value = "special"
        """
        # python-style range: use directly, no -1 needed
        for i in range(scene_range[0], scene_range[1]):
            print(f"Starting processing scene id: {self.book_json.scenes[i].scene_id}")
            # create flag for scene is narrativ type or reference
            is_narrative = True if self.book_json.scenes[i].instruction != "special" else False
            # calc novel progress of scene
            novel_progress = self._calc_novel_progress(self.book_json.scenes[i].scene_id)
            print("Query LLM ...")
            # get updated rolling summary from llm & format it for saving at scene obj
            new_running_summary = self.llm.get_llm_running_summary(
                self.book_json.scenes[i],
                novel_progress,
                is_narrative,
            )
            new_running_summary = self._format_running_summary(new_running_summary)
            # save new running summary at following scene
            self.book_json.scenes[i+1].running_summary = new_running_summary
            # use pydantic json model dump method to write obj into json
            with open(self.book_json_path, mode="w", encoding="utf-8") as f:
                json.dump(self.book_json.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
            print(f"Did save LLM summary to scense id: {self.book_json.scenes[i+1].scene_id}")

    def run(self, scene_range: Tuple[int, int] = None):
        """
        - validate scene range if user-provided, otherwise construct default to do all scenes
        - if scene processing starts with 1st scene, root summary must be inserted
        """
        len_scenes = self.book_json.meta.total_scenes
        # default: set scene range to process all scenes from start to end
        if scene_range is None:
            scene_range = (0, len_scenes)
        else:
            # only validate user-provided range
            if scene_range[0] < 0:
                sys.exit("Scene range logic error: start must be >= 0")
            if scene_range[1] > len_scenes:
                sys.exit(f"Scene range logic error: end must be <= {len_scenes}")
            if scene_range[0] >= scene_range[1]:
                sys.exit("Scene range logic error: start must be < end")
        print(f"Starting process book: {self.book_json.meta.title} ...")
        # check if roots summary needs to be inserted at 1st scene
        if scene_range[0] == 0:
            print("Setting root summary at 1st scene manually ...")
            self._set_root_summary()
        print(f"Start processing scenes range: start {scene_range[0]} - end {scene_range[1]}...")
        print("---------------------------------------------")
        self._process_scenes(scene_range)
        print("---------------------------------------------")
        print("Operation finished")


if __name__ == "__main__":
    """
    - parse cli arguments for missing args & wrong format if optional scene range given
    - specifying optional scene range means summaries are created only for such; otherwise for all
    - if valid args:
        1. book json path is used to setup SummaryProcessor main obj
        2. scene range is used to start execution; if not specified, default is set in run method
    """
    if len(sys.argv) < 2:
        print("Usage: python summary_creator.py <input_book.json> 0,3 #Scene range 0,3 = Optional")
        print("Optional Scene range (0,3): python semantics: start inclusive, end exclusive")
        sys.exit(2)
    else:
        scene_range = None
        if len(sys.argv) == 3:
            try:
                parts = sys.argv[2].split(",")
                scene_range = (int(parts[0]), int(parts[1]))
            except (ValueError, IndexError):
                print("Invalid range format. Use: start,end (e.g., 0,10)")
                sys.exit(2)
        sp = SummaryProcessor(sys.argv[1])
        sp.run(scene_range)
