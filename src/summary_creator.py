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
from typing import Dict, Tuple
import logging
from src.logger import setup_logger
from datetime import datetime


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
    def __init__(self, config: SummaryConfig, world_context: str, stats: Dict):
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
        # stats obj to track progress
        self.stats = stats
        # init logger
        self.logger = logging.getLogger("summary_engine")

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
NOVEL PROGRESS: {novel_progress}%
{scene.running_summary}
</current_rolling_summary>

<scene_text>
{scene.text}
</scene_text>

<instruction>
{prompt_instruction}
</instruction>
"""

    def _compress_summary(self, scene: Scene, prompt: str, response: Dict, amount_words: int):
        """
        - if llm created running summary was greater than allowed max words, compress it
        - send prompt & response to llm again and instruct it to compress the responsed
        - retry x times defined in cfg
        """
        # save copy of orig response; to send it back as is if llm fails
        adapted_prompt = f"""CRITICAL: You received following prompt earlier: {prompt}
You generated the following response for it: {json.dumps(response, indent=2)}
It had {amount_words} words when counting only the json values.
**BUT the maximum of total words is {self.cfg.max_words}**.

You must now compress the response by at least {amount_words - self.cfg.max_words} words to bring
it into the valid max word range.

**Do it in a way, which keeps the most relevant content to fullfill the task best possible
but within the constraints**. Try to preserve emotional turning points and character motivations.
Cut repetition and scene logistics first."""
        for run in range(self.cfg.max_retry + 1):
            # log full prompt to logfile before llm query
            self.logger.debug(
                f"\n=== SCENE {scene.scene_id} PROMPT START ===\n"
                f"{adapted_prompt}\n"
                f"=== SCENE {scene.scene_id} PROMPT END ==="
            )
            compressed_response = self.client.chat.completions.create(
                model=LLM,
                messages=[{"role": "user", "content": adapted_prompt}],
                temperature=0.2,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "running_summary",
                        "strict": True,
                        "schema": RunningSummary.model_json_schema()
                    }
                },
                # # only needed for qwen3!!!
                # extra_body={
                #     "provider": {
                #         "only": ["DeepInfra"]
                #     }
                # }
            )
            # grab content in raw json for logging
            compressed_content = compressed_response.choices[0].message.content
            # log llm response before parsing / formatting
            self.logger.debug(
                f"\n=== SCENE {scene.scene_id} RESPONSE START ===\n"
                f"{compressed_content}\n"
                f"=== SCENE {scene.scene_id} RESPONSE END ==="
            )
            # parse into python dict rep & count words
            compressed_result = json.loads(compressed_content)
            # count words from LLM response (dict values) & update stats
            total_words = sum(len(str(v).split()) for v in compressed_result.values())
            self.logger.info(f"Compress run # {run}: LLM response amount words: {total_words}")
            self.stats["compress_runs"] += 1
            # if response valid return
            if total_words <= self.cfg.max_words:
                self.logger.info(f"Compressed successfully after: run # {run}")
                self.stats["compressed_successfully"] += 1
                return compressed_result
        # if compression did fail in all iterations, return compressed version of last iteration
        self.logger.info(f"Compression failed; returned last run # {run} as response.")
        return compressed_result

    def create_summary(
            self,
            scene: Scene,
            novel_progress: int,
            is_narrative: bool
    ) -> dict:
        """
        - prompt llm to create updated running summary for scene
        - if response > max words, do max. 2 trimming llm calls with adapted prompt
        """
        prompt = self._construct_prompt(scene, novel_progress, is_narrative)
        # log full prompt to logfile before llm query
        self.logger.debug(
            f"\n=== SCENE {scene.scene_id} PROMPT START ===\n"
            f"{prompt}\n"
            f"=== SCENE {scene.scene_id} PROMPT END ==="
        )
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
            # # only needed for qwen3!!!
            # extra_body={
            #     "provider": {
            #         "only": ["DeepInfra"]
            #     }
            # }
        )
        # grab content in raw json for logging
        result_content = response.choices[0].message.content
        # log llm response before parsing / formatting
        self.logger.debug(
            f"\n=== SCENE {scene.scene_id} RESPONSE START ===\n"
            f"{result_content}\n"
            f"=== SCENE {scene.scene_id} RESPONSE END ==="
        )
        # parse into python dict rep & count words
        result = json.loads(result_content)
        # count words from LLM response (dict values) & update stats / logs
        total_words = sum(len(str(v).split()) for v in result.values())
        self.logger.info(f"Summary: LLM response amount words: {total_words}")
        self.stats["created"] += 1
        # if llm response not in total words constraint + buffer, compress it
        if total_words > (self.cfg.max_words + self.cfg.max_words_buffer):
            self.stats["compressed"] += 1
            result = self._compress_summary(scene, prompt, result, total_words)
        # count words final response (compressed or not) to save at stats word counter
        self.stats["total_words"] += sum(len(str(v).split()) for v in result.values())
        # finally return result in any case
        return result


class SummaryProcessor:
    def __init__(self, book_json_path: str, config=None):
        # enable init with argument for testing; normal case create SceneConfig obj from config.py
        self.cfg = config if config is not None else SummaryConfig()
        # save path of book
        self.book_json_path = book_json_path
        # save name of book file for logging
        # load book json & map into pydantic obj
        with open(book_json_path, mode="r", encoding="utf-8") as f:
            self.book_json = Book(**json.load(f))
        # track stats for summary creation ops
        self.stats = {
            "created": 0,
            "compressed": 0,
            "compress_runs": 0,
            "compressed_successfully": 0,
            "total_words": 0,
            "total_tokens": 0,
        }
        # init llm
        self.llm = SummaryCreatorLLM(self.cfg, self.book_json.meta.world_context, self.stats)
        # init logger
        ts = datetime.now().strftime("%H%M%S")
        book_name = os.path.basename(book_json_path).removesuffix(".json")
        log_path = os.path.join(self.cfg.debug_dir, f"{book_name}_summary_{ts}.log")
        self.logger = setup_logger(log_path)
        self.logger.info(f"Logger initialized. Processing book: {book_name}")

    def _format_running_summary(self, summary_dict: Dict) -> str:
        """
        - take running summary as python dict rep and map into target .md styled str format
        - stay in sync to ## .md format of earlier generated content: scenes, world_context
        """
        lines = ["# Running Summary\n"]
        for key, value in summary_dict.items():
            # transform snake_case key to markdown header: scene_end_state -> ## SCENE END STATE
            header = "## " + key.upper().replace("_", " ")
            lines.append(f"{header}: {value}")
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
            self.logger.info(f"Starting processing scene id: {self.book_json.scenes[i].scene_id}")
            # create flag for scene is narrativ type or reference
            is_narrative = True if self.book_json.scenes[i].instruction != "special" else False
            # calc novel progress of scene
            novel_progress = self._calc_novel_progress(self.book_json.scenes[i].scene_id)
            self.logger.info("Query LLM ...")
            # get updated rolling summary from llm & format it for saving at scene obj
            new_running_summary = self.llm.create_summary(
                self.book_json.scenes[i],
                novel_progress,
                is_narrative,
            )
            # bring dict response into target .md format to save at json scene
            new_running_summary = self._format_running_summary(new_running_summary)
            # log amount tokens of summary in target format & save to stats
            amount_tokens = len(tokenizer.encode(new_running_summary))
            self.logger.info(f"Total amount tokens: {amount_tokens}")
            self.stats["total_tokens"] += amount_tokens
            # save new running summary at following scene
            self.book_json.scenes[i+1].running_summary = new_running_summary
            # use pydantic json model dump method to write obj into json
            with open(self.book_json_path, mode="w", encoding="utf-8") as f:
                json.dump(self.book_json.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
            self.logger.info(f"Summary saved to scene id: {self.book_json.scenes[i+1].scene_id}")
            self.logger.info("---------------------------------------------")

    def run(self, scene_range: Tuple[int, int] = None):
        """
        - validate scene range if user-provided, otherwise construct default to do all scenes
        - if scene processing starts with 1st scene, root summary must be inserted
        """
        len_scenes = self.book_json.meta.total_scenes
        # default: set scene range to process all scenes from start to end
        if scene_range is None:
            scene_range = (0, len_scenes - 1)
        else:
            # only validate user-provided range
            if scene_range[0] < 0:
                sys.exit("Scene range logic error: start must be >= 0")
            if scene_range[1] > len_scenes - 1:
                sys.exit(f"Scene range logic error: end must be <= {len_scenes - 1}")
            if scene_range[0] >= scene_range[1]:
                sys.exit("Scene range logic error: start must be < end")
        self.logger.info(f"Starting process book: {self.book_json.meta.title} ...")
        # check if roots summary needs to be inserted at 1st scene
        if scene_range[0] == 0:
            self.logger.info("Setting root summary at 1st scene manually ...")
            self._set_root_summary()
        self.logger.info(f"Processing scenes: start {scene_range[0]} - end {scene_range[1]}...")
        self.logger.info("---------------------------------------------")
        # execute summary creation for specified scenes
        self._process_scenes(scene_range)
        # calc & create some states for the ops
        self.logger.info("---------------------------------------------")
        total_scenes = self.book_json.meta.total_scenes
        self.logger.info(f"Total summaries created: {self.stats["created"]} / {total_scenes}")
        self.logger.info(f"Total summaries compressed: {self.stats["compressed"]} in compress runs: {self.stats["compress_runs"]}")
        # calc shares with division by zero guard
        if self.stats["created"] > 0:
            pct = int((self.stats["compressed"] / self.stats["created"]) * 100)
            self.logger.info(f"Share needing compression: {pct}%")
        if self.stats["compressed"] > 0:
            pct = int((self.stats["compressed_successfully"] / self.stats["compressed"]) * 100)
            self.logger.info(f"Share compression successful: {pct}%")
        if self.stats["created"] > 0:
            words_avg = self.stats["total_words"] / self.stats["created"]
            tokens_avg = self.stats["total_tokens"] / self.stats["created"]
            self.logger.info(f"Avg per summary: {words_avg:.1f} words, {tokens_avg:.1f} tokens")
        self.logger.info("---------------------------------------------")
        self.logger.info("Operation finished")


if __name__ == "__main__":
    """
    - parse cli arguments for missing args & wrong format if optional scene range given
    - specifying optional scene range means summaries are created only for such; otherwise for all
    - if valid args:
        1. book json path is used to setup SummaryProcessor main obj
        2. scene range is used to start execution; if not specified, default is set in run method
    """
    if len(sys.argv) < 2:
        print("Usage: python summary_creator.py <input_book.json> [start,end]")
        print("Range uses array indices. For 18 scenes: 0,17 processes all.")
        print("Processing scene at index i saves summary to scene at index i+1.")
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
