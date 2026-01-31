"""
Compress states of novel narrative into Rolling Summaries (like LSTM but in natural language) along
Semantic Scenes as timesteps. Each Semantic Scene's Rolling Summary attribute contains compressed
Narrative: what happened so far up to this specific Semantic Scene?

Process:
- Create Root Summary for scene 1 with empty "story begins" values
- Take world context + running summary current scene (n) + text current scene (n) & construct prompt
- Query LLM with JSON response enforcement schema
- If LLM "create summary" response too long:
  - Execute follow-up LLM compress calls on the previous response content
  - Higher temperature compared to "create summary" calls
  - Try up to 3 times using same input
- If all compress calls fail to deliver response under token threshold
  - Take response of last compress call
  - Log violation
- Map LLM response into final str format and save at book json scene obj
- Take this new gen running summary to construct prompt to create running summary for next scene...
"""

import sys
import argparse
from src.utils import parse_scene_range
import json
import os
from src.config import (
    Book, Scene, get_tokenizer, SummaryConfig, RunningSummary, Stats,
    get_root_summary_narrative, get_root_summary_reference
)
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Tuple
import logging
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
    def __init__(
        self,
        config: SummaryConfig,
        world_context: str,
        stats: Stats,
        logger: logging.Logger
    ):
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
            base_url=self.cfg.api_base_url,
            api_key=api_key,
            max_retries=self.cfg.api_max_retries
        )
        # stats obj to track progress
        self.stats = stats
        # init logger
        self.logger = logger

    def _construct_prompt_summary(
            self,
            scene: Scene,
            novel_progress: int,
            is_narrative: bool
    ) -> str:
        """
        - construct prompt for case "summary creation" on base of .md prompt files
        - content from this script added to .md prompt files:
            - world_context, novel_progress, rolling summary, scene text
        """
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

    def _construct_prompt_compress(self, prompt: str, response: Dict, amount_words: int) -> str:
        """
        - construct prompt for case "compress summary" on base of previous summary prompt & response
        - no .md prompt files in this special case, instead all defined in this method
        """
        return f"""
<system>
{self.prompt_system}
</system>

<instruction>
CRITICAL: You received following prompt earlier: {prompt}
You generated the following response for it: {json.dumps(response, indent=2)}
It had {amount_words} words when counting only the json values.
**BUT the maximum of total words is {self.cfg.max_words}**.

You must now compress the response by at least {amount_words - self.cfg.max_words} words to bring
it into the valid max word range.

**Do it in a way, which keeps the most relevant content to fullfill the task best possible
but within the constraints**. Try to preserve emotional turning points and character motivations.
Cut repetition and scene logistics first.
</instruction>
"""

    def _compress_summary(
            self,
            scene: Scene,
            prompt: str,
            response: Dict,
            amount_words: int
    ) -> Dict:
        """
        - if llm created running summary was greater than allowed max words, compress it
        - send prompt & response to llm again and instruct it to compress the responsed
        - do max x runs (defined in cfg) using same prompt each time for well compressed response
        - 1x retry loop for invalid response (gemini glitch) for each run
        """
        adapted_prompt = self._construct_prompt_compress(prompt, response, amount_words)
        # max x runs to get well compressed summary; response len decisive if break loop or continue
        for run in range(self.cfg.max_compress_attempts):
            # max 1 retry per run for json error: response format decisive if break loop or continue
            for attempt in range(2):
                # log full prompt to logfile before llm query
                self.logger.debug(
                    f"\n=== SUMMARY COMPRESSION: SCENE {scene.scene_id} PROMPT START ===\n"
                    f"{adapted_prompt}\n"
                    f"=== SUMMARY COMPRESSION: SCENE {scene.scene_id} PROMPT END ==="
                )
                compressed_response = self.client.chat.completions.create(
                    model=LLM,
                    messages=[{"role": "user", "content": adapted_prompt}],
                    temperature=0.5,  # higher vs. summary creation -> must express in other words
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
                    f"\n=== SUMMARY COMPRESSION: SCENE {scene.scene_id} RESPONSE START ===\n"
                    f"{compressed_content}\n"
                    f"=== SUMMARY COMPRESSION: SCENE {scene.scene_id} RESPONSE END ==="
                )
                # catch json deserialize error
                try:
                    # parse into python dict rep & count words
                    compressed_result = json.loads(compressed_content)
                    break
                except json.JSONDecodeError as e:
                    if attempt == 0:
                        self.logger.warning(f"Invalid JSON response at scene {scene.scene_id}: {e}")
                        continue
                    raise
            # count words from LLM response (dict values) & update stats
            total_words = sum(len(str(v).split()) for v in compressed_result.values())
            self.logger.info(f"Compress run # {run}: LLM response amount words: {total_words}")
            self.stats.compress_runs += 1
            # if response valid return
            if total_words <= (self.cfg.max_words + self.cfg.max_words_buffer):
                self.logger.info(f"Compressed successfully after: run # {run}")
                self.stats.compressed_successfully += 1
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
        - 1x retry loop with exception control flow for invalid json format response (gemini glitch)
        """
        prompt = self._construct_prompt_summary(scene, novel_progress, is_narrative)
        for attempt in range(2):
            # log full prompt to logfile before llm query
            self.logger.debug(
                f"\n=== SUMMARY CREATION: SCENE {scene.scene_id} PROMPT START ===\n"
                f"{prompt}\n"
                f"=== SUMMARY CREATION: SCENE {scene.scene_id} PROMPT END ==="
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
                f"\n=== SUMMARY CREATION: SCENE {scene.scene_id} RESPONSE START ===\n"
                f"{result_content}\n"
                f"=== SUMMARY CREATION: SCENE {scene.scene_id} RESPONSE END ==="
            )
            # catch json deserialize error
            try:
                # parse into python dict rep & count words
                result = json.loads(result_content)
                break
            # at this certain error try 1x again with same prompt & log warning; next time: crash it
            except json.JSONDecodeError as e:
                if attempt == 0:
                    self.logger.warning(f"Invalid JSON response at scene {scene.scene_id}: {e}")
                    continue
                raise
        # count words from LLM response (dict values) & update stats / logs
        total_words = sum(len(str(v).split()) for v in result.values())
        self.logger.info(f"Summary: LLM response amount words: {total_words}")
        self.stats.created += 1
        # if llm response not in total words constraint + buffer, compress it
        if total_words > (self.cfg.max_words + self.cfg.max_words_buffer):
            self.stats.compressed += 1
            result = self._compress_summary(scene, prompt, result, total_words)
        # count words final response (compressed or not) to save at stats word counter
        self.stats.total_words += sum(len(str(v).split()) for v in result.values())
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
        self.stats = Stats()
        # init logger
        self.logger = self._init_logger()
        # init llm
        self.llm = SummaryCreatorLLM(
            self.cfg,
            self.book_json.meta.world_context,
            self.stats,
            self.logger
        )

    def _init_logger(self) -> logging.Logger:
        """ setup logging for module with params from config.py """
        # set logfile dir, name & path
        os.makedirs(self.cfg.debug_dir, exist_ok=True)
        ts = datetime.now().strftime("%H%M%S")
        book_name = os.path.basename(self.book_json_path).removesuffix(".json")
        log_path = os.path.join(self.cfg.debug_dir, f"{book_name}_summary_{ts}.log")
        # setup logger
        logger = logging.getLogger(__name__)
        # set to debug at highest level
        logger.setLevel(logging.DEBUG)
        # guard to prevent duplicate logging
        if logger.hasHandlers():
            logger.handlers.clear()
        # create formatters: file detailed: [Time] [Level] Message; console minimal
        file_formatter = logging.Formatter(
            # We add .{msecs:03.0f} right after {asctime}
            fmt="[{asctime}.{msecs:03.0f}] [{levelname}] {message}",
            datefmt="%Y-%m-%d %H:%M:%S",
            style="{"
        )
        console_formatter = logging.Formatter(
            fmt="{message}",
            style="{"
        )
        # file_handler
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)  # file gets everything
        # console_handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)  # console only INFO and above (hide DEBUG noise)
        # add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    @staticmethod
    def _format_running_summary(summary_dict: Dict) -> str:
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
        total = len(self.book_json.scenes)
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
            try:
                new_running_summary = self.llm.create_summary(
                    self.book_json.scenes[i],
                    novel_progress,
                    is_narrative,
                )
            except Exception:
                self.logger.exception(f"Failed process scene {self.book_json.scenes[i].scene_id}")
                raise
            # bring dict response into target .md format to save at json scene
            new_running_summary = self._format_running_summary(new_running_summary)
            # log amount tokens of summary in target format & save to stats
            amount_tokens = len(tokenizer.encode(new_running_summary))
            self.logger.info(f"Total amount tokens: {amount_tokens}")
            self.stats.total_tokens += amount_tokens
            if amount_tokens > self.cfg.max_tokens:
                self.stats.too_large += 1
            # save new running summary at following scene
            self.book_json.scenes[i+1].running_summary = new_running_summary
            # use pydantic json model dump method to write obj into json
            with open(self.book_json_path, mode="w", encoding="utf-8") as f:
                json.dump(self.book_json.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
            self.logger.info(f"Summary saved to scene id: {self.book_json.scenes[i+1].scene_id}")
            self.logger.info("---------------------------------------------")

    def _create_report(self) -> None:
        """ print report with stats collected during processing & config params """
        total_scenes = len(self.book_json.scenes)
        s = self.stats
        self.logger.info(f"Summaries created: {s.created} of {total_scenes}")
        self.logger.info(f"Compressed: {s.compressed} in {s.compress_runs} runs")
        self.logger.info(f"Too large (>{self.cfg.max_tokens} tokens): {s.too_large}")
        # calc shares with division by zero guard
        if s.created > 0:
            pct = int((s.compressed / s.created) * 100)
            self.logger.info(f"Share needing compression: {pct}%")
            pct = int((s.too_large / s.created) * 100)
            self.logger.info(f"Share too large: {pct}%")
            words_avg = s.total_words / s.created
            tokens_avg = s.total_tokens / s.created
            self.logger.info(f"Avg per summary: {words_avg:.1f} words, {tokens_avg:.1f} tokens")
        if s.compressed > 0:
            pct = int((s.compressed_successfully / s.compressed) * 100)
            self.logger.info(f"Share compression successful: {pct}%")
        # print relevant params used for this ops
        self.logger.info("---------------------------------------------")
        self.logger.info(f"Max tokens: {self.cfg.max_tokens}")
        self.logger.info(f"Max words: {self.cfg.max_words}")
        self.logger.info(f"Max words buffer: {self.cfg.max_words_buffer}")
        self.logger.info(f"Max compress attempts: {self.cfg.max_compress_attempts}")
        self.logger.info("---------------------------------------------")

    def run(self, scene_range: Tuple[int, int] = None):
        """
        - validate scene range if user-provided, otherwise construct default to do all scenes
        - if scene processing starts with 1st scene, root summary must be inserted
        """
        len_scenes = len(self.book_json.scenes)
        # default: set scene range to process all scenes from start to end
        if scene_range is None:
            scene_range = (0, len_scenes - 1)
        else:
            # only validate user-provided range
            if scene_range[0] < 0:
                raise ValueError("start must be >= 0")
            if scene_range[1] > len_scenes - 1:
                raise ValueError(f"end must be <= {len_scenes - 1}")
            if scene_range[0] >= scene_range[1]:
                raise ValueError("start must be < end")
        self.logger.info(f"Starting process book: {self.book_json.meta.title} ...")
        # check if roots summary needs to be inserted at 1st scene
        if scene_range[0] == 0:
            self.logger.info("Setting root summary at 1st scene manually ...")
            self._set_root_summary()
        self.logger.info(f"Processing scenes: start {scene_range[0]} - end {scene_range[1]}...")
        self.logger.info("---------------------------------------------")
        # execute summary creation for specified scenes
        self._process_scenes(scene_range)
        # create closing report
        self.logger.info("---------------------------------------------")
        self._create_report()
        self.logger.info("Operation finished")


def main():
    """
    cli entry point for summary creation on book json
    - default: for each scene in book json scene list a summary is created and saved at next scene
    - provide optional scene range arg to create summaries for only certain range of scenes
    - "Usage: python summary_creator.py <input_book.json> <start,end>"
    - attention:
        - Due to logic "process scene n -> creates summary for scene n+1", last scene not processed!
        - To process all of 18 total scenes -> specify: 0,17
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "book_path",
        help="path to book json file",
    )
    parser.add_argument(
        "scene_range",
        nargs="?",
        type=parse_scene_range,
        help="optional range as start,end (e.g. 0,10)",
    )
    args = parser.parse_args()
    sp = SummaryProcessor(args.book_path)
    sp.run(args.scene_range)


if __name__ == "__main__":
    main()
