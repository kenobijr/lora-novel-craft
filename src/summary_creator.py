"""
Compress states of novel narrative into Running Summaries (like LSTM but in natural language) along
Semantic Scenes as timesteps. Each Semantic Scene's Running Summary contains compressed Narrative:
what happened so far up to this specific Semantic Scene?
process:
- create root summary for scene 1 with empty "story begins" values
- take world context + running summary current scene (n) + text current scene (n), construct prompt
and query llm to produce running summary to be attached to following scene n+1:
- if response too long, do follow-up llm compress call with previous response content (2x max)
- repeat process until reaching last scene -> since no following scene exists, process stops here
- differentiate between narrative & reference scenes -> distinct prompts & root summaries used
"""

import argparse
import json
import os
from src.config import (
    TOKENIZER, MODEL_REGISTRY, BaseLLM, SummaryConfig, Book, Scene, RunningSummary, SummaryStats,
    get_root_summary_narrative, get_root_summary_reference
)
from src.utils import parse_range, init_logger, format_llm_response
from typing import Dict, Tuple
import logging


class SummaryCreatorLLM(BaseLLM):
    """
    - handles llm related logic: model / api / key / connections / ...
    - loading & saving base prompts, logger & openai client done via base class from config.py
    - constructing prompt with task-specific content & llm queries done in each subclass
    """
    def __init__(
        self,
        config: SummaryConfig,
        logger: logging.Logger,
        world_context: str,
        stats: SummaryStats,
    ):
        super().__init__(config, logger)
        # world context from book json needed for each llm call
        self.wc = world_context
        # load separate instruction for reference material scenes
        with open(self.cfg.prompt_instruction_reference, mode="r", encoding="utf-8") as f:
            self.prompt_instruction_ref = f.read()
        # stats obj to track progress
        self.stats = stats

    def _construct_prompt_create(
            self,
            scene: Scene,
            novel_progress: int,
            is_narrative: bool
    ) -> str:
        """
        - include world_context, novel_progress, running summary, scene text
        - differentiate between narrative & reference scenes
        """
        prompt_instruction = (
            self.prompt_instruction
            if is_narrative
            else self.prompt_instruction_ref
        )
        return f"""
<system>
{self.prompt_system}
</system>

<world_context>
{self.wc}
</world_context>

<current_running_summary>
NOVEL PROGRESS: {novel_progress}%
{scene.running_summary}
</current_running_summary>

<scene_text>
{scene.text}
</scene_text>

<instruction>
{prompt_instruction}
</instruction>
"""

    def _construct_prompt_compress(self, prompt: str, response: Dict, amount_words: int) -> str:
        """ prompt for case "compress summary" on base of previous summary prompt & llm response """
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
        - if running summary response greater than allowed max words, content must be compressed
        - send prompt & response to llm again with special instruction added
        - do max x runs (defined in cfg) using same prompt each time
        - if compressed response < token threshold, return; otherwise return last comp response
        """
        adapted_prompt = self._construct_prompt_compress(prompt, response, amount_words)
        # max x runs to get compressed summary; break with return depending on response len
        for run in range(self.cfg.max_compress_attempts):
            for attempt in range(self.cfg.query_retry):
                self.logger.debug(
                    f"\n=== SUMMARY COMPRESSION: SCENE {scene.scene_id} PROMPT START ===\n"
                    f"{adapted_prompt}\n"
                    f"=== SUMMARY COMPRESSION: SCENE {scene.scene_id} PROMPT END ==="
                )
                compressed_response = self.client.chat.completions.create(
                    model=self.cfg.llm,
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
                    **MODEL_REGISTRY.get(self.cfg.llm, {}),
                )
                compressed_content = compressed_response.choices[0].message.content
                self.logger.debug(
                    f"\n=== SUMMARY COMPRESSION: SCENE {scene.scene_id} RESPONSE START ===\n"
                    f"{compressed_content}\n"
                    f"=== SUMMARY COMPRESSION: SCENE {scene.scene_id} RESPONSE END ==="
                )
                try:
                    compressed_result = json.loads(compressed_content)
                    break
                except json.JSONDecodeError as e:
                    if attempt < self.cfg.query_retry - 1:
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
        - if response too large (> max words), send prompt & response to method _compress_summary
        - in this case: use return value from _compress_summary as final running summary
        - 1x retry loop with exception control flow for invalid json format response (gemini glitch)
        """
        prompt = self._construct_prompt_create(scene, novel_progress, is_narrative)
        # max 1 retry per run for json error: response format decisive if break loop or continue
        for attempt in range(self.cfg.query_retry):
            self.logger.debug(
                f"\n=== SUMMARY CREATION: SCENE {scene.scene_id} PROMPT START ===\n"
                f"{prompt}\n"
                f"=== SUMMARY CREATION: SCENE {scene.scene_id} PROMPT END ==="
            )
            response = self.client.chat.completions.create(
                model=self.cfg.llm,
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
                **MODEL_REGISTRY.get(self.cfg.llm, {}),
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
            except json.JSONDecodeError as e:
                if attempt < self.cfg.query_retry - 1:
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
        return result


class SummaryProcessor:
    def __init__(self, book_path: str, config=None):
        self.cfg = config if config is not None else SummaryConfig()
        self.book_path = book_path
        with open(book_path, mode="r", encoding="utf-8") as f:
            self.book_content = Book(**json.load(f))
        # setup output path in stage-specific dir
        book_name = os.path.basename(book_path).removesuffix(".json")
        self.output_path = os.path.join(self.cfg.output_dir, f"{book_name}.json")
        # setup logfile & init logger with it
        self.logger = init_logger(self.cfg.operation_name, self.cfg.debug_dir, book_name)
        # track stats for report creation at end
        self.stats = SummaryStats()
        # init llm
        self.llm = SummaryCreatorLLM(
            self.cfg,
            self.logger,
            self.book_content.meta.world_context,
            self.stats,
        )

    def _set_root_summary(self) -> None:
        """
        - set root summary at first scene manually
        - pydanctic obj -> py dict rep -> str to mirror llm response flow and use same logic
        - distinguish between narrative vs. reference root type
        """
        first_scene = self.book_content.scenes[0]
        # get root narrative or reference summary depending on scene instruction attribute
        is_narrative = True if first_scene.instruction != "special" else False
        root = get_root_summary_narrative() if is_narrative else get_root_summary_reference()
        root = format_llm_response(root.model_dump(), "Running Summary")
        first_scene.running_summary = root

    def _process_scenes(self, scene_range: Tuple):
        """
        - process scenes specified in range to create running summary for the following scene
        - (0, 3) processes scenes 0, 1, 2 -> scene 3 not processed, but receives final summary
        - distinguish scene type: narrative vs. reference -> reference instruction value = "special"
        - save & write to json book file at the end of processing scene
        """
        len_scenes = len(self.book_content.scenes)
        for i in range(scene_range[0], scene_range[1]):
            current_scene = self.book_content.scenes[i]
            next_scene = self.book_content.scenes[i+1]
            self.logger.info(f"Starting processing scene id: {current_scene.scene_id}")
            # create flag for scene is narrativ type or reference
            is_narrative = True if current_scene.instruction != "special" else False
            # calc novel progress of scene
            novel_progress = int(((current_scene.scene_id - 1) / len_scenes) * 100)
            self.logger.info("Query LLM ...")
            # get updated running summary from llm & format it for saving at scene obj
            try:
                new_running_summary = self.llm.create_summary(
                    current_scene,
                    novel_progress,
                    is_narrative,
                )
            except Exception:
                self.logger.exception(f"Failed process scene {current_scene.scene_id}")
                raise
            # bring dict response into target .md format to save at json scene
            new_running_summary = format_llm_response(new_running_summary, "Running Summary")
            # log amount tokens of summary in target format & save to stats
            amount_tokens = len(TOKENIZER.encode(new_running_summary))
            self.logger.info(f"Total amount tokens: {amount_tokens}")
            self.stats.total_tokens += amount_tokens
            if amount_tokens > self.cfg.max_tokens:
                self.stats.too_large += 1
            # save new running summary at following scene & save with pydantic json model dump
            next_scene.running_summary = new_running_summary
            with open(self.output_path, mode="w", encoding="utf-8") as f:
                json.dump(
                    self.book_content.model_dump(mode="json"),
                    f,
                    indent=2,
                    ensure_ascii=False
                )
            self.logger.info(f"Summary saved to scene id: {next_scene.scene_id}")
            self.logger.info("---------------------------------------------")

    def _create_final_report(self) -> None:
        """ print report with stats collected during processing & config params """
        total_scenes = len(self.book_content.scenes)
        s = self.stats
        op_label = self.cfg.operation_name.replace("_", " ").title()
        self.logger.info(f"{op_label} created: {s.created} of {total_scenes}")
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
            self.logger.info(f"Avg per {op_label}: {words_avg:.1f} words, {tokens_avg:.1f} tokens")
        if s.compressed > 0:
            pct = int((s.compressed_successfully / s.compressed) * 100)
            self.logger.info(f"Share compression successful: {pct}%")
        # print relevant params used for this ops
        self.logger.info("---------------------------------------------")
        self.logger.info(f"Max tokens: {self.cfg.max_tokens}")
        self.logger.info(f"Max words: {self.cfg.max_words}")
        self.logger.info(f"Max words buffer: {self.cfg.max_words_buffer}")
        self.logger.info(f"Max compress attempts: {self.cfg.max_compress_attempts}")
        self.logger.info(f"LLM used: {self.cfg.llm}")
        self.logger.info("---------------------------------------------")

    def run(self, scene_range: Tuple[int, int] | None = None) -> None:
        """
        - validate scene range if user-provided, otherwise construct default to do all scenes
        - if scene processing starts with 1st scene, root summary must be inserted
        """
        len_scenes = len(self.book_content.scenes)
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
        self.logger.info(f"Starting process book: {self.book_content.meta.title} ...")
        # roots summary needs to be manually inserted only at 1st scene
        if scene_range[0] == 0:
            self.logger.info("Setting root summary at 1st scene manually ...")
            self._set_root_summary()
        self.logger.info(f"Processing scenes: start {scene_range[0]} - end {scene_range[1]}...")
        self.logger.info("---------------------------------------------")
        # execute summary creation for specified scenes
        self._process_scenes(scene_range)
        # create closing report
        self.logger.info("---------------------------------------------")
        self._create_final_report()
        self.logger.info("------Operation completed successfully-------")
        return self.output_path


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
        type=parse_range,
        help="optional range as start,end (e.g. 0,10)",
    )
    args = parser.parse_args()
    sp = SummaryProcessor(args.book_path)
    sp.run(args.scene_range)


if __name__ == "__main__":
    main()
