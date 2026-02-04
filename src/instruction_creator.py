import json
import argparse
import os
from openai import OpenAI
from src.utils import parse_range, init_logger
from src.config import (
    API_KEY, TOKENIZER, MODEL_REGISTRY,
    InstructionConfig, Book, Scene, SceneInstruction, InstructionStats
)
from typing import Tuple, Dict
import logging


class InstructionCreatorLLM:
    def __init__(
        self,
        config: InstructionConfig,
        world_context: str,
        logger: logging.Logger,
        stats: InstructionStats
    ):
        self.cfg = config
        self.wc = world_context
        # load prompts to use at llm call
        with open(self.cfg.prompt_system, mode="r", encoding="utf-8") as f:
            self.prompt_system = f.read()
        with open(self.cfg.prompt_instruction, mode="r", encoding="utf-8") as f:
            self.prompt_instruction = f.read()
        # load inference systemmessage to add it as metadata content to prompt
        with open(self.cfg.inference_systemmessage, mode="r", encoding="utf-8") as f:
            self.inference_systemmessage = f.read()
        self.logger = logger
        self.stats = stats
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=self.cfg.api_base_url,
            max_retries=self.cfg.api_max_retries,
        )

    def _construct_prompt(self, scene: Scene, novel_progress: int) -> str:
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

<author_ai_persona>
{self.inference_systemmessage}
</author_ai_persona>

<scene_text>
{scene.text}
</scene_text>

<instruction>
{self.prompt_instruction}
</instruction>
"""

    def create_instruction(self, scene: Scene, novel_progress: int):
        prompt = self._construct_prompt(scene, novel_progress)
        for attempt in range(self.cfg.json_parse_retries):
            # log full prompt to logfile before llm query
            self.logger.debug(
                f"\n=== INSTRUCTION CREATION: SCENE {scene.scene_id} PROMPT START ===\n"
                f"{prompt}\n"
                f"=== INSTRUCTION CREATION: SCENE {scene.scene_id} PROMPT END ==="
            )
            response = self.client.chat.completions.create(
                model=self.cfg.llm,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "scene_instruction",
                        "strict": True,
                        "schema": SceneInstruction.model_json_schema()
                    }
                },
                **MODEL_REGISTRY.get(self.cfg.llm, {}),
            )
            # grab content in raw json for logging
            result_content = response.choices[0].message.content
            # log llm response before parsing / formatting
            self.logger.debug(
                f"\n=== INSTRUCTION CREATION: SCENE {scene.scene_id} RESPONSE START ===\n"
                f"{result_content}\n"
                f"=== INSTRUCTION CREATION: SCENE {scene.scene_id} RESPONSE END ==="
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
        self.logger.info(f"Instruction: LLM response amount words: {total_words}")
        self.stats.created += 1
        self.stats.total_words += sum(len(str(v).split()) for v in result.values())
        return result


class InstructionProcessor:
    def __init__(self, book_path: str, config=None):
        self.cfg = config if config is not None else InstructionConfig()
        self.book_path = book_path
        with open(book_path, mode="r", encoding="utf-8") as f:
            self.book_content = Book(**json.load(f))
        book_name = os.path.basename(book_path).removesuffix(".json")
        self.logger = init_logger(self.cfg.operation_name, self.cfg.debug_dir, book_name)
        self.stats = InstructionStats()
        self.llm = InstructionCreatorLLM(
            self.cfg,
            self.book_content.meta.world_context,
            self.logger,
            self.stats,
        )

    @staticmethod
    def _format_instruction(response: Dict) -> str:
        """
        - take instruction as python dict and map into target .md styled str format
        - stay in sync to ## .md format of earlier generated content: scenes, world_context
        """
        lines = ["# Instruction\n"]
        for key, value in response.items():
            header = "## " + key.upper().replace("_", " ")
            lines.append(f"{header}: {value}")
        return "\n".join(lines)

    def _create_final_report(self) -> None:
        """ print report with stats collected during processing & config params """
        total_scenes = len(self.book_content.scenes)
        s = self.stats
        op_label = self.cfg.operation_name.replace("_", " ").title()
        self.logger.info(f"{op_label} created: {s.created} of {total_scenes}")
        self.logger.info(f"Too large (>{self.cfg.max_tokens} tokens): {s.too_large}")
        if s.created > 0:
            pct = int((s.too_large / s.created) * 100)
            self.logger.info(f"Share too large: {pct}%")
            words_avg = s.total_words / s.created
            tokens_avg = s.total_tokens / s.created
            self.logger.info(f"Avg per {op_label}: {words_avg:.1f} words, {tokens_avg:.1f} tokens")
        self.logger.info("---------------------------------------------")
        self.logger.info(f"Max tokens: {self.cfg.max_tokens}")
        self.logger.info(f"Max words: {self.cfg.max_words}")
        self.logger.info(f"LLM used: {self.cfg.llm}")
        self.logger.info("---------------------------------------------")

    def _process_scenes(self, scene_range: Tuple[int, int]) -> None:
        len_scenes = len(self.book_content.scenes)
        for i in range(scene_range[0], scene_range[1]):
            current_scene = self.book_content.scenes[i]
            self.logger.info(f"Starting processing scene id: {current_scene.scene_id}")
            novel_progress = int(((current_scene.scene_id - 1) / len_scenes) * 100)
            self.logger.info("Query LLM ...")
            try:
                new_instruction = self.llm.create_instruction(
                    current_scene,
                    novel_progress
                )
            except Exception:
                self.logger.exception(f"Failed process scene {current_scene.scene_id}")
                raise
            new_instruction = self._format_instruction(new_instruction)
            amount_tokens = len(TOKENIZER.encode(new_instruction))
            self.logger.info(f"Total amount tokens: {amount_tokens}")
            self.stats.total_tokens += amount_tokens
            # if tokens greater than boundary just log
            if amount_tokens > self.cfg.max_tokens:
                self.stats.too_large += 1
            current_scene.instruction = new_instruction
            with open(self.book_path, mode="w", encoding="utf-8") as f:
                json.dump(
                    self.book_content.model_dump(mode="json"),
                    f,
                    indent=2,
                    ensure_ascii=False
                )
            self.logger.info(f"Instruction saved to scene id: {current_scene.scene_id}")
            self.logger.info("---------------------------------------------")

    def run(self, scene_range: Tuple[int, int] | None = None) -> None:
        len_scenes = len(self.book_content.scenes)
        if scene_range is None:
            scene_range = (0, len_scenes)
        else:
            if scene_range[0] < 0:
                raise ValueError("start must be >= 0")
            if scene_range[1] > len_scenes:
                raise ValueError(f"end must be <= {len_scenes}")
            if scene_range[0] >= scene_range[1]:
                raise ValueError("start must be < end")
        self.logger.info(f"Starting process book: {self.book_content.meta.title} ...")
        self.logger.info(f"Processing scenes: start {scene_range[0]} - end {scene_range[1]}...")
        self.logger.info("---------------------------------------------")
        self._process_scenes(scene_range)
        self.logger.info("---------------------------------------------")
        self._create_final_report()
        self.logger.info("------Operation completed successfully-------")


def main():
    """
    cli entry point for instruction creation on book json
    - default: instruction is created for each scene in book json scene list
    - provide optional scene range arg to create instructions for only certain range of scenes
    - "Usage: python instruction_creator.py <input_book.json> <start,end>"
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
    p = InstructionProcessor(args.book_path)
    p.run(args.scene_range)


if __name__ == "__main__":
    main()
