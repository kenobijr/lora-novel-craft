"""
Split up a novel narrative from Chapters into smaller Semantic Scenes.
1. chapter text is splitted into text chunks
2. llm partitions these text chunks into atomic scenes along logical breakpoints (place, moood, ...)
3. atomic scenes are merged into final semantic scenes depending on token amount restraints
input args:
- json book file with narrative split by chapters
- chapter range to be be processed; default: all chapters are processed
"""

import os
import re
import json
import argparse
from src.config import API_KEY, TOKENIZER, SceneConfig, Book, Scene, ScenePartitioning, SceneStats
from src.utils import parse_range, init_logger
import logging
from openai import OpenAI
from typing import List, Tuple


# llm model = openrouter id
LLM = "qwen/qwen-2.5-72b-instruct"
# "google/gemini-2.5-pro"
# "qwen/qwen-2.5-72b-instruct"
# "google/gemini-2.0-flash-lite-001"


class SceneSplitterLLM:
    """
    - handles llm related logic: model / api / key / connections / ...
    - world_context directly delivered as str -> book metadata needed for llm call
    - manages & formats prompts / systemmessages
    """
    def __init__(
            self,
            config: SceneConfig,
            world_context: str,
            logger: logging.Logger,
            stats: SceneStats
    ):
        self.cfg = config
        self.wc = world_context
        # load prompts: systemmesage, input content description & instruction
        with open(self.cfg.prompt_system, mode="r", encoding="utf-8") as f:
            self.prompt_system = f.read()
        with open(self.cfg.prompt_instruction, mode="r", encoding="utf-8") as f:
            self.prompt_instruction = f.read()
        # init llm
        self.client = OpenAI(
            base_url=self.cfg.api_base_url,
            api_key=API_KEY,
            max_retries=self.cfg.api_max_retries
        )
        self.logger = logger
        self.stats = stats

    def _annotate_text_chunks(self, chapter: List[str]) -> str:
        """
        - llm must know pos and token amount of each text chunk to merge them within constraints
        - annotate each text chunks with consecutive number and token amount as header
        - merge it into combined str in following target format:
        [C:1|Tok:23] Example text 123....
        [C:2|Tok:4] Example 456 .....
        """
        lines = []
        for i, chunk in enumerate(chapter, start=1):
            tok_chunk = len(TOKENIZER.encode(chunk))
            lines.append(f"[C:{i}|Tok:{tok_chunk}] {chunk}")
        return "\n".join(lines)

    def _create_prompt(self, annotated_text_chunks: str) -> str:
        return f"""
<system>
{self.prompt_system}
</system>

<world_context>
{self.wc}
</world_context>

<text_chunks>
{annotated_text_chunks}
</text_chunks>

<instruction>
{self.prompt_instruction}
</instruction>
"""

    def get_scene_boundaries(self, text_chunks: str, chapter_idx: int) -> List[int]:
        """
        - scene target size in tokens via prompt: 400 - 1000 tokens; ~600-800 goldilocks
        - annotate chapter text chunks into llm format and create prompt
        - llm response returns list of dict with n end_chunks for each atomic semantic scene
        - each end_chunk contains int as value, which relates to position of text chunk list
        """
        # create systemmessage using annotated text chunks
        prompt = self._create_prompt(self._annotate_text_chunks(text_chunks))
        # prompt llm
        for attempt in range(self.cfg.json_parse_retries):
            self.logger.debug(
                f"\n=== CHAPTER PARTITIONING: Chapter # {chapter_idx} PROMPT START ===\n"
                f"{prompt}\n"
                f"=== CHAPTER PARTITIONING: Chapter # {chapter_idx} PROMPT END ==="
            )
            response = self.client.chat.completions.create(
                model=LLM,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "scene_partition",
                        "strict": True,
                        "schema": ScenePartitioning.model_json_schema()
                    }
                },
                # fits only on qwen3!!!
                extra_body={
                    "provider": {
                        "only": ["DeepInfra"]
                    }
                }
            )
            # grab content in raw json for logging
            result_content = response.choices[0].message.content
            # log llm response before parsing / formatting
            self.logger.debug(
                f"\n=== CHAPTER PARTITIONING: Chapter # {chapter_idx} RESPONSE START ===\n"
                f"{result_content}\n"
                f"=== CHAPTER PARTITIONING: Chapter # {chapter_idx} RESPONSE END ==="
            )
            try:
                result = json.loads(result_content)
                break
            except json.JSONDecodeError as e:
                if attempt == 0:
                    self.logger.warning(f"Invalid JSON response at Chapter {chapter_idx}: {e}")
                    continue
                raise
        # extract scene boundaries (cut llm intermediate data) and validate logically
        boundaries = [scene["chunk_boundary"] for scene in result["scenes"]]
        # 1. check strictly increasing -> current boundary must be greater than the one before
        last = 0
        for i, boundary in enumerate(boundaries):
            if boundary <= last:
                # no logging necessary -> exceptions from llm query caught & logged downstream
                raise ValueError(f"Atomic Sem Scene {i}: boundary {boundary} <= previous {last}")
            last = boundary
        # 2. chunk_boundary of final scene must equal total amount of text chunks in chapter
        # correct if llm calcs falsely; otherwise text could be lost
        len_chunks = len(text_chunks)
        if last < len_chunks:
            diff = len_chunks - last
            self.logger.info("---------------------------------------------")
            self.logger.warning(f"LLM missed chunks: Ended at {last} vs. {len_chunks})")
            self.stats.invalid_partitioning += 1
            # force extend the very last scene to include the missing chunks
            boundaries[-1] = len_chunks
            self.logger.info(f"Auto-correct last chunk boundary completed for diff of {diff}")
            self.logger.info("---------------------------------------------")
        # return flattened list of chunk boundaries ints
        return boundaries


class SceneProcessor:
    """
    - handles all chapter to semantic scenes mapping logic
    - all llm related logic is done by SceneSplitterLLM, which is instantiated by SceneProcessor
    """
    def __init__(self, book_path: str, config=None):
        self.cfg = config if config is not None else SceneConfig()
        self.book_path = book_path
        # unpack dict from json.load into kw arguments -> 2. create pydantic book obj (=validate)
        with open(book_path, mode="r", encoding="utf-8") as f:
            self.book_content = Book(**json.load(f))
        # setup logger & stats
        book_name = os.path.basename(book_path).removesuffix(".json")
        self.logger = init_logger(self.cfg.operation_name, self.cfg.debug_dir, book_name)
        self.stats = SceneStats(original_word_count=self.book_content.meta.word_count)
        self.llm = SceneSplitterLLM(
            self.cfg,
            self.book_content.meta.world_context,
            self.logger,
            self.stats,
        )

    def _chunk_chapter(self, chapter: str) -> List[str]:
        """
        - split up chapter text into into list of smaller text chunks
        - llm will merge these text chunks into atomic semantic scenes
        - each text_block must have min size, so llm is not crowded with tiny chunks
        """
        # split text along \n\n and filter empty paragraphs within each chapter
        paragraphs = [p.strip() for p in chapter.split("\n\n") if p.strip()]
        text_chunks = []
        bucket = ""
        bucket_counter = 0
        for p in paragraphs:
            p_tok = len(TOKENIZER.encode(p))
            # case 1: bucket is empty
            if not bucket:
                # if atomic paragraph is greater than min size append it to p_blocks else bucket
                if p_tok >= self.cfg.chunk_min_tokens:
                    text_chunks.append(p)
                else:
                    # in this case bucket is always empty, so add without \n\n added to p
                    bucket += p
                    bucket_counter += p_tok
            # case 2: content in bucket
            else:
                # if p & bucket content are greater than threshold, empty bucket; else add to it
                if bucket_counter + p_tok >= self.cfg.chunk_min_tokens:
                    text_chunks.append(f"{bucket}\n\n{p}")
                    bucket = ""
                    bucket_counter = 0
                else:
                    bucket += f"\n\n{p}"
                    bucket_counter += p_tok
        # flush bucket after loop if not empty
        if bucket:
            text_chunks.append(bucket)
        return text_chunks

    @staticmethod
    def _extract_chapter_metadata(chapter: str) -> Tuple[int, str | None]:
        """
        - extract chapter idx & title (if available) from raw chapter str
        - format always like # Chapter 1; with title: # Chapter 1: Title
        """
        first_line = chapter.split("\n")[0]
        match = re.match(r"^#\s*Chapter\s+(\d+)(?::\s*(.+))?", first_line)
        if not match:
            raise ValueError(f"Invalid chapter header: {first_line}")
        return int(match.group(1)), match.group(2)

    @staticmethod
    def _apply_partitioning(text_chunks: List[str], partitioning: List[int]) -> List[str]:
        """
        - merge text chunks into llm cut atomic scenes using scene partitioning
        - partitioning ints represent chunk_boundaries relating to list position of text_chunks
        """
        prev_end = 0
        atomic_scenes = []
        for chunk_boundary in partitioning:
            end = chunk_boundary
            atomic_scenes.append("\n\n".join(text_chunks[prev_end:end]))
            prev_end = end
        return atomic_scenes

    @staticmethod
    def _merge_atomic_scenes(atomic_scenes: List[str], scene_max_tokens: int) -> List[str]:
        """
        - take atomic scenes and merge them until specified max scene token len
        - atomic scenes were cut with optimal semantic breakpoints (prompt: 400 - 1000) in mind
        - how many of these are merged into final semantic scenes, depends on train len restraints
        """
        semantic_scenes = []
        token_counter = 0
        running_scene = ""
        for scene in atomic_scenes:
            tok_current = len(TOKENIZER.encode(scene))
            # if running scene, together with current scene, under threshold -> add up
            if token_counter + tok_current <= scene_max_tokens:
                running_scene += ("\n\n" + scene) if running_scene else scene
                token_counter += tok_current
            # otherwise finalise & reset running scene & counter; add current scene after it
            else:
                # append running scene to list as final processed scene
                semantic_scenes.append(running_scene)
                # reset running scene & counter
                running_scene = ""
                token_counter = 0
                # add current scene to running scene & update counter
                running_scene += scene
                token_counter += tok_current
        # if residual after loop, append rest from running scene
        if running_scene:
            semantic_scenes.append(running_scene)
        return semantic_scenes

    def _save_scenes(
            self,
            semantic_scenes: List[str],
            chapter_idx: int,
            chapter_title: str | None,
    ) -> None:
        """
        - create scene objects for each scene in list in pydantic specified format
        - save scene objects to book content
        - write to target file
        """
        # create raw scenes and append to book content
        for content in semantic_scenes:
            new_scene = Scene(
                chapter_index=chapter_idx,
                chapter_title=chapter_title,
                text=content
            )
            self.book_content.scenes.append(new_scene)
        # update all scene_id's
        for i, scene in enumerate(self.book_content.scenes, start=1):
            scene.scene_id = i
        # use pydantic json model dump method to write to json
        with open(self.book_path, mode="w", encoding="utf-8") as f:
            json.dump(self.book_content.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
        self.logger.info(f"Scenes of Chapter {chapter_idx} saved to {self.book_path}")

    def _process_chapters(self, chapter_range: Tuple[int, int]):
        """
        - process chapters specified in range to create semantic scenes for each
        - convert chapter text into text chunks -> atomic scenes -> semantic scenes
        """
        for i in range(chapter_range[0], chapter_range[1]):
            current_chapter = self.book_content.chapters[i]
            # extract chapter number & title metadata from raw chapter str
            chapter_idx, chapter_title = self._extract_chapter_metadata(current_chapter)
            self.logger.info(f"Started processing Chapter: # {chapter_idx}; title: {chapter_title}")
            # split chapter str into text chunks
            text_chunks = self._chunk_chapter(current_chapter)
            chunk_amount = len(text_chunks)
            chunk_tokens = sum(len(TOKENIZER.encode(chunk)) for chunk in text_chunks)
            chunk_avg = chunk_tokens / chunk_amount
            self.stats.chunk_amount += chunk_amount
            self.stats.chunk_tokens += chunk_tokens
            self.logger.info(f"Gen {chunk_amount} Text Chunks; Avg: {chunk_avg:,.2f} tok")
            # query llm to get partitioning schema to map text chunks into atomic semantic scenes
            self.logger.info("Query LLM for scene partitioning...")
            try:
                scene_partitions = self.llm.get_scene_boundaries(text_chunks, chapter_idx)
            except Exception:
                self.logger.exception(f"LLM query error scene partitioning Chapter # {chapter_idx}")
                raise
            self.logger.info(f"LLM partitioning response -> Amount scenes: {len(scene_partitions)}")
            # apply partitioning on text chunks to get llm cut atomic scenes; log stats
            atomic_scenes = self._apply_partitioning(text_chunks, scene_partitions)
            atomic_amount = len(atomic_scenes)
            atomic_tokens = sum(len(TOKENIZER.encode(scene)) for scene in atomic_scenes)
            atomic_avg = atomic_tokens / atomic_amount
            self.stats.atomic_amount += atomic_amount
            self.stats.atomic_tokens += atomic_tokens
            self.logger.info(f"Gen {atomic_amount} Atomic Scenes; Avg: {atomic_avg:,.2f} tok")
            # merge atomic scenes into bigger semantic scenes depending on train token constraints
            semantic_scenes = self._merge_atomic_scenes(atomic_scenes, self.cfg.scene_max_tokens)
            semantic_amount = len(semantic_scenes)
            semantic_tokens = sum(len(TOKENIZER.encode(scene)) for scene in semantic_scenes)
            semantic_avg = semantic_tokens / semantic_amount
            self.stats.semantic_amount += semantic_amount
            self.stats.semantic_tokens += semantic_tokens
            self.logger.info(f"Gen {semantic_amount} Semantic Scenes; Avg: {semantic_avg:,.2f} tok")
            # create scene objects, save them and write to output file
            self._save_scenes(semantic_scenes, chapter_idx, chapter_title)

    def _create_final_report(self) -> None:
        s = self.stats
        b = self.book_content
        # general
        self.logger.info(f"Converted {b.meta.total_chapters} Chapters")
        self.logger.info(f"into {b.meta.total_scenes} Semantic Scenes")
        avg_scene_per_chapter = b.meta.total_scenes / b.meta.total_chapters
        self.logger.info(f"Avg Scenes / Chapter: {avg_scene_per_chapter:.1f}")
        self.logger.info(f"BEFORE: total word count: {s.original_word_count:,}")
        self.logger.info(f"AFTER: total word count: {b.meta.word_count:,}")
        # processing
        chunk_avg = s.chunk_tokens / s.chunk_amount
        self.logger.info(f"Gen {s.chunk_amount} Text Chunks; Avg: {chunk_avg:,.2f} tokens")
        atomic_avg = s.atomic_tokens / s.atomic_amount
        self.logger.info(f"Gen {s.atomic_amount} Atomic Scenes; Avg: {atomic_avg:,.2f} tokens")
        semantic_avg = s.semantic_tokens / s.semantic_amount
        self.logger.info(f"Gen {s.semantic_amount} Semantic Scenes;Avg: {semantic_avg:,.2f} tokens")
        self.logger.info(f"Invalid LLM partitionings (auto-corrected): {s.invalid_partitioning}")
        # print relevant params used for this ops
        self.logger.info("---------------------------------------------")
        self.logger.info(f"Semantic Scene max token: {self.cfg.scene_max_tokens}")
        self.logger.info(f"Text Chunk min token: {self.cfg.chunk_min_tokens}")
        self.logger.info(f"LLM used: {LLM}")

    def run(self, chapter_range: Tuple[int, int] | None = None) -> None:
        """
        - validate or create chapter range to be processed
        - trigger chapter processing with semantic scene creation & log some stats
        - delete chapter text & update book meta stats after processing -> write to book file
        """
        len_chapters = len(self.book_content.chapters)
        # default: set chapter range to process all chapters from start to end
        if chapter_range is None:
            chapter_range = (0, len_chapters)
        else:
            # only validate user-provided range
            if chapter_range[0] < 0:
                raise ValueError("start must be >= 0")
            if chapter_range[1] > len_chapters:
                raise ValueError(f"end must be <= {len_chapters}")
            if chapter_range[0] >= chapter_range[1]:
                raise ValueError("start must be < end")
        self.logger.info(f"Starting process book: {self.book_content.meta.title} ...")
        self.logger.info(f"Processing scenes: start {chapter_range[0]} - end {chapter_range[1]}...")
        self.logger.info("---------------------------------------------")
        self._process_chapters(chapter_range)
        self.logger.info("---------------------------------------------")
        # update global word counter & scene counter at book meta
        full_text = " ".join([scene.text for scene in self.book_content.scenes])
        self.book_content.meta.word_count = len(full_text.split())
        self.book_content.meta.total_scenes = len(self.book_content.scenes)
        # delete processed chapters & write final state to file
        del self.book_content.chapters[chapter_range[0]:chapter_range[1]]
        with open(self.book_path, mode="w", encoding="utf-8") as f:
            json.dump(self.book_content.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
        # create closing report
        self._create_final_report()
        self.logger.info("------Operation completed successfully-------")


def main():
    """
    cli entry point for converting chapters into smaller semantic scene chunks
    - Usage: python scene_creator.py <input_book.json> <start,end>
    - provide optional chapter range arg to process only certain range of chapters
    - default: each chapter at input book json is concerted into semantic scenes
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "book_path",
        help="path to input book json file",
    )
    parser.add_argument(
        "chapter_range",
        nargs="?",
        type=parse_range,
        help="optional range as start,end (e.g. 0,3)",
    )
    args = parser.parse_args()
    bp = SceneProcessor(args.book_path)
    bp.run(args.chapter_range)


if __name__ == "__main__":
    main()
