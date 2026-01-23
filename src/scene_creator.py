"""
Split up a novel saved in .md into logical Semantic Scene chunks (smaller than chapters) as
perparation to use them in a dataset for LLM Finetuning.
- judge LLM is tasked to split each book chapter into "Atomic Scenes"
- after this scenes are merged into final Semantic Scenes up to a specified token boundary

Files needed via CLI:
1. input_book.md
- must contain cleaned book text for n chapters
- chapters must be separated by "# Chapter 1" or "# Chapter 2: Some title" as context anchors
2. output_book.json
- must be in specefied format and contain "world context" content about book (is send to llm)
"""
import sys
import os
import re
import json
from src.config import get_tokenizer, SceneConfig, Book, Scene, ScenePartitioning
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Tuple, Dict

# llm model = openrouter id
LLM = "qwen/qwen-2.5-72b-instruct"
# "google/gemini-2.5-pro"
# "qwen/qwen-2.5-72b-instruct"
# "google/gemini-2.0-flash-lite-001"
SYSTEM_MESSAGE = "./prompts/scene_splitting.md"

# load api key
load_dotenv()
api_key = os.getenv("OPEN_ROUTER_KEY")
if not api_key:
    raise ValueError("could not load API key...")

# load tokenizer
tokenizer = get_tokenizer()
if not tokenizer:
    raise ValueError("could not load tokenizer...")


class SceneSplitterLLM:
    """
    - handles llm related logic: model / api / key / connections / ...
    - world_context directly delivered as str; book metadata needed for llm call
    - manages & formats prompts / systemmessages
    """
    def __init__(self, world_context: str, book_json: str, config: SceneConfig):
        self.cfg = config
        self.wc = world_context
        # init system_message
        with open(SYSTEM_MESSAGE, mode="r", encoding="utf-8") as f:
            self.sm = f.read()
        # save cleaned book title for debugging
        self.title = os.path.basename(book_json).removesuffix(".json")
        # init llm
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=3  # standard SDK feature: try 3 times before giving up for certain errors
        )

    def _create_systemmessage(self, chapter_formatted: str) -> str:
        return f"""
<system>
{self.sm}
</system>

<world_context>
{self.wc}
</world_context>

<text_paragraphs>
{chapter_formatted}
</text_paragraphs>
"""

    def _debug_llm_call(self, prompt: str, response: str) -> None:
        """
        - switch on / off with config
        - if activated, on every llm call prompt & response are saved into debug folder
        """
        os.makedirs(self.cfg.debug_dir, exist_ok=True)
        ts = datetime.now().strftime("%H%M%S_%f")
        prompt_path = os.path.join(self.cfg.debug_dir, f"debug_prompt.{self.title}.{ts}.md")
        response_path = os.path.join(self.cfg.debug_dir, f"debug_llm.{self.title}.{ts}.json")
        with open(prompt_path, mode="w", encoding="utf-8") as f:
            f.write(prompt)
        with open(response_path, mode="w", encoding="utf-8") as f:
            json.dump(response, f, indent=2, ensure_ascii=False)

    def get_llm_scene_partitions(self, chapter_formatted: str, amount_p: int) -> List[Dict]:
        """
        - get formatted chapter content for 1 book chapter ready to send to llm
        - use amount_paragraphs for chapter to check llm response for consistency
        - return list of dicts with end_paragraph int for each semantic scene
        """
        # parse systemmessage
        full_prompt = self._create_systemmessage(chapter_formatted)
        # prompt llm
        response = self.client.chat.completions.create(
            model=LLM,
            messages=[{"role": "user", "content": full_prompt}],
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
        result = json.loads(response.choices[0].message.content)

        # # DEBUGGING without live llm calls // replace llm response by test file in same format
        # with open("./data/debug/json/test_scene_partition.json", mode="r", encoding="utf-8") as f:
        #     result = json.load(f)

        # base validation
        if not result:
            raise ValueError(f"No api result for prompt: {full_prompt}")

        # if debug mode activated prompt & llm response will be saved
        if self.cfg.debug_mode:
            self._debug_llm_call(full_prompt, result)

        # further validations
        # scenes are consecutive (no gaps/overlaps)
        last_boundary = 0
        for i, scene in enumerate(result["scenes"]):
            current_boundary = scene["end_paragraph"]
            # check: strictly increasing -> current boundary must be greater than the one before
            if current_boundary <= last_boundary:
                raise ValueError(
                    f"Validation Failed: Scene {i} boundary ({current_boundary}) "
                    f"is not greater than previous boundary ({last_boundary})."
                )
            last_boundary = current_boundary
        # end_paragraph final scene shouldequal total amount paragraphs in chapter
        # correct it llm calcs falsely; otherwise text could be lost
        if last_boundary < amount_p:
            diff = amount_p - last_boundary
            print("---------------------------------------------")
            print(f"WARN: LLM missed p: Ended at {last_boundary}vs.{amount_p}). Auto-correcting...")
            # force extend the very last scene to include the missing paragraphs
            result["scenes"][-1]["end_paragraph"] = amount_p
            print(f"correction completed; difference of {diff} corrected.")
            print("---------------------------------------------")

        # return validated list
        return result["scenes"]


class BookProcessor:
    """
    - steering the whole book to scene splitting process
    - pass world_context further & init scene splitting llm class with it
    - save scene objects to self.book_json during processing
    """
    def __init__(self, input_book_md: str, output_book_json: str, config=None):
        # get globals / config parameters from data_prep config dataclass
        self.cfg = config if config is not None else SceneConfig()
        # parse & save content from book md file: narrative as one formatted str
        with open(input_book_md, mode="r", encoding="utf-8") as f:
            self.raw_text = f.read()
        # file path target json file
        self.book_json_path = output_book_json
        # parse & save content from book json file: structured book meta data
        # 1. unpack dict from json.load into kw arguments -> 2. create pydantic book obj (=validate)
        with open(output_book_json, mode="r", encoding="utf-8") as f:
            self.book_json = Book(**json.load(f))
        # save original wc from book json sep for stats
        self.orig_wc = self.book_json.meta.word_count
        # init llm -> pass world context strfrom book json, book json path & config
        self.llm = SceneSplitterLLM(
            self.book_json.meta.world_context,
            output_book_json,
            self.cfg
        )
        # raw text splitted into chapters during processing
        self.chapters = None
        # list of final semantic scene objects as saved to target json
        self.scene_objects = None

    @staticmethod
    def _extract_chapter_metadata(chapter: List[str]) -> Tuple[int, str]:
        """
        - extract chapter idx & title (if available) from chapter in form chapter_paragraphs
        - format always like # Chapter 1; with title: # Chapter 1: Title
        """
        match = re.match(r'^#\s*Chapter\s+(\d+)(?::\s*(.+))?', chapter[0])
        chapter_index = int(match.group(1))
        chapter_title = match.group(2)
        return chapter_index, chapter_title

    def _format_paragraphs(self, chapter: List[str]) -> str:
        """
        - add consecutive numbers and token amount as headers to paragraphs
        - merge it all to 1 string ready for llm to consume
        - target format:
        [P:1|Tok:23] Example text 123....
        [P:2|Tok:4] Example 456 .....
        """
        lines = []
        for i, p in enumerate(chapter, start=1):
            tok_p = len(tokenizer.encode(p))
            lines.append(f"[P:{i}|Tok:{tok_p}] {p}")
        return "\n".join(lines)

    def _create_semantic_scenes(self, chapter_paragraphs: List[List[str]]) -> List[Tuple]:
        """
        process chapter by chapter to create semantic scenes:
        1. extract chapter metadata (idx, title) at start -> must be in 1st paragraph
        2. bring each chapter into llm format with numbered p + token amount
        3. send formatted str to llm + total amount paragraphs in chapter (for testing)
        -> get boundaries with "end_paragraph" back for semanctic scenes in form:
        [
            {
            "final_token_sum": "P1(4) + P2(95) ...+ P5(196) = 884.",
            "end_paragraph": 5
            },
        ]
        4. use llm partitions to merge paragraphs into semantic scenes -> List[List[str]]
        5. for return list of all scenes with tuples chapter metadata + scenes
        """
        all_scenes = []
        for chapter in chapter_paragraphs:
            # extract chapter metadata
            chapter_idx, chapter_title = self._extract_chapter_metadata(chapter)
            print(f"Started processing chapter: {chapter_idx}")
            # format & enrich chapter content for llm
            chapter_formatted = self._format_paragraphs(chapter)
            print(f"Prompting LLM for chapter: {chapter_idx} ...")
            # get atomic semantic scene boundaries for chapter by llm
            scene_partitions = self.llm.get_llm_scene_partitions(chapter_formatted, len(chapter))
            # combine paragraphs to scenes with partitioning dicts
            prev_end = 0
            atomic_scenes = []
            for scene_content in scene_partitions:
                end = scene_content["end_paragraph"]
                atomic_scenes.append("\n\n".join(chapter[prev_end:end]))  # directly join here
                prev_end = end
            # stats for scene size / amount of llm cut atomic scenes
            print(f"Received response by LLM with amount of Atomic Scenes: {len(atomic_scenes)}")
            avg_atomic_scene = sum(
                len(tokenizer.encode(scene)) for scene in atomic_scenes
            ) / len(atomic_scenes)
            print(f"Token avg per Atomic Scene before processing: {avg_atomic_scene:,.2f}")
            # merge atomic scenes into bigger target semantic scenes up to specified max size
            semantic_scenes = []
            token_counter = 0
            running_scene = ""
            for scene in atomic_scenes:
                tok_current = len(tokenizer.encode(scene))
                # if running scene, together with current scene, under threshold -> add up
                if token_counter + tok_current <= self.cfg.max_scene_size:
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
            # stats for final processed semantic scenes
            print(f"Amount Semantic scenes after processing: {len(semantic_scenes)}")
            avg_semantic_scene = sum(
                len(tokenizer.encode(scene)) for scene in semantic_scenes
            ) / len(semantic_scenes)
            print(f"Token avg per Semantic Scene after processing: {avg_semantic_scene:,.2f}")
            print("---------------------------------------------")
            # add all scenes to list with chapter metadata
            all_scenes.append((chapter_idx, chapter_title, semantic_scenes))

        return all_scenes

    def _save_scenes(self, scenes: List[Tuple[str, str, List]]) -> List:
        """
        - loop through chapter tuples & create consecutive scenes with text & chapter meta data
        - use total scenes from book json metadata to create scene ids and update it
        - create scene objects for each scene in pydantic specified format
        - save scene objects to self.book_json every iteration
        - write to target file one time after loop
        """
        for chapter_tpl in scenes:
            # get global book scene counter
            global_scene_counter = self.book_json.meta.total_scenes
            for scene_number, scene_content in enumerate(chapter_tpl[2], start=1):
                # scene attributes instruction & running_summary via defaults for now
                new_scene = Scene(
                    # scene_id equals global scene counter + scene number
                    scene_id=global_scene_counter + scene_number,
                    chapter_index=chapter_tpl[0],
                    # pass chapter_title if available at scene (book) as txt, otherwise None
                    chapter_title=chapter_tpl[1],
                    text=scene_content
                )
                self.book_json.scenes.append(new_scene)
            # update global scene counter with all scenes of a chapter
            self.book_json.meta.total_scenes += len(chapter_tpl[2])
        # update global word counter
        full_text = " ".join([scene.text for scene in self.book_json.scenes])
        self.book_json.meta.word_count = len(full_text.split())
        # use pydantic json model dump method to write obj into json
        with open(self.book_json_path, mode="w", encoding="utf-8") as f:
            json.dump(self.book_json.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    def _split_to_p_blocks(self) -> List[List[str]]:
        """
        - split up each chapter into into list of paragraph blocks splitted by \n\n:
        - each p_block must have min size -> prevent bloating llm with many p from (e.g. dialogues)
        - target format: - List[str] -> [['# Chapter 1', 'Some text..... ]
        """
        processed_chapters = []
        for chapter_text in self.chapters:
            # split text along \n\n and filter empty paragraphs within each chapter
            raw_paragraphs = [p.strip() for p in chapter_text.split("\n\n") if p.strip()]
            chapter_blocks = []
            bucket = ""
            bucket_counter = 0
            for p in raw_paragraphs:
                # calc size once
                p_tok = len(tokenizer.encode(p))
                # case 1: bucket is empty
                if not bucket:
                    # if atomic paragraph is greater than min size append it to p_blocks else bucket
                    if p_tok >= self.cfg.min_paragraph_size:
                        chapter_blocks.append(p)
                    else:
                        # in this case bucket is always empty, so add without \n\n added to p
                        bucket += p
                        bucket_counter += p_tok
                # case 2: content in bucket
                else:
                    # if p & bucket content are greater than threshold, empty bucket; else add to it
                    if (bucket_counter + p_tok) >= self.cfg.min_paragraph_size:
                        chapter_blocks.append(f"{bucket}\n\n{p}")
                        bucket = ""
                        bucket_counter = 0
                    else:
                        bucket += f"\n\n{p}"
                        bucket_counter += p_tok
            # flush bucket after loop if not empty
            if bucket:
                chapter_blocks.append(bucket)
            processed_chapters.append(chapter_blocks)
        return processed_chapters

    def _text_to_chapters(self) -> List[str]:
        """
        - split up book into chapters along these chapter anchors: # Chapter 1
        - use re (?=...) lookahead splits before the match
        - check amount parsed chapters against "total_chapters" in book metadata json
        """
        # ^ + multiline flag: match only valid at start of newline, but every newline due to flag
        chapters = re.split(r"(?=^# Chapter)", self.raw_text, flags=re.MULTILINE)
        # remove empty string at first pos from split before # Chapter 1
        chapters = [i for i in chapters if i.strip()]
        # check against meta chapter numbers
        assert self.book_json.meta.total_chapters == len(chapters), "check chapter processing"
        return chapters

    def run(self):
        """
        execute book processing & print stats:
        1. split narrative in form of 1 consecutive str into list of str along chapters
        2. split chapters into paragraphs along linebreaks (\n\n)
        3. create logical semantic scenes by combining chapter_paragraphs steered by llm calls
        4. build scene objects ready to save to target book json file
        """
        print("Starting process ...")
        self.chapters = self._text_to_chapters()
        print(f"Splitted chapters into list: {len(self.chapters)}")
        print("---------------------------------------------")
        chapter_paragraphs = self._split_to_p_blocks()
        print("Splitted chapters into paragraphs as additional level...")
        print(f"Amount paragraphs in 1st chapter after splitting: {len(chapter_paragraphs[0])}")
        print("---------------------------------------------")
        scenes = self._create_semantic_scenes(chapter_paragraphs)
        print(f"Received valid semantic scene responses by LLM for: {len(scenes)} Chapters")
        print("---------------------------------------------")
        print("Start processing scene objs and save to target JSON file ...")
        self._save_scenes(scenes)
        print(f"Did create {len(self.book_json.scenes)} semantic scenes in total.")
        print("---------------------------------------------")
        # print book wc from meta before op vs. wc of all scenes
        print(f"Original Word Count from book meta (before operation): {self.orig_wc}")
        print(f"Word Count all scenes combined (after): {self.book_json.meta.word_count}")
        print("---------------------------------------------")
        print(f"Book JSON file written to: {self.book_json_path}")
        print("-------Operation completed successfully.-------")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scene_creator.py <input_book.md> <output_book.json")
    else:
        bp = BookProcessor(sys.argv[1], sys.argv[2])
        bp.run()
