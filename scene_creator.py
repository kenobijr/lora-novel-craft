"""
Target is to split up a whole novel into logical "Semantic Scene" chunks (smaller than chapters) to
improve the dataset quality vs. splitting into fixed / repetitive word / token boundaries.
A judge / teacher LLM is tasked to split each book chapter into these Semantic Scenes which must be
within a defined token / word amount range.
To enable the LLM to "count" words / tokens, book chapters are splitted into smaller paragraphs.
These paragraphs are numbered consecutively and their respective token amount is counted, too.
With these information the LLM returns boundaries and paragraphs are merged into Semantic Scenes.

Process:
    1. map book md str into list of chapters along anchors: # Chapter 1, ...
        str -> ['# Chapter 1\n\nMr. Jon.... , '# Chapter 2 ..]
    2. map chapters into list of paragraphs splitted by \n\n:
        List[str] -> [['# Chapter 1', 'Mr. Jones..... ]
    3. map list of paragraphs to list of semantic scenes splited along smart llm defined boundaries:
        List[List[paragraphs]] -> List[List[scenes]]

          - add scenes to target book json

"""
import sys
import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from transformers import AutoTokenizer
from typing import List, Tuple, Dict

# llm model = openrouter id
LLM = "qwen/qwen-2.5-72b-instruct"
# "google/gemini-2.0-flash-lite-001"

SYSTEM_MESSAGE = "./prompts/scene_splitting.md"

# llm debugging
DEBUG_MODE = True
DEBUG_DIR = "./data/debug/"

# api key
load_dotenv()
api_key = os.getenv("OPEN_ROUTER_KEY")
if not api_key:
    raise ValueError("could not load API key...")


# pydantic classes for llm response format enforcement
class SceneBoundary(BaseModel):
    """ single scene boundary marker """
    final_token_sum: str
    end_paragraph: int


class ScenePartitionResponse(BaseModel):
    """ llm response schema for scene partitioning """
    scenes: List[SceneBoundary]


class SceneSplitterLLM:
    """
    - handles llm related logic: model / api / key / connections / ...
    - world_context directly delivered as str; book metadata needed for llm call
    - manages & formats prompts / systemmessages
    """
    def __init__(self, world_context: str, book_json: str):
        self.wc = world_context
        # init system_message
        with open(SYSTEM_MESSAGE, mode="r", encoding="utf-8") as f:
            self.sm = f.read()
        # save cleaned book title for debugging
        self.title = os.path.basename(book_json).removesuffix(".json")
        # load qwen 3 tokenizer from transformers
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Thinking-2507")
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
        - switch on / off with global
        - if activated, on every llm call prompt & response are saved into debug folder
        """
        os.makedirs(DEBUG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%H%M%S_%f")
        prompt_path = os.path.join(DEBUG_DIR, f"debug_prompt.{self.title}.{ts}.md")
        response_path = os.path.join(DEBUG_DIR, f"debug_llm.{self.title}.{ts}.json")
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
                    "schema": ScenePartitionResponse.model_json_schema()
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
        # with open("./data/json/test_scene_partition_old.json", mode="r", encoding="utf-8") as f:
        #     result = json.load(f)

        # base validation
        if not result:
            raise ValueError(f"No api result for prompt: {full_prompt}")

        # if debug mode activated prompt & llm response will be saved
        if DEBUG_MODE:
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
            print(f"WARN: LLM missed p: Ended at {last_boundary}vs.{amount_p}). Auto-correcting...")
            # force extend the very last scene to include the missing paragraphs
            result["scenes"][-1]["end_paragraph"] = amount_p
            print(f"correction completed; difference of {diff} corrected.")

        # return validated list
        return result["scenes"]


class BookProcessor:
    """
    - steering the whole book to scene splitting process
    - process all arguments
    - pass world_context further & init scene splitting llm class with it
    - save scene objects to self.book_json during processing
    """
    def __init__(self, input_book_md: str, output_book_json: str):
        # parse & save content from book md file: narrative as one formatted str
        with open(input_book_md, mode="r", encoding="utf-8") as f:
            self.raw_text = f.read()
        # file path target json file
        self.book_json_path = output_book_json
        # parse & save content from book json file: structured book meta data
        with open(output_book_json, mode="r", encoding="utf-8") as f:
            self.book_json = json.load(f)
        # init llm -> pass world context str extracted from book json
        self.llm = SceneSplitterLLM(self.book_json["meta"]["world_context"], output_book_json)
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
            tok_p = len(self.llm.tokenizer.encode(p))
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
                len(self.llm.tokenizer.encode(scene)) for scene in atomic_scenes
            ) / len(atomic_scenes)
            print(f"Token avg per Atomic Scene before processing: {avg_atomic_scene:,.2f}")
            # merge atomic scenes into bigger target semantic scenes up to specified max size
            max_range = 3000
            semantic_scenes = []
            token_counter = 0
            running_scene = ""
            for scene in atomic_scenes:
                tok_current = len(self.llm.tokenizer.encode(scene))
                # if running scene, together with current scene, under threshold -> add up
                if token_counter + tok_current <= max_range:
                    running_scene += scene
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
                len(self.llm.tokenizer.encode(scene)) for scene in semantic_scenes
            ) / len(semantic_scenes)
            print(f"Token avg per Semantic Scene after processing: {avg_semantic_scene:,.2f}")
            # add all scenes to list with chapter metadata
            all_scenes.append((chapter_idx, chapter_title, semantic_scenes))

        return all_scenes

    def _save_scenes(self, scenes: List[Tuple[str, str, List]]) -> List:
        """
        - loop through chapter tuples & create consecutive scenes with text & chapter meta data
        - use total scenes from book json metadata to create scene ids and update it
        create scene objects for each scene in following format:
        {
        "scene_id": 1,
        "chapter_index": 1,
        "chapter_title": null,
        "instruction": null,
        "text": "some text content....",
        "recursive_summary": null,
        "thought_plan": null
        }
        - save scene objects to self.book_json every iteration
        - write to target file one time after loop
        """
        for chapter_tpl in scenes:

            # get global book scene counter
            global_scene_counter = self.book_json["meta"]["total_scenes"]
            for scene_number, scene_content in enumerate(chapter_tpl[2], start=1):
                scene_obj = {}
                # scene_id equals sequence num within chapter + running counter of prev book scenes
                scene_obj["scene_id"] = global_scene_counter + scene_number
                # add chapter idx & title if available
                scene_obj["chapter_index"] = chapter_tpl[0]
                scene_obj["chapter_title"] = chapter_tpl[1] if chapter_tpl[1] else None
                # add semantic scene text
                scene_obj["text"] = scene_content
                # add empty fields for summary & thought plan per scene for later
                scene_obj["recursive_summary"] = None
                scene_obj["thought_plan"] = None
                # add scene obj to book & write to json
                self.book_json["scenes"].append(scene_obj)
            # update global scene counter with all scenes of a chapter
            self.book_json["meta"]["total_scenes"] += len(chapter_tpl[2])
        # write to json
        with open(self.book_json_path, mode="w", encoding="utf-8") as f:
            json.dump(self.book_json, f, indent=2, ensure_ascii=False)

    def _split_to_paragraphs(self) -> List[List[str]]:
        """
        - split up each chapter into into list of paragraphs splitted by \n\n:
        - List[str] -> [['# Chapter 1', 'Some text..... ]
        """
        chapter_paragraphs = [chapter.split("\n\n") for chapter in self.chapters]
        # filter empty paragraphs within each chapter
        chapter_paragraphs = [[p for p in chapter if p.strip()] for chapter in chapter_paragraphs]
        return chapter_paragraphs

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
        assert self.book_json["meta"]["total_chapters"] == len(chapters), "check chapter processing"
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
        chapter_paragraphs = self._split_to_paragraphs()
        print("Splitted chapters into paragraphs as additional level...")
        print(f"Amount paragraphs in 1st chapter after splitting: {len(chapter_paragraphs[0])}")
        print("---------------------------------------------")
        scenes = self._create_semantic_scenes(chapter_paragraphs)
        print(f"Received valid semantic scene responses by LLM for: {len(scenes)} Chapters")
        print("---------------------------------------------")
        print("Start processing scene objs and save to target JSON file ...")
        self._save_scenes(scenes)
        print(f"Did create {len(self.book_json['scenes'])} semantic scenes in total.") 
        print(f"Book JSON file written to: {self.book_json_path}")
        print("-------Operation completed successfully.-------")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python book_to_scenes.py <input_book.md> <output_book.json")
    else:
        bp = BookProcessor(sys.argv[1], sys.argv[2])
        bp.run()
