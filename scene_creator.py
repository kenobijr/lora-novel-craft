"""
Split up book in .md format into "Semantic Scenes" and add them to target book json
  - read in book -> split up into chapters -> loop through & process each chapter
  - preprocess each chapter by splitting it by \n\n into paragraphs and number each one
  - add amount tokens of each paragraph to obj with tiktokenizer
  - take llm response to create scenes with combined paragraphs
  - add scenes to target book json
Processing:
    1. map book md str into list of chapters along anchors: # Chapter 1, ...
        str -> ['# Chapter 1\n\nMr. Jon.... , '# Chapter 2 ..]
    2. map chapters into list of paragraphs splitted by \n\n:
        List[str] -> [['# Chapter 1', 'Mr. Jones..... ]
    3. map list of paragraphs to list of semantic scenes splited along smart llm defined boundaries:
        List[List[paragraphs]] -> List[List[scenes]]


"""
import sys
import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from transformers import AutoTokenizer
from typing import List


class SceneBoundary(BaseModel):
    """ single scene boundary marker """
    token_math_log: str
    end_paragraph: int


class ScenePartitionResponse(BaseModel):
    """ llm response schema for scene partitioning """
    scenes: List[SceneBoundary]


# api key
load_dotenv()
api_key = os.getenv("OPEN_ROUTER_KEY")
if not api_key:
    raise ValueError("could not load API key...")

# init tokenizer
tokenizer = AutoTokenizer.from_pretrained(
      "Qwen/Qwen3-30B-A3B-Thinking-2507"
)

# init system_message
with open("./prompts/scene_splitting.md", mode="r", encoding="utf-8") as f:
    data = f.read()
system_message = data

# init llm
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)


def get_scene_partition(p_block: str, context: str):
    """
    Input:
    - chapter str ready processed for llm
    - world context as str already read out from .md file
    Actions:
    - read out

    """

    full_prompt = f"""
<system>
{system_message}
</system>

<world_context>
{context}
</world_context>

<text_paragraphs>
{p_block}
</text_paragraphs>
"""

    response = client.chat.completions.create(
      model="google/gemini-2.0-flash-lite-001",
      messages=[{"role": "user", "content": full_prompt}],
      temperature=0.1,
      response_format={
          "type": "json_schema",
          "json_schema": {
              "name": "scene_partition",
              "strict": True,
              "schema": ScenePartitionResponse.model_json_schema()
          }
      }
    )

    # parse llm response to str
    result = json.loads(response.choices[0].message.content)
    # print(json.dumps(result, indent=2))
    
    # print safety copy of llm json response
    # with open("./data/json/test_scene_partition.json", mode="w", encoding="utf-8") as f:
    #     json.dump(result, f, indent=2)
    
    # # test logic against safety copy
    # with open("./data/json/test_scene_partition_old.json", mode="r", encoding="utf-8") as f:
    #     result = json.load(f)

    # check llm response form & values
    if not result:
        raise ValueError(f"No api result for prompt: {full_prompt}")
    assert result["scenes"][0]["end_paragraph"] > 0, "first scene end pnot valid"
    assert isinstance(result["scenes"][-1]["end_paragraph"], int), "last scene end p not valid"
    return result["scenes"]


def process_chapters(chapters: List[str], context: str, book_json: str):
    # split up each chapter into paragraph divided by \n\n
    chapters = [chapter.split("\n\n") for chapter in chapters]
    # filter empty paragraphs within each chapter
    chapters = [[p for p in chapter if p.strip()] for chapter in chapters]
    print("Splitted chapters into paragraphs...")
    print(f"Amount paragraphs in 1st chapter after splitting: {len(chapters[0])}")
    # print("Sample 1st chapter paragraphs:")
    # print(chapters[0][:3])
    print("---------------------------------------------")

    # process chapter by chapter:
    # 1. extract chapter metadata (idx, title) at start for saving to json at end of loop iteration
    # 2. bring each chapter into llm format with numbered p + token amount
    # 3. send chapter paragraphs to llm -> get boundaries back for semanctic scenes
    # 4. split chapter paragraphs into list of semantic scenes defined by llm boundaries
    # 5. TBD: glue paragraphs together into semantic scenes -> List[List[str]]
    for chapter in chapters:
        # extract chapter metadata: chapter index & chapter title (optional)
        match = re.match(r'^#\s*Chapter\s+(\d+)(?::\s*(.+))?', chapter[0])
        chapter_index = int(match.group(1))
        chapter_title = match.group(2)
        print(f"Started processing chapter: {chapter_index}")

        # bring each chapter into target text format with numbered p + token amount
        lines = []
        for scene_number, p in enumerate(chapter, start=1):
            tok_p = len(tokenizer.encode(p))
            lines.append(f"[P:{scene_number}|Tok:{tok_p}] {p}")
        p_block = "\n".join(lines)

        # print("Sample 1st chapter formatted numbered paragraphs with token amount:")
        # print(p_block[:500])
        # print("---------------------------------------------")

        print(f"Prompted LLM for chapter: {chapter_index}")
        # partitioning: list of dicts; entry for each scene with end_paragraph
        partitions = get_scene_partition(p_block, context)

        # amount of paragraphs in chapter must equal last scene end_paragraph
        assert len(chapter) == partitions[-1]["end_paragraph"], \
            f"wrong end_paragraph for chapter: {chapter[0]}"
        
        # combine paragraphs to scenes with partitioning dicts
        prev_end = 0
        scenes = []
        for scene_content in partitions:
            end = scene_content["end_paragraph"]
            scenes.append(chapter[prev_end:end])  # slice from prev to current
            prev_end = end
        print(f"Received response by LLM with amount of scenes: {len(scenes)}")

        # open target json file
        with open(book_json, mode="r", encoding="utf-8") as f:
            base_json = json.load(f)
        
        # per scene, add such an object; get scene id from json meta
        # {
        # "scene_id": null,
        # "chapter_index": null,
        # "chapter_title": null,
        # "text": null,
        # "recursive_summary": null,
        # "thought_plan": null
        # }

        # get global book scene counter
        global_scene_counter = base_json["meta"]["total_scenes"]
        
        # add scenes to json
        for scene_number, scene_content in enumerate(scenes, start=1):
            
            scene_obj = {}
            
            # scene_id equals sequence num within chapter + running counter of prev book scenes
            scene_obj["scene_id"] = global_scene_counter + scene_number
            # add previous extracted chapter idx & title if available
            scene_obj["chapter_index"] = chapter_index
            scene_obj["chapter_title"] = chapter_title if chapter_title else None
            # add scene text -> paragraphs joined together again with \n\n
            scene_obj["text"] = "\n\n".join(scene_content)
            # add empty fields for summary & thought plan per scene; will be filled later in process
            scene_obj["recursive_summary"] = None
            scene_obj["thought_plan"] = None
        
            # add scene obj to book & write to json
            base_json["scenes"].append(scene_obj)
        # update global scene counter
        base_json["meta"]["total_scenes"] += len(scenes)

        with open(book_json, mode="w", encoding="utf-8") as f:
            json.dump(base_json, f, indent=2, ensure_ascii=False)
        
        print(f"Did write {len(scenes)} scenes for Chapter {chapter_index} to {book_json}")
        print("---------------------------------------------")


def steer_mapping(book_md: str, world_context_md: str, book_json: str):
    print("Starting process")
    # read in world_context
    with open(world_context_md, mode="r", encoding="utf-8") as f:
        context = f.read()
    # read in .md
    with open(book_md, mode="r", encoding="utf-8") as f:
        content = f.read()
    # split up book into chapters along these chapter anchors: # Chapter 1
    # (?=...) lookahead splits before the match
    # ^ + multiline flag: match only valid at start of newline, but every newline due to flag
    chapters = re.split(r"(?=^# Chapter)", content, flags=re.MULTILINE)
    # remove empty string at first pos from split before # Chapter 1
    chapters = [i for i in chapters if i.strip()]
    # read in target book json; check meta chapter numbers against processed number
    with open(book_json, mode="r", encoding="utf-8") as f:
        json_content = json.load(f)
    assert json_content["meta"]["total_chapters"] == len(chapters), "check chapter processing"

    print(f"Splitted chapters into list: {len(chapters)}")
    print("---------------------------------------------")
    process_chapters(chapters, context, book_json)
    print(f"-------Operation completed successfully for {len(chapters)} chapters.-------")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python book_to_scenes.py <book.md> <world_context.md> <output_file.json")
    else:
        steer_mapping(sys.argv[1], sys.argv[2], sys.argv[3])
