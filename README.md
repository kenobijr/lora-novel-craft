# LoRA Novel Craft

Finetune SOTA small-midsize Open Source LLMs with LoRA / QLoRA to become skilled Novel Authors of certain flavors.

---

## Live Model / Setcard

---

## Model / Tech Stack / Tools

### Base Model: Qwen3-30B-A3B-Thinking-2507
- Total params: 30.5B
- Active params: 3.3B (MoE: 128 experts, 8 active per token)
- Context: 256K native, extendable to 1M tokens
- Special feature: Thinking mode (outputs <think> reasoning)

https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507

---

## Dataset Pipeline

### Context Len / Token Math
- Total context length of training data: 4096 tokens (1.3 tokens per word) for efficient Finetuning
    - Must not be exceeded during train due to fixed setup
- Context Injection (prepended before Scene / Narrative):
  - System prompt: ~150 tokens
  - World Context: ~400 tokens
  - Recursive Summary: ~400 tokens
  - Instruction / Prompt: ~50 tokens
  - Buffer diverse: ~50 tokens
  - Chain of Thought (CoT) : ~1500 tokens
- Scene / Narrative: ~650-1500 tokens (~500 - 1150 words)

### Stage 1: Convert .epub to .md via Pandoc (CLI) & manual base cleaning
- Convert .epub inot .md with Pandoc to to clean XML / HTML tags but preserve semantic logic (italics / bold ...)
```pandoc input.epub -f epub -t gfm-smart --wrap=none -o output.md```
- Manual / Claude Code pre-cleaning & formatting:
  1. Manually delete noise: Title Page / End of Book / ...
  2. Standardize Chapter Headers: # Chapter 1: My Eagle
  3. Look for noise in content, e.g. "{#9780063068452_Chapter_21.xhtml_page_136 .right_1 .pagebreak title="136"}" and adapt md_cleaner script regardingly
- Split content into 2 separate .md files, if related metadata available in book:
  1. Novel / Story / Narrative
  2. "World Rules" meta content / context
- **Novel Target Format**:
  - Content Anchors: "# Chapter 1" (only arab numbers)
  - If section title exists: "# Chapter 1: My Eagle"

### Stage 2: Clean .md with script
- Process common anchors / css attritbutes / footnote patterns  into .md format
- Clean noise & normalise formatting
- save as cleaned .md
- LINK TO SCRIPT [...]

### Stage 3: Create Base .json files
- Create 1 json template file per book for 1x metadata and n scenes
- Template:
{
  "meta": {
    "book_id": "iron_heel_london",
    "title": "The Iron Heel",
    "author": "Jack London",
    "word_count": 17673,
    "total_chapters": 12,
    "total_scenes": 36,
    "world_context": "..."
  },
  "scenes": [
    {
      "scene_id": 1,
      "chapter_index": 1,
      "chapter_title": null,
      "instruction": null,
      "text": "some text content....",
      "recursive_summary": null, 
      "thought_plan": null
    }
  ]
}

- Use Claude Code agent; let it calc meta world_count & total_chapters with bash calls; fill the rest with "null"
- LINK TO Agent [...]

#### Stage 4: Create World Context for each book
- Create "World Context" / "World Rules" as constitution for each book
- Must not exceed 400 tokens -> 300 - 320 words
- Will be used downstream to split up chapters into scenes intelligently and for recursive summarys
- Chosen model: **Gemini 3 Flash**:
  - 1M token context = can process a 500K word book in one single pass
  - Cheap
  - SOTA performace by Gemini model family
- Save it into json meta.worldcontext attribute
- LINK to systemmessage: [...]

#### Stage 5: Parse Chapters into Semanctic Scenes
- Parse Chapters into base unit "Semantic Scenes":
  - ~650-1500 tokens (~500 - 1150 words) per Scene
  - Intelligently parsed by some LLM
  - Inject book world context for better understanding
  - Chapter numbers / titles
- Add each Scene into scenes array of the respective book json

- scene data model:
  - "scene_id" (Unique identifier 1-n)
  - "instruction": 
    - null for all "normal" narrative cases, which will draw the same instruction "create scene..." together with same systemmessage from other file
    - for special cases / references (forword / glossary) we will need a special instruction, saved at the scene
    - script must check: if instruction null -> take default; else -> take instruction saved at scene

- logic to parse scenes:
  - read in book -> split up into chapters -> loop through & process each chapter
  - preprocess each chapter by splitting it by \n\n into paragraphs and number each one
  - add amount tokens of each paragraph to obj with tiktokenizer
  - chapter text block format for llm in plain text with inline paragraph metadata:
[P:1|Tok:23] The morning sun cast long shadows across the courtyard as the
workers began to gather.
[P:2|Tok:4] I moved to stand beside him...
  - Send to LLM with world context + instruction: "group into scenes of 650-1500 tokens"
  - ADD LINK TO PROMPT [...]
  - llm response must be strictly in this json format with json enforcement enabled:
{
  "scenes": [
    {
      "token_math_log": "P1(4) + P2(95) + P3(139) + P4(450) + P5(196) = 884 tokens. Valid range (650-1500).",
      "end_paragraph": 5
    },
  ]
}
- token_math_log:
  - internal "Scratchpad" for llm token calculation
  - without llms failed to calc the tokens
  - probably due to being forced strictly into json output structure enforcement

- LLM model / SDK:
  - Gemini 2.0 Flash Lite
  - OpenAI SDK
  - JSON Schema Enforcement enabled





### Stage 6: Create recursive LLM summaries
- Recursive / Rolling Memory (narrative summary up to exactly current scene) per Scene
- Add them to the Jsons
