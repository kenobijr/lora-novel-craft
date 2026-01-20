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
  -> Context Total: ~1050 tokens
- Scene / Narrative: ~2000-3000 tokens (~1500 - 2300 words)
-> Share Narrative vs. Context: 2:1 to 3:1

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
  - ~~2000-3000 tokens (~1500 - 2300 words) per Scene
  - Intelligently parsed by some LLM & with deterministic python script
  - Inject book world context for better understanding
  - Chapter numbers / titles
- Add each Scene into scenes array of the respective book json

{
  "scene_id": 2,
  "chapter_index": 1,
  "chapter_title": "My Eagle",
  "text": "........."
  "recursive_summary": null,
},

- LLM model / SDK:
  - Gemini 2.0 Flash Lite / qwen/qwen-2.5-72b-instruct
  - OpenAI SDK
  - JSON Schema Enforcement enabled

##### Logic to parse scenes
**1. LLM splits into semantic scenes atomic unit**
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
      "final_token_sum": "P1(4) + P2(95) + P3(139) + P4(450) + P5(196) = 884",
      "end_paragraph": 5
    },
  ]
}
- token_math_log:
  - internal "final_token_sum" for llm token calculation

**2. Python script merges LLM semantic scenes deterministically**
- LLM output typically comes with entropy around token range, but good semantic breakpoints!
- Merge them together into ~2000-3000 token range scenes as final script output

**3. Insert special content as references / separate scenes**
- e.g. *The Iron Heel*: Bake in the Foreword written by Anthony Meredith (historian) in another time ~2600 AD (419 B.O.M.). and another Style: Academic, distant, analytical.
-  Keep it as Scene 00. It establishes the "truth" of the world (that the Iron Heel eventually falls), which creates dramatic irony.
- But create different systemmessage (vs. default):
  - Role: "Future Historian."
  - Instruction: "Write the academic foreword to the 'Everhard Manuscript,' analyzing its historical significance from the perspective of the 27th Century."


### Stage 6: Create recursive LLM summaries
- Recursive / Rolling Memory (narrative summary up to exactly current scene) per Scene
- Add them to the Jsons

### Stage 7: Define special tokens & prompts
- Add new TBD special tokens; use existing ones <think> / </think>
- Add suitable prompts <user>......</user> / <assistant> , .....

### Stage 8: Final training format JSONL with prompts & special tokens
- run a simple "Compiler Script" that flattens the JSONs + related content into efficient JSONL
- this format is native to HuggingFace datasets library and Industry standard for LLM fine-tuning

## Train -> Dataset Scale (DRAFT)

For 5-10 books, assuming 50,000 words per book, we have ~500,000 words (~750,000 tokens).

Chunk size: 4096 tokens.

VRAM Usage	~18-22 GB (Fits on 1x RTX 3090/4090)

Total Chunks: ~180 - 200.

This is a "Small Data" regime. To prevent overfitting (where the model memorizes the books verbatim), we must use:

High LoRA Rank (r=64 or 128): To allow sufficient capacity for stylistic adaptation.

Low Epochs (3-5): Monitoring validation loss strictly.

## Train -> Model Tuning / LoRA setup / Train strategy
Model Architecture (Confirmed from HuggingFace)

  Qwen3-30B-A3B-Thinking-2507:
  - Total params: 30.5B
  - Active params: 3.3B (MoE: 128 experts, 8 active per token)
  - Context: 256K native, extendable to 1M tokens
  - Special feature: Thinking mode (outputs <think> reasoning)

  ---
  QLoRA Compatibility Analysis

  1. Unsloth Support:
    - Officially supports Qwen3-30B-A3B QLoRA fine-tuning
    - Claims 17.5GB VRAM (optimized) to ~40GB (realistic for full sequences)
    - 2x faster training, 70% less VRAM vs. standard PEFT
    - Router layer disabled by default (correct approach for MoE)
  2. MS-SWIFT Support:
    - Full support for LoRA/QLoRA/DoRA on Qwen3-MoE
    - Production-ready framework
  3. MoE + Quantization:
    - AWS successfully fine-tuned Mixtral 8x7B MoE with QLoRA
    - MoEs work especially well with quantization - experts less affected by lower precision
    - bitsandbytes + PEFT fully supports MoE architectures

## Machine
  - RunPod: ~$0.30-0.50/hr for RTX 4090 (24GB VRAM) - sufficient for Mistral Nemo 12B QLoRA
  - Vast.ai: Similar pricing, more options
  - Lambda Labs: ~$1.10/hr for A100 40GB - overkill but very stable
  - Recommendation: RunPod or Vast.ai with RTX 4090 (24GB) for cost efficiency

## Evaluation
Perplexity alone won't capture creative quality - you'll need manual review





## After MVP
- Add Chain of Thought (CoT) into the context for reasoning models

## Example Book GH:
- Jack London - The Iron Heel
- Public Domain: https://www.gutenberg.org/ebooks/1164
- Used version: EPUB (no images, older E-readers)

## Appendix