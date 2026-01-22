# LoRA Novel Craft

- Finetune SOTA small-midsize Open Source LLMs with LoRA / QLoRA to become skilled Novel Authors of certain flavors.
- Finetuned Models should be able to create 100 pages cohesive novels
- Tech Stack: Python, Pytorch, Hugging Face Transformers, ...
- Core modules will be data-prep, train (= lora finetune), inference
- Narrative is split up into "Semantic Scenes" chunks
- World Context & Rolling Summaries are used to ensure narrative cohesion (Recursive Reprompting)

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

**Context Len / Token Math**
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

### Stage 1: Convert .epub to .md via Pandoc (CLI) & manual cleaning -> ./data/md_raw/
- Convert .epub inot .md with Pandoc to to clean XML / HTML tags but preserve semantic logic (italics / bold ...)
```pandoc input.epub -f epub -t gfm-smart --wrap=none -o output.md```
- Manual / Claude Code pre-cleaning & formatting:
  1. Manually delete noise: Title Page / End of Book / ... or add related logic to downstream md_cleaner.py script
  2. Standardize Chapter Headers as anchors: "# Chapter 1: My Eagle" or "# Chapter 1: My Eagle"
- Split book into separate .md files for certain reference data (e.g. character list, ....)
  1. Novel / Story / Narrative
  2. Reference data

### Stage 2: Clean .md with script -> ./data/md_clean/
- Process common anchors / css attritbutes / footnote patterns  into .md format
- Clean noise & normalise formatting
- save as cleaned .md
- [md_cleaner.py](./scripts/md_cleaner.py)

### Stage 3: Create Base .json files -> ./data/json_base/
- Create 1 json template file per book for 1x metadata and n scenes
- Data format:
{
  "meta": {
    "book_id": "iron_heel_london",
    "title": "The Iron Heel",
    "author": "Jack London",
    "word_count": 17673,
    "total_chapters": 12,
    "total_scenes": 0,
    "world_context": null
  },
  "scenes": []
}
- Use Claude Code agent; let it calc meta world_count & total_chapters with bash calls; fill the rest with "null"
- [base_json_creator_agent](./.claude/agents/agent-base_json_creator.md)

#### Stage 4: Create World Context for each book -> ./data/json_base/
- Create "World Context" / "World Rules" as constitution for each book
- Must not exceed 400 tokens -> 300 - 320 words
- Will be added into training data and to give LLMs context for splitting chapters into Semantic scenes and for creating recursive summarys
- Chosen model: **Gemini 3 Flash**:
  - 1M token context = can process a 500K word book in one single pass
  - Cheap
- Save response into json meta["world_context"]
- [world_context_creation_prompt](./prompts/world_context_creation.md)

#### Stage 5: Parse Book text into Semanctic Scenes -> ./data/json_scenes/
- Parse Chapters into base unit "Semantic Scenes" (smaller than chapters)
- Target: ~2000-3000 tokens (~1500 - 2300 words) per Scene
- Tokenizer: "Qwen/Qwen3-30B-A3B-Thinking-2507" (transformers)
##### Logic Scene creation (Chapter by Chapter):
  1. Split into paragraphs with sep: "\n\n"
  2. Merge to paragraph blocks of min size: 75 tokens
  3. Task LLM to merge paragraph blocks along semantic breakpoints (setting change, location)
    - Range: 400 - 1000 tokens per scene
    - "Goldilocks Zone": ~600-800 tokens
  4. Create final **Semantic Scenes** by merging LLM cut scenes to target range: 3000k tokens
##### LLM model / SDK:
  - qwen-2.5-72b-instruct / Gemini 2.0 Flash Lite / Gemini 2.5 Pro
  - OpenAI SDK
  - JSON Schema Enforcement enabled
##### Data Format Scene

{
  "meta": {
    "book_id": "iron_heel_london",
    "title": "The Iron Heel",
    "author": "Jack London",
    "word_count": 86487,
    "total_chapters": 25,
    "total_scenes": 54,
    "world_context": "# World Context\n\n## TONE & STYLE\n- Era/Genre: Early 20th-century"
  },
  "scenes": [
    {
      "scene_id": 2,
      "chapter_index": 1,
      "chapter_title": "My Eagle",
      "instruction": null,
      "text": "",
      "recursive_summary": null
    },
  ]
}

##### Execution
- Scene ID is counted up from book meta "total_scenes": 0"
- Meta "word_count" & "total_scenes" is updated by script after processing
- Insert special content as references / separate scenes manually
[scene_creator.py](./scripts/scene_creator.py)
[scene_splitting_prompt](./prompts/scene_splitting.md)


### Stage 6: Create Rolling Summaries for Semantic Scenes
- This approach, validated by research into "Infinite Context" agents, effectively compresses the "Long Term Memory" into a semantic vector (the summary text) that fits within the prompt
- Process:
  - Create Root Summary for scene 1
  - Take world context + Rolling summary current scene (n) + scene text current scene (n)
  - Add prompt & query llm to update the current rolling summary with this content best possible (within the token constraints) to produce the Rolling summary for the next scene (n + 1)
  - Output is saved at each scene object




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

