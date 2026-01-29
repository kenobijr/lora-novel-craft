# LoRA Novel Craft

- Finetune SOTA small-midsize Open Source LLMs with LoRA / QLoRA to become skilled Novel Authors of certain flavors.
- Finetuned Models should be able to create 100 pages cohesive novels
- Tech Stack: Python, Pytorch, Hugging Face Transformers, ...
- Core modules will be data-prep, train, inference
- Narrative is split up into "Semantic Scenes" chunks
- World Context & Rolling Summaries are used to ensure narrative cohesion (Recursive Reprompting)

---

## Live Model / Setcard

---

## Model / Tech Stack / Tools

### Base Model: Qwen3-30B-A3B-Thinking-2507 (TBD)
- Total params: 30.5B
- Active params: 3.3B (MoE: 128 experts, 8 active per token)
- Context: 256K native, extendable to 1M tokens

https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507

---

## Dataset Pipeline

**Context Len / Token Math**
- Total context length of training data: 4096 tokens (1.3 tokens per word) for efficient Finetuning
- Tokenizer used at data-prep: "Qwen/Qwen3-30B-A3B-Thinking-2507" (transformers)
- Context Injection (prepended before Scene / Narrative):
  - System prompt: ~150 tokens
  - World Context: ~400 tokens
  - Recursive Summary: ~400 tokens
  - Instruction / Prompt: ~50 tokens
  - Buffer diverse: ~50 tokens
  -> Context Total: ~1050 tokens
- Scene / Narrative: ~2000-3000 tokens (~1500 - 2300 words)

-> Share Narrative vs. Context: 2:1 to 3:1

### Stage 1: Convert .epub to .md via Pandoc (CLI) & manual cleaning -> ./data/md/raw/
- Convert .epub to .md with Pandoc to to clean XML / HTML tags but preserve semantic logic (italics / bold ...)
```pandoc input.epub -f epub -t gfm-smart --wrap=none -o output.md```
- Manual / Claude Code pre-cleaning & formatting:
  1. Manually delete noise: Title Page / End of novel / ... or add related logic to downstream md_cleaner.py script
  2. Standardize Chapter Headers as anchors: "# Chapter 1: My Eagle" or "# Chapter 1: My Eagle"

### Stage 2: Clean .md with script & split Narrative from Reference content -> ./data/md/final/
- Process common anchors / css attritbutes / footnote patterns  into .md format
- Clean noise & normalise formatting
- Split book into separate .md files for certain reference data (e.g. character list, ....)
  1. Novel / Story / Narrative (./data/md/final/text/)
  2. Reference data (./data/md/final/ref/)

### Stage 3: Create Base .json files -> ./data/json/base/
- Create 1 json base file per novel with Claude Code agent
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

#### Stage 4: Create World Context for each novel -> ./data/json/base/
- Create "World Context" / "World Rules" as constitution for each novel
- Must not exceed 400 tokens -> 300 - 320 words

#### Stage 5: Split Narrative into Semanctic Scenes -> ./data/json/scenes/
- Parse Chapters into base unit "Semantic Scenes" (smaller than chapters)
- Target: ~2000-3000 tokens (~1500 - 2300 words) per Scene
##### 5.1 Process:
  1. Process Chapter by Chapter of input book json (deterministic)
  2. Split into paragraphs with sep: "\n\n" (deterministic)
  3. Merge to paragraph blocks of min size: 75 tokens (deterministic)
  4. LLM merges paragraph blocks into *Atomic Semantic Scenes* (AI)
    - Along semantic breakpoints (setting change, location)
    - LLM receives context in form of "World Context" & detailed instructions
    - Target Range: 400 - 1000 tokens per atomic scene
    - **Goldilocks Zone: ~600-800 tokens**
  5. Merge LLM cut atomic scenese into final **Semantic Scenes**: (deterministic)
    - Target range: 3000k tokens hard max
    - Greedy Merge
##### 5.2 JSON Data Schema:
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
      "running_summary": null
    },
  ]
}
##### 5.3 Handling of Reference Content
- Relevant special content like Vocab / Foreword / ... is split into scenes manually
- Such scenes are prepended to the Narrative Semantic Scenes and flagged

#### Stage 6: Create Semantic Scenes specific Rolling Summaries
- Compress states of narrative into Rolling Summaries (like LSTM but in natural language) along Semantic Scenes as timesteps
- Each Semantic Scene's Rolling Summary attribute contains the compressed Narrative: **what happened so far up to this specific Semantic Scene?**
- Running Summary must not be greater than 400 tokens
##### 6.1 Structure / Content
Each Rolling Summary is clustered into 2 logically separate categories / 8 attributes with separate token / word restraints:
- **LOCAL MOMENTUM**| Last scene / Labels / Mood / Suspense | max 60 words combined:
  - scene_end_state: | MAX 25 words
  - emotional_beat: | MAX 15 words
  - immediate_tension: | MAX 20 words
- **GLOBAL STATE**| Whole story / What happened up to now? Threads? | max 140 words combined
  - global_events: | MAX 60 words
  - unresolved_threads: | MAX 35 words
  - world_state: | MAX 20 words
  - active_characters: | MAX 15 words
  - global_shift: | MAX 20 words
##### 6.2 Process
- Create Root Summary for scene 1 with empty "story begins" values
- Take world context + running summary current scene (n) + text current scene (n) to construct prompt
- Query LLM with JSON response enforcement schema
- If LLM "create summary" response too long:
  - Execute follow-up LLM compress calls on the previous response content
  - Try up to 3 times using same input
- If all compress calls fail to deliver response under token threshold, take response of last compress call
- Take this new gen running summary to construct prompt to create running summary for next scene and so on
##### 6.3 Prompt Setup
- Systemmessage
- Input / Content description
- World Context (Fixed static file, as used before)
- Running Summary
  - NOVEL PROGRESS: [Progress: XX%] (=Current Scene / Total amount scenes %)
  - LOCAL MOMENTUM: -> check llm response template
  - WORLD STATE: -> check llm response template
- Text content current scene
- Instruction
##### 6.4 Handling of Reference Content Scenes
- Construct special root running summary for reference scenes
- Construct special prompt instruction to handle reference scenes


### Stage 7: Define special tokens & prompts
- Add new TBD special tokens; use existing ones <think> / </think>
- Add suitable prompts <user>......</user> / <assistant> , .....

### Stage 8: Final training format JSONL with prompts & special tokens
- run a simple "Compiler Script" that flattens the JSONs + related content into efficient JSONL
- this format is native to HuggingFace datasets library and Industry standard for LLM fine-tuning

## Train -> Dataset Scale (LLM-DRAFT)

- For 5-10 books, assuming 50,000 words per book, we have ~500,000 words (~750,000 tokens).
- Chunk size: 4096 tokens.
- VRAM Usage	~18-22 GB (Fits on 1x RTX 3090/4090)
- Total Chunks: ~180 - 200.
- This is a "Small Data" regime. To prevent overfitting (where the model memorizes the books verbatim), we must use:
- High LoRA Rank (r=64 or 128): To allow sufficient capacity for stylistic adaptation.
- Low Epochs (3-5): Monitoring validation loss strictly.

## Train -> Model Tuning / LoRA setup / Train strategy (LLM-DRAFT)
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


## Inference / Novel creation pipeline
1. Planning / Global / Hierarchical Outline
- Models first generate a multi-level outline (e.g., acts > chapters > scenes), then fill iteratively. This mimics human writing, improving global coherence 
- generate a list of 20-50 semantic scene "stubs" or one-sentence goals
- Include "Reasoning about Turning Points": Let llm not only "outline some story", but "include certain turning points": setback, climax, ...
2. create / define metadata to create world context / root rolling summary / root novel / base story / amount semantic scenes / chapters
3. python script loops the model through creating defined (or range defined) amount of scenes / chapters, each with
  - systemmessage to model, all this aforementioned metadata, and then instruction like "write semantic scene / chapter 1 / 10"
  - after each new gen semantic scene, the rolling summary is updated with the additional scene
  - so in next loop call for next semantic scene, the model with have the updated summary


## Evaluation
Perplexity alone won't capture creative quality - you'll need manual review


## Sources
- Are Large Language Models Capable of Generating Human-Level Narratives? (https://arxiv.org/pdf/2407.13248)
- Plan-and-Write: Towards Better Automatic Storytelling (https://www.researchgate.netpublication335380574_Plan-and-Write_Towards_Better_Automatic_Storytelling)
- RecurrentGPT: Interactive Generation of (Arbitrarily) Long Text (https://arxiv.org/abs/2305.13304)




## After MVP
- Add Chain of Thought (CoT) into the context for reasoning models

## Example Book GH:
- Jack London - The Iron Heel
- Public Domain: https://www.gutenberg.org/ebooks/1164
- Used version: EPUB (no images, older E-readers)

