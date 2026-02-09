## TASK RULES - REFERENCE MATERIAL EXTRACTION

### CORE PRINCIPLES
1. **Knowledge Extraction**: You are processing REFERENCE material (glossary, character roster, foreword, appendix) that precedes or frames the main narrative. Extract and codify facts, not plot.
2. **Foundation Building**: Capture key terminology, character introductions, world rules, or historical framing that will inform the upcoming story.
3. **No Plot Summary**: This is NOT narrative content — do not describe "what happened" but rather "what is established/defined."
4. **Terminology Accuracy**: Use proper nouns, vocabulary terms, and names exactly as defined in the reference material.

### KNOWLEDGE RESTRICTION
- You MUST derive ALL information ONLY from the provided content in this prompt. Do NOT use any external knowledge about this book, its characters, or world beyond this content.
- **WARNING**: This book may be present in your training data. Extract ONLY what is explicitly stated in this reference section.

### Field 1-3: LOCAL STATE (45-60 words combined)
- Derived ONLY from this reference section — ignore previous state entirely:
- scene_end_state: What type of reference material was just established (vocabulary, characters, historical frame, rules). Be specific and concrete | 15-25 words
- emotional_beat: The tone or atmosphere this reference material establishes for the narrative | 8-15 words
- immediate_tension: Any foreshadowing, warnings, or tensions hinted at in this reference material | 11-20 words

### Field 4-8: ACCUMULATED KNOWLEDGE (135-145 words combined)
- Merge previous reference knowledge WITH new reference content:
- global_events: "[REFERENCE PHASE]" prefix + Key facts, rules, or definitions extracted from ALL reference sections so far. Be exhaustive | 40-60 words
- unresolved_threads: World rules, mysteries, or questions established but not yet explored in narrative | 1-5 items, 22-35 words
- world_state: The foundational setting, vocabulary, or rules now established for the narrative | 11-20 words
- active_characters: "[NONE] - Reference phase; characters not yet active in narrative." | 8-15 words
- global_shift: What this implies for the reader (e.g., "Language barrier established," "Historical irony enabled") | 11-20 words

**WORD BUDGET: 180-200 words MANDATORY. Below 180 = INVALID. Above 200 = INVALID. Word count = JSON values only, excluding keys and formatting. USE the full budget — every word under 180 is lost narrative context that degrades downstream scene generation.**

## OUTPUT FORMAT
Return a valid JSON object matching this structure:

{
  "scene_end_state": "<What reference material was established in this section>",
  "emotional_beat": "<Tone/atmosphere this reference establishes>",
  "immediate_tension": "<Any foreshadowing or tensions hinted at>",
  "global_events": "[REFERENCE PHASE] <Key facts, rules, vocabulary extracted>",
  "unresolved_threads": "<World rules or mysteries established>",
  "world_state": "<Foundational setting/vocabulary/rules now defined>",
  "active_characters": "[NONE] - Reference phase; characters not yet active in narrative.",
  "global_shift": "<What this implies for the reader/narrative>"
}

## VALIDATION GATE — OUTPUT BLOCKED UNTIL ALL PASS

Before outputting JSON, verify:

□ All 8 fields present with snake_case keys
□ Fields 1-3 reflect ONLY this reference section — no carryover from previous
□ Fields 4-8 reflect ACCUMULATED knowledge from all reference sections processed so far
□ global_events starts with "[REFERENCE PHASE]" prefix
□ Each field within its word range (check BOTH min AND max)
□ Total output: 180-200 words (HARD MIN: 180, HARD MAX: 200)
□ Valid JSON with snake_case keys

**ONE FAILURE = REJECT OUTPUT. Below 180 or above 200 = INVALID. Go back and fix.**
