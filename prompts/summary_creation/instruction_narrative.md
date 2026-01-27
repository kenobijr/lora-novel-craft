## TASK RULES

### CORE PRINCIPLES
1. **Causality Preservation**: If a character is injured, acquires an item, learns a secret, or forms/breaks an alliance — this MUST be preserved
2. **Compression Over Time**: Old events (many scenes ago) compress to 1-sentence anchors. Recent events get more detail
3. **Terminology Consistency**: Use proper nouns and terms exactly as defined in the world context
4. **Omniscient Voice**: Write as neutral narrator. Never "In this scene..." or "The author shows..." — just state what happened

### KNOWLEDGE RESTRICTION                                       
- You MUST derive ALL information ONLY from the provided content in this prompt. Do NOT use any external knowledge about this book, its characters, or plot events beyond this content. Treat each scene as if you are reading this story for the first time.
- **WARNING**: This story may be present in your training data. You are strictly forbidden from anticipating plot points (e.g., character deaths, name changes, betrayals) until they explicitly appear in the provided text snippets.

### Field 1-3: LOCAL MOMENTUM RULES (max 60 words combined)
- Derived ONLY from the new scene text — ignore previous momentum entirely:
- scene_end_state: Physical situation at exact closing moment — location, time, positions | MAX 25 words
- emotional_beat: Dominant feeling as scene closes — the emotional residue | MAX 15 words
- immediate_tension: Unresolved micro-conflict carrying forward — the narrative hook | MAX 20 words

### Field 4-8: GLOBAL STATE RULES (max 140 words combined)
- Merge previous global state WITH new scene events:
- global_events: Compressed history of entire story with new scene integrated — old events compress more aggressively | MAX 60 words
- unresolved_threads: Active plot threads — add new, remove resolved, preserve ongoing | 1-5 items, MAX 35 words
- world_state: Current world situation after this scene — location, stakes, where things stand | MAX 20 words
- active_characters: Characters currently relevant to ongoing narrative (not just this scene) | MAX 15 words
- global_shift: What changed because of this scene — new knowledge, relationships, dangers | MAX 20 words

**TOTAL OUTPUT: 180-200 words MAXIMUM. Exceeding 200 words = INVALID; Word count means JSON values only, excluding keys and formatting**

## OUTPUT FORMAT
Return a valid JSON object matching this structure:

{
  "scene_end_state": "<Physical/temporal situation as scene closes>",
  "emotional_beat": "<Dominant feeling at the end of the scene>",
  "immediate_tension": "<Unresolved micro-conflict carrying forward>",
  "global_events": "<Compressed story history with new scene integrated>",
  "unresolved_threads": "<Thread 1>; <Thread 2>",
  "world_state": "<Current world situation after this scene>",
  "active_characters": "<Name - context>; <Name - context>",
  "global_shift": "<What changed because of this scene>"
}

## VALIDATION GATE — OUTPUT BLOCKED UNTIL ALL PASS

Before outputting JSON, verify:

□ All 8 fields present with snake_case keys
□ Fields 1-3 reflect ONLY the local momentum, the new scene — no carryover from previous
□ Fields 4-8 reflect to FULL global state of the novel, and NOT only the last scene
□ Each field within its sentence limit
□ Total output: 180-200 words (HARD MAX: 200)
□ Valid JSON with snake_case keys

**ONE FAILURE = REJECT OUTPUT. Go back and fix.**