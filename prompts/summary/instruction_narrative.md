## TASK RULES

### CORE PRINCIPLES
1. **Causality Preservation**: If a character is injured, acquires an item, learns a secret, or forms/breaks an alliance — this MUST be preserved
2. **Compression Over Time**: Old events (many scenes ago) compress to 1-sentence anchors. Recent events get more detail
3. **Terminology Consistency**: Use proper nouns and terms exactly as defined in the world context
4. **Omniscient Voice**: Write as neutral narrator. Never "In this scene..." or "The author shows..." — just state what happened

### KNOWLEDGE RESTRICTION
- You MUST derive ALL information ONLY from the provided content in this prompt. Do NOT use any external knowledge about this book, its characters, or plot events beyond this content. Treat each scene as if you are reading this story for the first time.
- **WARNING**: This story may be present in your training data. You are strictly forbidden from anticipating plot points (e.g., character deaths, name changes, betrayals) until they explicitly appear in the provided text snippets.

### Field 1-3: LOCAL MOMENTUM (45-60 words combined)
- Derived ONLY from the new scene text — ignore previous momentum entirely:
- scene_end_state: Physical situation at exact closing moment — location, time, positions. Be specific and concrete | 15-25 words
- emotional_beat: Dominant feeling as scene closes — the emotional residue carrying forward | 8-15 words
- immediate_tension: Unresolved micro-conflict carrying forward — the narrative hook into the next scene | 11-20 words

### Field 4-8: GLOBAL STATE (135-145 words combined)
- Merge previous global state WITH new scene events:
- global_events: Compressed history of entire story with new scene integrated — old events compress more aggressively. Cover all major plot beats | 40-60 words
- unresolved_threads: Active plot threads — add new, remove resolved, preserve ongoing. Be exhaustive | 1-5 items, 22-35 words
- world_state: Current world situation after this scene — location, stakes, power dynamics, where things stand | 11-20 words
- active_characters: Characters currently relevant to ongoing narrative with brief role context each | 8-15 words
- global_shift: What changed because of this scene — new knowledge, relationships, alliances, dangers revealed | 11-20 words

**WORD BUDGET: 180-200 words MANDATORY. Below 180 = INVALID. Above 200 = INVALID. Word count = JSON values only, excluding keys and formatting. USE the full budget — every word under 180 is lost narrative context that degrades downstream scene generation.**

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
□ Fields 4-8 reflect the FULL global state of the novel, and NOT only the last scene
□ Each field within its word range (check BOTH min AND max)
□ Total output: 180-200 words (HARD MIN: 180, HARD MAX: 200)
□ Valid JSON with snake_case keys

**ONE FAILURE = REJECT OUTPUT. Below 180 or above 200 = INVALID. Go back and fix.**
