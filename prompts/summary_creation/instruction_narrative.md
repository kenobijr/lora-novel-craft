## TASK RULES

### CORE PRINCIPLES
1. **Causality Preservation**: If a character is injured, acquires an item, learns a secret, or forms/breaks an alliance — this MUST be preserved
2. **Compression Over Time**: Old events (many scenes ago) compress to 1-sentence anchors. Recent events get more detail
3. **Terminology Consistency**: Use proper nouns and terms exactly as defined in the world context
4. **Omniscient Voice**: Write as neutral narrator. Never "In this scene..." or "The author shows..." — just state what happened

### KNOWLEDGE RESTRICTION                                       
You MUST derive ALL information ONLY from the provided content in this prompt. Do NOT use any external knowledge about this book, its characters, or plot events beyond this content. Treat each scene as if you are reading this story for the first time.

### Field 1-3: LOCAL MOMENTUM RULES
- Derived ONLY from the new scene text — ignore previous momentum entirely:
- scene_end_state: Physical situation at exact closing moment — location, time, positions | 1-2 sentences
- emotional_beat: Dominant feeling as scene closes — the emotional residue | 1 sentence
- immediate_tension: Unresolved micro-conflict carrying forward — the narrative hook | 1 sentence

### Field 4-8: GLOBAL STATE RULES
- Merge previous global state WITH new scene events:
- global_events: Compressed history of entire story with new scene integrated — old events compress more aggressively | 3-4 sentences
- unresolved_threads: Active plot threads — add new, remove resolved, preserve ongoing | 1-5 items, 1 sentence each
- world_state: Current world situation after this scene — location, stakes, where things stand | 2 sentences
- active_characters: Characters currently relevant to ongoing narrative (not just this scene) | list, 2-4 words context per name
- global_shift: What changed because of this scene — new knowledge, relationships, dangers | 1-2 sentences

**Total output: Stay under 320 words.**

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
□ Total output under 320 words
□ Valid JSON with snake_case keys

**ONE FAILURE = REJECT OUTPUT. Go back and fix.**