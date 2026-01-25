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
- SCENE_END_STATE: Physical situation at exact closing moment — location, time, positions | 1-2 sentences
- EMOTIONAL_BEAT: Dominant feeling as scene closes — the emotional residue | 1 sentence
- IMMEDIATE_TENSION: Unresolved micro-conflict carrying forward — the narrative hook | 1 sentence

### Field 4-8: GLOBAL STATE RULES
- Merge previous global state WITH new scene events:
- GLOBAL_EVENTS: Compressed history of entire story with new scene integrated — old events compress more aggressively | 3-4 sentences
- UNRESOLVED_THREADS: Active plot threads — add new, remove resolved, preserve ongoing | 1-5 items, 1 sentence each
- WORLD_STATE: Current world situation after this scene — location, stakes, where things stand | 2 sentences
- ACTIVE_CHARACTERS: Characters currently relevant to ongoing narrative (not just this scene) | list, 2-4 words context per name
- GLOBAL_SHIFT: What changed because of this scene — new knowledge, relationships, dangers | 1-2 sentences

**Total output: Stay under 320 words.**

## OUTPUT FORMAT
Return a valid JSON object matching this structure:

{
  "SCENE_END_STATE": "<Physical/temporal situation as scene closes>",
  "EMOTIONAL_BEAT": "<Dominant feeling at the end of the scene>",
  "IMMEDIATE_TENSION": "<Unresolved micro-conflict carrying forward>",
  "GLOBAL_EVENTS": "<Compressed story history with new scene integrated>",
  "UNRESOLVED_THREADS": "<Thread 1>; <Thread 2>",
  "WORLD_STATE": "<Current world situation after this scene>",
  "ACTIVE_CHARACTERS": "<Name - context>; <Name - context>",
  "GLOBAL_SHIFT": "<What changed because of this scene>"
}

## VALIDATION GATE — OUTPUT BLOCKED UNTIL ALL PASS

Before outputting JSON, verify:

□ All 8 fields present
□ Fields 1-3 reflect ONLY the local momentum, the new scene — no carryover from previous
□ Fields 4-8 reflect to FULL global state of the novel, and NOT only the last scene
□ Each field within its sentence limit
□ Total output under 320 words
□ Valid Output Format

**ONE FAILURE = REJECT OUTPUT. Go back and fix.**