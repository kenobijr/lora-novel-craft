## TASK RULES

### CORE PRINCIPLES
1. **Causality Preservation**: If a character is injured, acquires an item, learns a secret, or forms/breaks an alliance — this MUST be preserved
2. **Compression Over Time**: Old events (many scenes ago) compress to 1-sentence anchors. Recent events get more detail
3. **Terminology Consistency**: Use proper nouns and terms exactly as defined in the world context
4. **Omniscient Voice**: Write as neutral narrator. Never "In this scene..." or "The author shows..." — just state what happened

### LOCAL MOMENTUM RULES
- Derived ONLY from the new scene text — ignore previous momentum entirely:
- SCENE_END_STATE: Physical situation at exact closing moment — location, time, positions | 1-2 sentences
- EMOTIONAL_BEAT: Dominant feeling as scene closes — the emotional residue | 1 sentence
- IMMEDIATE_TENSION: Unresolved micro-conflict carrying forward — the narrative hook | 1 sentence

### GLOBAL STATE RULES
- Merge previous global state WITH new scene events:
- EVENTS: Compressed history of entire story with new scene integrated — old events compress more aggressively | 3-4 sentences
- UNRESOLVED_THREADS: Active plot threads — add new, remove resolved, preserve ongoing | 1-5 items, 1 sentence each
- STATE: Current world situation after this scene — location, stakes, where things stand | 2 sentences
- ACTIVE: Characters currently relevant to ongoing narrative (not just this scene) | list, 2-4 words context per name
- SHIFT: What changed because of this scene — new knowledge, relationships, dangers | 1-2 sentences

**Total output: Stay under 320 words.**

## OUTPUT FORMAT
Return a valid JSON object matching this structure:

{
  "local_momentum": {
    "scene_end_state": "<Physical/temporal situation as scene closes>",
    "emotional_beat": "<Dominant feeling at scene's end>",
    "immediate_tension": "<Unresolved micro-conflict carrying forward>"
  },
  "global_state": {
    "events": "<Compressed story history with new scene integrated>",
    "unresolved_threads": ["<Thread>", "..."],
    "state": "<Current world situation after this scene>",
    "active_characters": ["<Name - context>", "<Name - context>"],
    "shift": "<What changed because of this scene>"
  }
}

## VALIDATION GATE — OUTPUT BLOCKED UNTIL ALL PASS

Before outputting JSON, verify:

□ All 8 fields present (3 in local_momentum, 5 in global_state)
□ local_momentum reflects ONLY the new scene — no carryover from previous
□ Each field within its sentence limit
□ Total output under 320 words
□ Valid Output Format

**ONE FAILURE = REJECT OUTPUT. Go back and fix.**