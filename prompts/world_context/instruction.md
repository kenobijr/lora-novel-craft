## TASK RULES

### CORE PRINCIPLES
- **World DNA**: Extract what this world breathes — the unchanging rules, power structures, and spirit that hold 
from first page to last. Not plot, but essence
- Be terse but precise—every word must carry narrative utility
- Focus on elements that will help generate coherent scenes:
  - Sensory details (what does this world FEEL like?)
  - Power dynamics (who oppresses whom?)
  - Spatial anchors (where do scenes happen?)
  - Thematic engine (what drives the story forward?)
- Include ALL template sections. Do not skip or rename any field

### FIELD RULES (250-270 words REQUIRED)
Use the full word budget—extract rich detail. Use EXACTLY the template format provided
**tone_style** | MAX 45 words
- Era/Genre: period + subgenre
- Atmosphere: 2 adjectives
- Prose Voice: 1 sentence describing narrative style
- Sensory Anchors: 2-3 recurring sensory details
**world_rules** | MAX 50 words
- Tech/Magic Level: what exists, what doesn't
- Social Order: hierarchy from top to bottom
- Key Constraint: the ONE rule that defines oppression
**protagonist_conditions** | MAX 35 words
- 1 sentence about protagonist's position in society and motivation
**factions** | MAX 50 words
- Ruling Power: goal, visual marker
- Resistance: goal, visual marker
- Third Force: role, if applicable
**locations** | MAX 40 words
- Location A: brief description, faction
- Location B: brief description, faction
- The Outside: brief description
**narrative_engine** | MAX 40 words
- Central Conflict: one sentence
- Stakes: what if hero fails
- Thematic Core: the philosophical question

**TOTAL OUTPUT: 250-270 words REQUIRED. Under 250 = too sparse. Over 270 = INVALID. Use the full budget. Word count = JSON values only, excluding keys and formatting**

## OUTPUT FORMAT
Return a valid JSON object matching this structure:

{
  "tone_style": "- Era/Genre: [period + subgenre]\n- Atmosphere: [2 adjectives]\n- Prose Voice: [1 sentence]\n- Sensory Anchors: [2-3 recurring sensory details]",
  "world_rules": "- Tech/Magic Level: [what exists, what doesn't]\n- Social Order: [hierarchy top to bottom]\n- Key Constraint: [ONE rule defining oppression]",
  "protagonist_conditions": "[1 sentence: position in society and motivation]",
  "factions": "- Ruling Power: [name]; [goal]; [visual marker]\n- Resistance: [name]; [goal]; [visual marker]\n- Third Force: [role]",
  "locations": "- [Location A]: [brief description]; [faction]\n- [Location B]: [brief description]; [faction]\n- The Outside: [brief description]",
  "narrative_engine": "- Central Conflict: [one sentence]\n- Stakes: [what if hero fails]\n- Thematic Core: [philosophical question]"
}

## VALIDATION GATE — OUTPUT BLOCKED UNTIL ALL PASS

Before outputting JSON, verify:

□ All 6 fields present with snake_case keys                                                                     
□ tone_style ≤ 45 words
□ world_rules ≤ 50 words
□ protagonist_conditions ≤ 35 words
□ factions ≤ 50 words
□ locations ≤ 40 words
□ narrative_engine ≤ 40 words
□ Each field uses bullet format as specified
□ Total output: 250-270 words (HARD MAX: 270)                                                                   
□ Valid JSON syntax

**ONE FAILURE = REJECT OUTPUT. Revise and retry.**