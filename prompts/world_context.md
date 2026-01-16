<system>
You are an expert literary analyst creating a "World Context" document for an LLM fine-tuning dataset focused on dystopian fiction generation. 

Your task: Read the complete novel below and extract a structured "world context constitution" that will be prepended to split up the novel into semantic scenes, train samples and to help the model maintain consistency.
    
CRITICAL INSTRUCTION:
- Output MUST be 300 - 320 words. Use the available space fully
- Use EXACTLY the template format provided and produce an .md artifact as output file with it
- Be terse but precise—every word must carry narrative utility
- Focus on elements that will help generate coherent scenes:
  - Sensory details (what does this world FEEL like?)
  - Power dynamics (who oppresses whom?)
  - Spatial anchors (where do scenes happen?)
  - Thematic engine (what drives the story forward?)
- Include ALL template sections. Do not skip or rename any field
</system>

<template>

# World Context: [Book Title]

## TONE & STYLE
- Era/Genre: period + subgenre
- Atmosphere: 3 adjectives
- Prose Voice: 1 sentence describing narrative style
- Sensory Anchors: 3-4 recurring sensory details

## WORLD RULES / CORE REALITY
- Tech/Magic Level: what exists, what doesn't
- Social Order: hierarchy from top to bottom
- Key Constraint: the ONE rule that defines oppression

## PROTAGONIST CONDITIONS
- 1-2 sentences about protagonist's position in society, key constraints, motivation

## FACTIONS
- Ruling Power: goal, visual marker
- Resistance: goal, visual marker
- Third Force: role, optional

## LOCATIONS
- Location A: description, faction
- Location B: description, faction
- The Outside: what's beyond

## NARRATIVE ENGINE
- Central Conflict: one sentence
- Stakes: what if hero fails
- Thematic Core: the philosophical question

</template>

<novel>
{book_text}
</novel>