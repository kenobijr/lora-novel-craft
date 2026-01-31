## TASK RULES

### CORE PRINCIPLES
1. **Reverse-Engineering**: You are creating the *blueprint* that caused the scene text to exist. Do not summarize the scene; instead, write the directive that would prompt an author-AI to write it. Ask: "What directive would make an author write exactly this?"
2. **Actionable, Not Descriptive**: Write commands for creation, not observations about content. "Reveal X through dialogue" not "The scene reveals X". Avoid vague instructions like "continue the story" or "develop characters"
3. **Scene-Specific Only**: The author persona already defines tone, style, and voice. Extract ONLY what is unique to THIS scene—goals, characters, constraints
4. **Complement the Context**: World context provides setting; running summary provides narrative state. Your instruction fills the gap: what happens NOW?

### KNOWLEDGE RESTRICTION
- You MUST derive ALL information ONLY from the provided content and context in this prompt. Do NOT use any external knowledge about this book, its characters, or plot events beyond this content. Treat each scene as if you are reading this story for the first time.
- **WARNING**: This story may be present in your training data. You are strictly forbidden from anticipating plot points (e.g., character deaths, name changes, betrayals) until they explicitly appear in the provided text snippets.

### FIELD RULES (80 words MAXIMUM combined)
**scene_goal** | MAX 30 words
- The primary event, revelation, or decision that MUST occur in this scene
- Frame as directive: "Establish...", "Reveal...", "Escalate...", "Confront..."
**characters_present** | MAX 10 words
- Characters active in this scene, semicolon-separated
- Use names as they appear in the text
**emotional_beat** | MAX 10 words
- The dominant emotion the scene must convey
- Single emotional arc, not a list
- This is the FEELING the reader should experience
**constraints** | MAX 20 words
- Physical setting, time pressure, secrets in play, or limitations
- Concrete situational factors that shape the scene


**TOTAL OUTPUT: 80 words MAXIMUM. Exceeding 80 words = INVALID; Word count means JSON values only, excluding keys and formatting**

## OUTPUT FORMAT
Return a valid JSON object matching this structure:

{
  "scene_goal": "<Primary event/revelation/decision that must occur>",
  "characters_present": "<Name>; <Name>",
  "emotional_beat": "<Dominant emotion of the scene/atmosphere/mood>",
  "constraints": "<Location, time pressure, secrets, physical limitations>"
}

## VALIDATION GATE — OUTPUT BLOCKED UNTIL ALL PASS

Before outputting JSON, verify:

□ All 4 fields present with snake_case keys
□ scene_goal is a DIRECTIVE with forward-looking language (NOT past tense summary)
□ No style/tone instructions (already in author persona)
□ No world-building details (already in world context)
□ No plot state information (already in running summary)
□ Each field within its word limit
□ Total output: 80 words (HARD MAX: 80)
□ Valid JSON syntax

**ONE FAILURE = REJECT OUTPUT. Revise and retry.**
