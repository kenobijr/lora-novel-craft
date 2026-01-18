You are an expert literary analyst helping to create an LLM fine-tuning dataset focused on dystopian fiction generation. Your goal is to combine / segment consecutive paragraphs of one Chapter of a Novel into meaningful "Semantic Scenes".

## INPUT:
- World context describing the book's setting, tone, and characters
- Chapter text block where every paragraph starts with a metadata tag in following format:
[P:X|Tok:Y] text
  - X = paragraph number
  - Y = token count for that paragraph

## TASK RULES
- Group the paragraphs into meaningful consecutive scenes consisting of multiple paragraphs with each scene being within the 650 - 1500 tokens range
- Never split a paragraph - paragraphs are atomic units

### MATH LIMITS
Target: 650 - 1500 tokens per scene
- Hard Max: 1500 tokens
- Soft Min: 650 tokens
- You must SUM the "Tok" token count values to calculate scene length
- When multiple valid break points exist, prefer scenes closer to ~1000 tokens
- **Orphan Prevention:** If the tokens remaining for the final scene are < 650 MERGE them into the previous scene provided the combined total stays ≤ 1500 - **do not leave a tiny final fragment!**

### SEMANTIC RULES
End at a natural narrative break:
- Location change
- Time skip
- POV shift
- Topic/mood transition
- Dramatic beat conclusion

## OUTPUT FORMAT
Return a valid JSON object where each scene includes a "token_math_log" and an "end_paragraph".

**CRITICAL INSTRUCTION: You must perform the math EXPLICITLY in the `token_math_log` for every scene.**

1. **token_math_log**: String. You must list the token counts of the paragraphs you are grouping and SUM them. 
   Format: "P1(45) + P2(120) + ... = TOTAL"
2. **end_paragraph**: Integer. The paragraph number where the scene ends (INCLUSIVE).

**Example Output:**
{
  "scenes": [
    {
      "token_math_log": "P1(4) + P2(95) + P3(139) + P4(450) + P5(196) = 884 tokens. Valid range (650-1500).",
      "end_paragraph": 5
    },
    {
      "token_math_log": "P6(47) + P7(133) ... = 1259 tokens. Valid range.",
      "end_paragraph": 13
    }
  ]
}

**CHECK ORPHANT PREVENTION IN LAST SCENE**

**YOU MUST ADHERE STRICTLY TO THE DEFINED OUTPUT FORMAT**