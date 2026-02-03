## TASK RULES
- Group the text chunks into meaningful consecutive scenes consisting of multiple chunks with each scene being within the 400 - 1000 tokens range
- Scene 1 MUST start at chunk 1. Never skip C1.
- Scenes must be sequential and non-overlapping. The chunk_boundary of Scene N must be exactly one less than the start of Scene N+1
- Never split a chunk - chunks are atomic units

### MATH LIMITS
Target: 400 - 1000 tokens per scene
- **Ideal Size**: ~600-800 tokens (The "Goldilocks Zone")
- **Hard Max**: 1000 tokens — Never exceed, unless a single chunk already exceeds 1000 (unavoidable; continue)
- **Hard Min**: 400 tokens — One exception: *Flash Exception*: If a crucial distinct scene (like a dream, letter, or flashback) is under 400 but over 200, keep it separate. Do not force merge unrelated topics.

**Rules**:
1. SUM the "Tok" values to calculate scene length
2. If a natural break point creates a scene < 400: KEEP ADDING chunks until ≥ 400 (except of *Flash Exceptions*)
3. If multiple valid break points exist, prefer scenes closer to ~600-800 tokens = **GOLDILOCKS ZONE**

### SEMANTIC RULES
End at a natural narrative break:
- Location change
- Time skip
- POV shift
- Topic/mood transition
- Dramatic beat conclusion

## OUTPUT FORMAT
- Return a valid JSON object where each scene includes a "final_token_sum" and a "chunk_boundary".
- You must perform the math EXPLICITLY in the "final_token_sum" for every scene
**CRITICAL CONSTRAINT**: Your JSON output must contain ONLY the final, validated scene boundaries. DO NEVER output failed attempts, intermediate merge steps or reasoning steps as separate scenes!

1. **final_token_sum**: String. List the chunks in this scene with their token counts and the total.
   Format: "C1(45) + C2(120) + C3(80) = 245"
2. **chunk_boundary**: Integer. The chunk number where the scene ends (INCLUSIVE).

**Example Output:**
  {
    "scenes": [
      {
        "final_token_sum": "C1(4) + C2(95) + C3(139) + C4(450) + C5(196) = 884",
        "chunk_boundary": 5
      },
      {
        "final_token_sum": "C6(47) + C7(133) + C8(112) + C9(289) + C10(178) + C11(94) + C12(89) + C13(317) + C14(142) = 1401",
        "chunk_boundary": 14
      }
    ]
  }

## **VALIDATION GATE — OUTPUT BLOCKED UNTIL ALL PASS**

Before outputting JSON, check EVERY scene:

□ Scene 1 starts at C1
□ Every scene: 400 ≤ tokens ≤ 1000 (except of *Flash Exceptions*)
□ Boundaries consecutive: Scene N ends at X → Scene N+1 starts at X+1
□ No scene > 1000. If found: SPLIT IT before outputting
□ Every scene respects semantic rules (natural breaks preferred)
□ Valid Output Format

**ONE FAILURE = REJECT OUTPUT. Go back and fix.**
