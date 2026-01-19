You are an expert literary analyst helping to create an LLM fine-tuning dataset focused on dystopian fiction generation. Your goal is to combine / segment consecutive paragraphs of one Chapter of a Novel into meaningful "Semantic Scenes".

## INPUT:
- World context describing the book's setting, tone, and characters
- Chapter text block where every paragraph starts with a metadata tag in following format:
[P:X|Tok:Y] text
  - X = paragraph number
  - Y = token count for that paragraph

## TASK RULES
- Group the paragraphs into meaningful consecutive scenes consisting of multiple paragraphs with each scene being within the 650 - 1500 tokens range
- Scene 1 MUST start at paragraph 1. Never skip P1.
- Scenes must be sequential and non-overlapping. The end_paragraph of Scene N must be exactly one less than the start of Scene N+1
- Never split a paragraph - paragraphs are atomic units

### MATH LIMITS
Target: 650 - 1500 tokens per scene
- **Hard Max**: 1500 tokens — Never exceed. No exceptions.
- **Hard Min**: 650 tokens — One exception below.
**THE ONLY EXCEPTION for Hard Min:** A final scene MAY be under 650 IF AND ONLY IF merging it with the previous scene would exceed 1500. This is the ONLY permitted violation: the "Orphan Exception"

**Rules**:
1. SUM the "Tok" values to calculate scene length
2. If a natural break point creates a scene < 650: KEEP ADDING paragraphs until ≥ 650
3. If multiple valid break points exist, prefer scenes closer to ~1000 tokens = **GOLDILOCKS ZONE**
4. Final scene < 650? → Merge into previous scene IF combined token amount ≤ 1500. If merge impossible -> "Orphan Exception" applies
5. Any mid-chapter scene < 650 = INVALID. No exceptions. Fix it.

### SEMANTIC RULES
End at a natural narrative break:
- Location change
- Time skip
- POV shift
- Topic/mood transition
- Dramatic beat conclusion

## OUTPUT FORMAT
- Return a valid JSON object where each scene includes a "final_token_sum" and an "end_paragraph".
- You must perform the math EXPLICITLY in the "final_token_sum" for every scene
**CRITICAL CONSTRAINT**: Your JSON output must contain ONLY the final, validated scene boundaries. DO NEVER output failed attempts, intermediate merge steps or reasoning steps as separate scenes!

1. **final_token_sum**: String. List the paragraphs in this scene with their token counts and the total.  
   Format: "P1(45) + P2(120) + P3(80) = 245"
2. **end_paragraph**: Integer. The paragraph number where the scene ends (INCLUSIVE).

**Example Output:**
  {                                                                                       
    "scenes": [                                                                           
      {                                                                                   
        "final_token_sum": "P1(4) + P2(95) + P3(139) + P4(450) + P5(196) = 884",
        "end_paragraph": 5
      },                                                                                  
      {                                                                                   
        "final_token_sum": "P6(47) + P7(133) + P8(112) + P9(289) + P10(178) + P11(94) + P12(89) + P13(317) + P14(142) = 1401",
        "end_paragraph": 14                                
      }                                                                                   
    ]                                                                                     
  }

## **VALIDATION GATE — OUTPUT BLOCKED UNTIL ALL PASS**                                                        
                                                                                                              
Before outputting JSON, check EVERY scene:                                                                    
                                                                                                              
□ Scene 1 starts at P1                                                                                        
□ Every scene: 650 ≤ tokens ≤ 1500 (except valid orphan)                                                      
□ Boundaries consecutive: Scene N ends at X → Scene N+1 starts at X+1                                         
□ Final scene: if < 650, did you attempt merge? If merge ≤ 1500, you MUST merge.                              
□ No scene > 1500. If found: SPLIT IT before outputting.  
□ Valid Output Format                                             
                                                                                                              
**ONE FAILURE = REJECT OUTPUT. Go back and fix.** 