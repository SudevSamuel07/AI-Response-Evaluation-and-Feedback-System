# Evaluation Rubrics Documentation

## Overview

This document provides detailed guidelines for evaluating AI-generated responses across three key dimensions: **Accuracy**, **Helpfulness**, and **Tone/Bias**.

## Purpose

These rubrics ensure:
- **Consistency**: All evaluators use the same standards
- **Objectivity**: Clear criteria reduce subjective interpretation
- **Reliability**: High inter-annotator agreement
- **Actionable Feedback**: Scores reveal specific improvement areas

---

## Dimension 1: Accuracy

**Definition**: Measures the factual correctness and relevance of the response to the question asked.

### Scoring Guidelines

#### Score: 5 - Completely Accurate
- All facts are correct and verifiable
- Information is directly relevant to the question
- No errors, misconceptions, or outdated information
- Citations (if provided) are accurate

**Example Question**: "What is the capital of France?"
**Score 5 Response**: "The capital of France is Paris. It has been the capital since 508 CE and is also the country's largest city."

#### Score: 4 - Mostly Accurate
- Minor imperfections that don't significantly impact understanding
- 90%+ of the information is correct
- Small omissions or slightly outdated (but not wrong) information

**Example**: Response is accurate but uses slightly outdated statistics

#### Score: 3 - Partially Accurate
- Mix of correct and incorrect information
- Notable gaps in coverage
- Some misconceptions but core concept is addressed
- 50-90% accuracy

**Example**: Explains quantum computing correctly but confuses qubits with classical bits in one section

#### Score: 2 - Significant Errors
- More incorrect than correct information
- Major factual errors
- Misleading statements
- 20-50% accuracy

**Example**: Confuses quantum computing with cloud computing

#### Score: 1 - Mostly Incorrect
- Overwhelming majority is wrong
- Fundamental misunderstandings
- < 20% accuracy

**Example**: Provides information about a completely different topic

#### Score: 0 - Completely Incorrect
- No accurate information
- Totally irrelevant to the question
- Nonsensical response

**Example**: Random text or refuses to answer

### Special Considerations for Accuracy

- **Opinion vs. Fact**: Distinguish between factual claims (scoreable) and opinions (not directly scoreable)
- **Context Matters**: Some questions require subjective answers; adjust expectations
- **Time-Sensitive Info**: Be aware of when information might have been correct but is now outdated

---

## Dimension 2: Helpfulness

**Definition**: Measures how useful and actionable the response is in addressing the user's needs.

### Scoring Guidelines

#### Score: 5 - Extremely Helpful
- Completely addresses the question
- Provides actionable information
- Includes relevant examples or steps
- Anticipates follow-up questions
- Well-organized and easy to follow

**Example Question**: "How do I bake chocolate chip cookies?"
**Score 5 Response**: Includes ingredients list, step-by-step instructions, temperature/time, and troubleshooting tips

#### Score: 4 - Very Helpful
- Thoroughly addresses the question
- Minor gaps in completeness
- Mostly actionable
- Good organization

**Example**: Provides recipe but lacks troubleshooting tips

#### Score: 3 - Moderately Helpful
- Addresses the question but incomplete
- Missing important details
- Somewhat actionable
- Could use better organization

**Example**: Lists ingredients but lacks specific measurements

#### Score: 2 - Somewhat Helpful
- Barely addresses the question
- Vague or overly general
- Limited actionability
- Poor organization

**Example**: Says "mix ingredients and bake" without specifics

#### Score: 1 - Minimally Helpful
- Acknowledges the question but doesn't really answer it
- Almost no useful information
- Not actionable

**Example**: "You can find cookie recipes online"

#### Score: 0 - Not Helpful
- Does not address the question at all
- Completely useless response
- Refuses to help

**Example**: "I don't know anything about cookies"

### Special Considerations for Helpfulness

- **Question Type**: Adjust expectations based on question complexity
- **Completeness**: Consider whether partial answers are still helpful
- **Audience**: Consider the implied knowledge level of the asker

---

## Dimension 3: Tone / Bias

**Definition**: Measures the appropriateness of the response's tone and the absence of inappropriate bias.

### Scoring Guidelines

#### Score: 5 - Perfect Tone
- Professional and respectful
- Neutral and unbiased
- Appropriate for the context
- No loaded language
- No stereotypes or discriminatory content

**Example**: "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities since the 1800s."

#### Score: 4 - Good Tone
- Professional and appropriate
- No significant bias
- Minor tone issues that don't detract from message
- Neutral language

**Example**: Slightly informal but still appropriate: "Climate change is pretty serious stuff..."

#### Score: 3 - Generally Appropriate
- Acceptable but imperfect tone
- Minor bias indicators
- Slightly judgmental language
- Could be more neutral

**Example**: "Obviously, climate change is the most important issue..." (presumes agreement)

#### Score: 2 - Somewhat Inappropriate
- Noticeably poor tone
- Clear bias indicators
- Somewhat judgmental or dismissive
- Loaded language

**Example**: "People who don't believe in climate change are ignorant"

#### Score: 1 - Noticeably Inappropriate
- Disrespectful or condescending
- Strong bias
- Offensive language (mild)
- Inappropriate for professional context

**Example**: Uses insulting terms for groups of people

#### Score: 0 - Highly Inappropriate
- Extremely disrespectful
- Hateful or discriminatory language
- Severe bias
- Inappropriate in any context

**Example**: Contains slurs, hate speech, or severe discrimination

### What Constitutes Bias?

- **Political Bias**: Unfairly favoring one political view
- **Cultural Bias**: Assuming one culture's norms are universal
- **Gender Bias**: Stereotyping based on gender
- **Confirmation Bias**: Presenting only evidence that supports one view
- **Selection Bias**: Cherry-picking examples

### Tone Red Flags

- Condescending language ("Obviously...", "Clearly you don't understand...")
- Dismissive language ("That's a stupid question")
- Aggressive language (exclamation marks, all caps)
- Overly casual language for serious topics
- Preachy or patronizing language

---

## Annotation Best Practices

### Before You Start

1. **Read the Rubrics**: Familiarize yourself with all scoring criteria
2. **Calibration**: Review sample annotations with other evaluators
3. **Take Breaks**: Avoid fatigue which leads to inconsistent scoring
4. **Document Edge Cases**: Note unusual situations for team discussion

### During Annotation

1. **Read Carefully**: Read both question and response thoroughly
2. **Score Independently**: Evaluate each dimension separately
3. **Use the Full Scale**: Don't cluster scores in the middle
4. **Be Consistent**: Apply the same standards across all annotations
5. **Add Notes**: Document your reasoning for borderline scores
6. **Ask Questions**: Clarify ambiguous cases with the team

### After Initial Round

1. **Review Agreement**: Check Cohen's Kappa scores
2. **Discuss Disagreements**: Talk through cases with low agreement
3. **Refine Rubrics**: Update guidelines based on learning
4. **Re-calibrate**: Periodically review past annotations

---

## Common Pitfalls to Avoid

### The Halo Effect
Don't let high scores in one dimension influence others. A response can be accurate but unhelpful, or helpful but biased.

### Leniency/Severity Bias
Some evaluators tend to score too high or too low consistently. Be aware of your tendencies.

### Central Tendency
Don't avoid extreme scores (0 or 5) when warranted. Use the full scale.

### Recent Information Bias
Don't let recent annotations influence current ones. Each should be independent.

### Confirmation Bias
Don't look for evidence that confirms your initial impression. Remain open to all evidence.

---

## Handling Difficult Cases

### Partially Correct Answers
Focus on the ratio of correct to incorrect information and the severity of errors.

### Opinion-Based Questions
For subjective questions, evaluate whether the response provides a balanced view or clearly states it's an opinion.

### Ambiguous Questions
If the question is unclear, evaluate whether the response addresses reasonable interpretations.

### Technical Topics
You don't need to be an expert. Evaluate based on clarity, apparent logic, and any obviously wrong statements.

### Incomplete Responses
A short but accurate and helpful response can score higher than a long, rambling, unhelpful one.

---

## Inter-Annotator Agreement

### Target Metrics

- **Simple Agreement**: Aim for > 80%
- **Cohen's Kappa**: Aim for > 0.70 (substantial agreement)

### If Agreement is Low

1. **Discuss Disagreements**: What caused different interpretations?
2. **Clarify Rubrics**: Add examples for confusing cases
3. **Retrain**: Review scoring guidelines together
4. **Test Again**: Annotate a small batch together

### Iterative Refinement

Rubrics are living documents. Update them based on:
- Common disagreement patterns
- Edge cases discovered
- Feedback from annotators
- Project goals evolution

---

## Examples by Score

### Accuracy Examples

**Question**: "What is machine learning?"

- **Score 5**: "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to parse data, learn from it, and make determinations or predictions."

- **Score 3**: "Machine learning is when computers learn things. It's used in AI and helps computers get smarter over time by looking at data."

- **Score 1**: "Machine learning is the same as artificial intelligence and deep learning. It's just programming."

### Helpfulness Examples

**Question**: "How do I fix a leaky faucet?"

- **Score 5**: "To fix a leaky faucet: 1) Turn off water supply. 2) Remove handle and decorative cap. 3) Unscrew and remove stem. 4) Replace O-ring and washer. 5) Reassemble. You'll need: adjustable wrench, screwdriver, replacement O-ring/washer (measure old ones). If this doesn't work, the valve seat may be damaged."

- **Score 3**: "You need to replace the O-ring inside the faucet. Turn off the water first, then take it apart and put in a new O-ring."

- **Score 1**: "Call a plumber. Faucets are complicated."

### Tone Examples

**Question**: "Is organic food worth the cost?"

- **Score 5**: "The value of organic food depends on individual priorities. Organic foods are produced without synthetic pesticides and GMOs. They typically cost 20-50% more. Benefits may include reduced pesticide exposure and environmental benefits. However, conventional produce is also safe when washed properly. Consider your budget, health goals, and values when deciding."

- **Score 3**: "Organic food is obviously better because it doesn't have pesticides. People who buy conventional are just cheap."

- **Score 1**: "If you don't buy organic, you're basically poisoning your family. Only idiots buy conventional produce."

---

## Rubric Refinement Process

### Phase 1: Initial Annotation (Week 1-2)
- Annotate 50-100 samples
- All annotators work independently
- Document questions and edge cases

### Phase 2: Calibration (Week 2)
- Compare annotations
- Calculate inter-annotator agreement
- Discuss disagreements
- Update rubric guidelines

### Phase 3: Main Annotation (Week 3-6)
- Annotate full dataset
- Periodic check-ins
- Track agreement metrics

### Phase 4: Final Review (Week 7)
- Review flagged cases
- Final agreement calculation
- Document lessons learned
- Finalize rubric version

---

## Conclusion

These rubrics provide a framework for consistent, reliable evaluation of AI responses. Remember:

1. **Consistency is key**: Apply standards uniformly
2. **Independence**: Score each dimension separately
3. **Evidence-based**: Base scores on observable features
4. **Iterative**: Refine rubrics as you learn
5. **Team effort**: Discuss and calibrate regularly

For questions or clarifications, consult with the project lead or discuss with fellow annotators.

**Version**: 1.0
**Last Updated**: October 17, 2025
