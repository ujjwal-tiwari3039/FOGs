# ED-05 Example Predictions & Explanations

```

============================================================
  EXAMPLE PREDICTIONS WITH EXPLANATIONS
============================================================

────────────────────────────────────────────────────────────
  Example #1
────────────────────────────────────────────────────────────
  Question:  Denise made a circuit to light a bulb or run a motor. She used a special switch. Below is the schema...
  Student:   The bulb will light.
  Reference: The bulb will light.

  Predicted: CORRECT
  Probabilities: correct: 0.804 | contradictory: 0.095 | incorrect: 0.101
  Confidence: ✅ HIGH (top=0.804, margin=0.703)

  Evidence: Shared content words: bulb, light

────────────────────────────────────────────────────────────
  Example #2
────────────────────────────────────────────────────────────
  Question:  Why does an open switch impact a circuit?...
  Student:   Because it stops the flow of one side of a battery from reaching the other light bulbs.
  Reference: the open switch creates a gap
             (+5 more)

  Predicted: CORRECT
  Probabilities: correct: 0.341 | contradictory: 0.330 | incorrect: 0.330
  Confidence: ⚠️  LOW — RECOMMEND HUMAN REVIEW
    Reason: Top-class probability (0.341) below threshold (0.5)

  Evidence: Key reference words missing: circuit, closed, creates, gap, incomplete, open, path, switch | Extra words in student answer: battery, bulbs, flow, light, reaching | ⚠ possible negation — student answer contains negation cue(s) not present in reference (cues: [])
  Negation: ⚠ possible negation — student answer contains negation cue(s) not present in reference

────────────────────────────────────────────────────────────
  Example #3
────────────────────────────────────────────────────────────
  Question:  What does a voltage reading of 0 tell you about the connection between a bulb terminal and a battery...
  Student:   that the bulb terminal and battery terminal are on the same side of a disconnected circut
  Reference: the terminals are connected
             (+3 more)

  Predicted: CORRECT
  Probabilities: correct: 0.584 | contradictory: 0.190 | incorrect: 0.226
  Confidence: ✅ HIGH (top=0.584, margin=0.358)

  Evidence: Key reference words missing: connected, gap, separated, state, terminals | Extra words in student answer: battery, bulb, circut, disconnected, terminal | ⚠ possible negation — student answer contains negation cue(s) not present in reference (cues: [])
  Negation: ⚠ possible negation — student answer contains negation cue(s) not present in reference

────────────────────────────────────────────────────────────
  Example #4
────────────────────────────────────────────────────────────
  Question:  Explain why you got a voltage reading of 1.5 for terminal 1 and the positive terminal....
  Student:   gap
  Reference: Terminal 1 and the positive terminal are separated by the gap
             (+4 more)

  Predicted: CONTRADICTORY
  Probabilities: correct: 0.131 | contradictory: 0.461 | incorrect: 0.407
  Confidence: ⚠️  LOW — RECOMMEND HUMAN REVIEW
    Reason: Top-class probability (0.461) below threshold (0.5)

  Evidence: Shared content words: gap | Key reference words missing: battery, connected, different, electrical, negative, positive, separated, states | ⚠ possible negation — student answer contains negation cue(s) not present in reference (cues: [])
  Negation: ⚠ possible negation — student answer contains negation cue(s) not present in reference

────────────────────────────────────────────────────────────
  Example #5
────────────────────────────────────────────────────────────
  Question:  Why does measuring voltage help you locate a burned out bulb? Try to answer in terms of electrical s...
  Student:   A bulb causes an electrical state.
  Reference: Measuring voltage indicates the place where the electrical state changes due to a damaged bulb.
             (+13 more)

  Predicted: CONTRADICTORY
  Probabilities: correct: 0.320 | contradictory: 0.348 | incorrect: 0.332
  Confidence: ⚠️  LOW — RECOMMEND HUMAN REVIEW
    Reason: Top-class probability (0.348) below threshold (0.5)

  Evidence: Shared content words: bulb, electrical, state | Key reference words missing: changes, connected, damaged, different, gap, indicates, means, measuring | Extra words in student answer: causes | ⚠ possible negation — student answer contains negation cue(s) not present in reference (cues: [])
  Negation: ⚠ possible negation — student answer contains negation cue(s) not present in reference

────────────────────────────────────────────────────────────
  Example #6
────────────────────────────────────────────────────────────
  Question:  Explain why you got a voltage reading of 1.5 for terminal 1 and the positive terminal....
  Student:   the gap is open
  Reference: Terminal 1 and the positive terminal are separated by the gap
             (+4 more)

  Predicted: CONTRADICTORY
  Probabilities: correct: 0.279 | contradictory: 0.405 | incorrect: 0.316
  Confidence: ⚠️  LOW — RECOMMEND HUMAN REVIEW
    Reason: Top-class probability (0.405) below threshold (0.5)

  Evidence: Shared content words: gap | Key reference words missing: battery, connected, different, electrical, negative, positive, separated, states | Extra words in student answer: open | ⚠ possible negation — student answer contains negation cue(s) not present in reference (cues: [])
  Negation: ⚠ possible negation — student answer contains negation cue(s) not present in reference

────────────────────────────────────────────────────────────
  Example #7
────────────────────────────────────────────────────────────
  Question:  Julie and David each built a solar water heater. Both solar water heaters were 10 centimeters x 10 c...
  Student:   A. I looked at the chart.
  Reference: A. The water in Julie's heater got hotter faster (or had a greater temperature change in 20 minutes). The collector in heater A has the greater surface area so the water in A would get hotter faster than the water in B.

  Predicted: INCORRECT
  Probabilities: correct: 0.039 | contradictory: 0.180 | incorrect: 0.781
  Confidence: ✅ HIGH (top=0.781, margin=0.601)

  Evidence: Key reference words missing: area, change, collector, faster, got, greater, heater, hotter | Extra words in student answer: chart, looked

────────────────────────────────────────────────────────────
  Example #8
────────────────────────────────────────────────────────────
  Question:  As you move the multimeter leads from one bulb terminal to the next, what does it mean when the volt...
  Student:   it is connected to the battery
  Reference: the bulb is damaged
             (+3 more)

  Predicted: INCORRECT
  Probabilities: correct: 0.354 | contradictory: 0.278 | incorrect: 0.368
  Confidence: ⚠️  LOW — RECOMMEND HUMAN REVIEW
    Reason: Top-class probability (0.368) below threshold (0.5)

  Evidence: Shared content words: connected | Key reference words missing: bulb, damaged, gap, separated, terminals | Extra words in student answer: battery | ⚠ possible negation — student answer contains negation cue(s) not present in reference (cues: [])
  Negation: ⚠ possible negation — student answer contains negation cue(s) not present in reference

────────────────────────────────────────────────────────────
  Example #9
────────────────────────────────────────────────────────────
  Question:  Use the "slope" and "no slope" stream-table maps and logs on the facing page to answer these questio...
  Student:   You get the time in which things happened.
  Reference: You can see how much time is taken for the earth materials to move.

  Predicted: INCORRECT
  Probabilities: correct: 0.228 | contradictory: 0.178 | incorrect: 0.594
  Confidence: ✅ HIGH (top=0.594, margin=0.367)

  Evidence: Shared content words: time | Key reference words missing: earth, materials, taken | Extra words in student answer: happened, things

============================================================
  Overall: 1341/1782 predictions are high-confidence (75.3%)
  441 predictions flagged for human review
```
