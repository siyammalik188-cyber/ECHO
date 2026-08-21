# ECHO 7–11 — Consolidated summary

Five challenges, built in order, each on top of the last rather than beside it. Every number below is recomputed by re-running the studies when this file is generated, so it cannot drift from the individual reports.

**511 tests pass.** Everything runs offline.

## Capability table

| Capability | Demonstrated? | Evidence |
| --- | --- | --- |
| **Memory** | Yes | Extraction is a separate gated step, never a transcript dump; three kinds of state kept structurally apart. `README.md` |
| **Belief revision** | Yes | Log-odds updates weighted by source reliability and relevance; every revision preserves the previous state. `docs/ECHO_2_BELIEF_REVISION.md` |
| **Prediction** | Yes | Sealed before the outcome exists, temporal-leakage tests, proper scoring. `docs/ECHO_3_PREDICTION.md` |
| **Learning** | Yes, narrowly | Post-change Brier 0.3957 → 0.3133 on ENV-B and 0.3051 → 0.1723 on ENV-D, with two false alarms reported as costs. `docs/ECHO_4_LEARNING.md` |
| **Discovery** | Yes | Found `PRODUCT(CHANGE(X2), LAG(X4, 3))` from 14,112 candidates; null and overfit-trap worlds correctly produced no discovery. `docs/ECHO_5_DISCOVERY.md` |
| **Transfer** | Yes | 3/3 structurally matching targets carried the shape; the statistically identical false-analogy world rejected it (edge −0.0015, t = −0.72). `docs/ECHO_6_TRANSFER.md` |
| **Experimentation** | Yes | Information-gain selection reached P(truth) 0.983 vs random 0.794 at 10 vs 23 cost; useless experiments scored exactly 0 bits without being told. `docs/ECHO_7_EXPERIMENTATION.md` |
| **Causal inference** | Yes, narrowly | Markov-equivalent worlds left at exactly 1/3 each after observation, resolved to 1.000 by intervention. Structure is chosen from four supplied models, not learned. `docs/ECHO_8_CAUSAL.md` |
| **Metacognition** | Yes, as measurement | Self-ranking of its own domains agreed with the next phase on 10/10 pairs; abstention lifted answered accuracy from 75.3% to 83.3% at a stated cost of 26 correct answers foregone. `docs/ECHO_9_METACOGNITION.md` |
| **Social learning** | Yes | Learned reliability ordering matched the truth exactly; 94.1% vs majority vote 58.8%, and 8/8 on rounds where the majority was wrong by construction. `docs/ECHO_10_SOCIAL_LEARNING.md` |
| **Integrated operation** | Yes, as composition | Eleven modules run as one cycle with end-to-end provenance; Brier 0.1404 vs baseline 0.2491, 3/3 injected failures detected, state survives restart. It does **not** recover from the regime change. `docs/ECHO_11_INTEGRATION.md` |

## What did not work

Collected here so it is not spread thin across five reports. Each of these is measured, not anticipated.

1. **ECHO never recovers from the regime change.** Discovery runs once, over the pre-change history, so the fitted relationship keeps predicting the old regime after the world reverses. The baseline recovers in 48 ticks; ECHO's rolling Brier never returns to its pre-change level within the run.
2. **Transfer buys no accuracy where data is plentiful.** In ECHO 6 and again in ECHO 11 the novel-world Brier is identical to cold start (0.1444 both). The benefit is search cost and sample efficiency, not accuracy, and claiming otherwise would be false.
3. **Chasing false alarms costs something.** Both stationary environments in ECHO 4 produced a confirmed regime-change detection; one cost +0.0025 Brier.
4. **A WEAK domain still answered.** In ECHO 9 the abstention gate is on meta-confidence, not on the competence band, so `transfer` — banded WEAK — kept answering and was confidently wrong 17 times.
5. **Reliability and competence are single numbers.** A source with genuine topic expertise has it averaged away; a domain-level abstention cannot notice an easy question inside a weak area.
6. **The ECHO 1.5 evaluation has never run.** The Anthropic account has no credits. Every results section in that report is marked NOT MEASURED with no placeholder numbers.

## Bugs found while building this

| Bug | How it surfaced | Fix |
| --- | --- | --- |
| `Conversation.__len__` made an empty conversation falsy, so `x or Conversation()` silently discarded a caller's system prompt | a test on system-prompt handling | `is None` |
| `LearningLedger.__len__` — same class of bug. `if ledger:` was false at zero experiences, so the incumbent strategy was never registered and **every promotion comparison was vacuous** | the candidate always won | `is not None`, plus a comment on every ledger since |
| Calibration buckets: `0.7 / 0.1 = 6.999…`, so a stated 0.7 was binned as 0.6–0.7, misreporting calibration at every boundary in the same direction | a bucket test spanning 0.0–1.0 | `int(p * buckets + 1e-9)` |
| `Claim.brier` scored the belief against *whether ECHO was right* rather than against the proposition's truth, so a confident 0.12 call on a false proposition scored as one of the worst possible | every domain banded WEAK, including one at 94.2% accuracy | store `outcome`, derive `correct` |
| The source-turn detector fired on all four agents when one had turned | a short window makes an ordinary run of bad luck look like a turn | widen to 24 and require the recent record to be genuinely poor, not merely below its own average |
| Two ECHO 6 target worlds had large offsets on the lagged factor, leaving it strictly positive and collapsing the interaction into a scaled `CHANGE` | ECHO correctly preferred the simpler component, returning PARTIAL where FULL was expected | fix the generators; no threshold touched |
| A naive source audit for simulated self-awareness failed on the module's own docstring promising not to do the thing | the test could not distinguish prose from code | `tests/source_audit.py`, which strips docstrings and comments but keeps string literals, plus a test proving it still catches a real cheat |

## What none of this establishes

ECHO is an experimental adaptive reasoning architecture. The measurements above are what they are and nothing more.

- **Not understanding.** ECHO reuses a shape without any account of why a change multiplied by a delay predicts anything. It manipulates conditional probability tables over meaningless labels.
- **Not self-awareness.** The metacognition layer is a scoreboard with thresholds on it. When it abstains, that is a comparison between two floats. It has no access to its own reasoning and no representation of itself beyond a table of past scores.
- **Not autonomy.** The loop is a fixed sequence written by a person. ECHO does not choose what to do next and cannot modify itself.
- **Not theory of mind.** The deceptive agent is detected as a low number, not as a liar. There is no model of what another agent believes or wants.
- **Not general intelligence.** Every capability is bounded: a finite hypothesis language, a supplied model set, three-variable causal graphs, synthetic worlds with one seed each.
- **Nothing here bears on consciousness or sentience**, and no measurement in any of these reports was designed to, or could.

## Scope held throughout

No internet access, no shell execution, no source-code rewriting, no credential access, no financial-market connection, no uncontrolled agents. Every environment is a seeded simulation. A test walks every module under `echo/` and asserts that `subprocess`, `socket`, `urllib`, `requests`, `os.system`, `eval(`, `exec(` and `__import__(` appear nowhere in the code.
