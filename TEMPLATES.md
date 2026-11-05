# TEMPLATES.md — the 5 Atropos templates (the hypothesis-killing system)

Five copyable templates: a prereg before the data, a diagnostic screen
without gates, a robustness battery, a post-mortem after death, a
subagent briefing. Each template = a skeleton with `<...>` placeholders —
filled in BEFORE the data is touched, with exactly the numbers the
slots ask for. The rule to live by: zero parameters after the data,
zero calendars, zero "might come in handy".

> The examples in the "Example output" blocks quote preregs and logs
> from a closed donor corpus — none of those files ship with this
> package or exist in your workspace. The workspace layout
> (`hypotheses/`, graveyard, registry) is scaffolded with
> `atropos init` (files are generated, never overwritten).

---

## Template 1. PREREG — pre-declaration before the first data point

Freezes the hypothesis, the design, the nulls and the kill criteria
BEFORE data collection; committing the file = the protocol's immutable
hash. The gold standard — the seven gaps of the OFI prereg: cadence,
executable economics, latency, directionality, pooling, de-clustering,
gates.

```text
# PREREG: <hypothesis name> (v1.0)

Date: YYYY-MM-DD. Status: frozen BEFORE the first data point.
The commit of this file = the protocol hash. Any edit = a new version;
the old test is void. Forward-only.

## 1. Hypothesis (1 line)
<Factor X predicts Y over horizon L. Not Z, not W —
<a one-line refinement of the target>.>

## 2. Frozen design (parameters / thresholds / normalization)
- Cadence: L_primary = <...>; L_secondary = {...} — robustness only,
  they CANNOT resurrect.
- Forward returns are NON-OVERLAPPING (otherwise HAC/Newey-West,
  lag >= L).
- Normalization: <e.g. a z-score by rolling std, window = N bars —
  the number is frozen>.
- Economics: executable (half-spread entry+exit + fees), NOT mid.
  "'<X> bp on mid' is not the '<X> bp you get to keep'".
- Latency gate: half-life > 2 × (poll interval + order RTT),
  not 2 × one-way; <ts_recv − ts_exchange> and <order_ack − order_send>
  are logged separately.
- Directionality: <entry threshold, one-sided/two-sided — before data>.
  The abs variant goes to the log only, never to a kill.
- Pooling: pooled with instrument fixed effects OR per-instrument +
  sign aggregation — exactly ONE is frozen before collection.
- De-clustering: events spaced >= L apart; "200 consecutive events in
  one window = 1 event"; N_min is computed over INdependent events.
- Power (recorded as a calculation, not an arbitrary choice): N = <...>
  gives t ~ 2 at <effect>; N_min = <...>.
- Data quality: exclude <level gaps / stale > Xs / zero volume>.

## 3. Null controls (3-6, each with a threshold)
| #  | Control                                          | Threshold              |
|----|--------------------------------------------------|------------------------|
| N1 | Placebo: 200 shuffled <labels/days> (seed)       | p < 0.05               |
| N2 | Incremental over controls <list>                 | t > 2 AND ΔR² >= <threshold> |
| N3 | Sign stable across <terciles/years> (pre-defined) | <threshold>          |
| N4 | <repeat measurements / second asset / costs>     | <threshold>            |

## 4. Kill criteria (verbatim, pre-declared)
KILL: <t < 2 incremental> OR <placebo p > 0.05> OR <economics <
threshold> on executable>.
Failing any one closes the family. No "but if you look at it from
another angle".

## 5. Immutable hash
The commit of this file = the hash BEFORE the first data point. An
edit after = a new version; the old test is void. Without this,
fitting the data is inevitable.

## 6. Honest forecast
Prior probability of passing: <X>%. What will most likely kill it:
<the specific gate with its number>. Being killed for the right
reason is success, not failure.

## 7. The queue after the commit
STOP: <how much data/time to accumulate>. Do NOT launch the parallel
families <the ones that produce correlated reads>. Their preregs can
be drafted with this same template, without committing.

## FORBIDDEN
- CALENDAR TERMS: <"the 2022 bear -> +15", yearly regimes, seasonal
  windows> — this is fitting known history; it does not transfer OOS
  (see the GLM-spot grave: pilot +0.53 -> OOS +0.053).
- Changing thresholds/windows/normalization after the first data point.
- A second run after a failed first = a new prereg, not an edit.
```

**Example output:** `PREREG_OFI_SHADOW_2026-11-02.md` — the gold
standard (§1-§7 = the seven gaps; kill: "t < 2 incremental OR
placebo p > 0.05"; §12 honest forecast: "gate 1 most likely kills at
5-25 bp instead of 40 — but for the right reason"). The forward-gated
variant: `PREREG_BAB_TIER2_2026-11-01.md` (52 weeks, K-F/K-D/K-DIR,
"any single failure is the S5 fate"). The portfolio + capacity
variant: `PREREG_PORT_VARIANT_C_2026-11-02.md` (the $50-70k frame was
revised down to $20-30k on a recount — an honest self-check before
the run).

---

## Template 2. DIAGNOSTIC SCREEN — reconnaissance without gates

A data screen with no gates and no verdicts: the candidate either
dies here visibly or graduates to a dated prereg. The cleanest kill —
XS-reversion: negative EV in almost every year at t −3.19.

```text
# DIAGNOSTIC SCREEN: <name> — NO GATES, NO VERDICT
Status: DIAGNOSTIC ONLY. This run yields no verdicts; the outcomes
are "grave", "premise death", or "dated prereg". No grails.

## 1. Disclosures (house findings relevant to the hypothesis)
(a) <a sister/cousin is already closed: what exactly, the numbers,
    the log path — why this test is NOT a repackaging (corr
    threshold)>;
(b) <survivorship: a panel of living listings — long-side reads are
    capped from above, shorts never see the worst>;
(c) turnover kill-line: > 20 RT/yr/symbol = automatic death;
(d) <a known negative prior on the family — why we check anyway
    (incremental, not a grail-flip)>.

## 2. Data: paths + limitations
- Panel: <data/...>, n = <...> symbols (excluded: <who, why>),
  grid <YYYY-MM-DD .. YYYY-MM-DD> (<N> days).
- Short windows: <file, coverage> — SHORT WINDOW DISCLOSED LOUDLY.

## 3. Premise check: does the condition even exist?
<Factor dispersion across the cross-section: p10/p50/p90; event
frequency; share of live days>. No premise — DEAD on the premise,
we go no further.

## 4. Signal test: raw effect + t + by year
- Book convention: <ranks, costs, timing without look-ahead>.
- Result: n=<...>  EV <...>%  t <...>  hit <...>%
- By year:
    <2019: <...>% (n=...)>
    <2020: <...>% (n=...)>
    <...>
  Sign consistency: <k/N> years.
- By asset: positive in <k/N>.

## 5. Incremental test: regression with controls
The factor coefficient AFTER controls <list of known factors>;
incremental t = <...>. t < 2 = the effect is the echo of a known
factor.

## 6. Null distributions (200 draws, seed pinned)
- Shuffle/permutation/placebo: <what is destroyed, what remains>.
- Real = <...> percentile of its own placebo sample (<k>/200 draws
  match-or-beat).

## 7. Honest outcome: what death looks like (report verbatim)
If dead — report verbatim: "<factor> does not predict: EV <...>%
t <...>, negative in <k/N> years, positive in <k/N> assets —
grave". No "but it lives at another window".
DIAGNOSTIC ONLY -- no gates, no verdict.
```

**Example output:** `runs/xs_reversion_screen.log` — the cleanest
kill: "XS-REVERSION alt/BTC: n=2553 ev −0.442% t −3.19 hit 60%",
negative in 7/8 years (the lone positive is 2025: +0.052%),
positive in 1/9 assets, final line "DIAGNOSTIC ONLY — no gates, no
verdict". Disclosures and data as in the live briefing:
`runs/round11_academic_screens.log` (HOUSE DISCLOSURES a-e), the
premise check and shuffle-200 per factor/arm:
`runs/round12_xs_academic.log`.

---

## Template 3. ROBUSTNESS BATTERY — 7 tests with thresholds

Sensitivity diagnostics on an already-found effect: an effect that
lives at a single parameter point or in a single period is an
artifact. No gates; the verdict is assembled from measured numbers,
not from narrative.

```text
# ROBUSTNESS BATTERY: <candidate> — 7 tests, thresholds frozen
Status: ROBUSTNESS — no gates. Sensitivity probes only; the pinned
candidate remains the original specification.
Anchor [0]: reproduce the original split bit-for-bit BEFORE the
tests (the real t = <...> matched the source).

## [1] THRESHOLD SENSITIVITY: threshold {0.5x, 1x, 1.5x, 2x} of base
  <Effect + 95% CI at each threshold>.
  Rule: an effect change > 2x from a threshold shift = DEAD (the
  effect lives in a one-parameter sliver — overfit).

## [2] WINDOW SENSITIVITY: window {0.5x, 1x, 2x} of base
  <Does the effect survive at neighboring windows? Alive only at
  the base window = window fitting>.

## [3] SUBPERIOD THIRDS: sample thirds (or years)
  <Effect + CI in each third>. Alive (sign + CI) in ALL three =
  ROBUST. Alive in one third = a regime artifact.

## [4] LEAVE-ONE-OUT: drop one symbol/asset at a time
  <Effect + CI after each drop>. The effect must not rest on one
  symbol: dropping the "carrier" kills it = not a market effect.

## [5] PLACEBO NULL: 200 draws (seed pinned)
  <What is destroyed (the signal x outcome alignment), what remains
  (the outcomes themselves)>. The real value must land above the
  95th percentile of its placebo sample. Below the 95th = DEAD,
  whatever the t says.

## [6] DE-CLUSTERING: drop the top-1% observations
  <Effect + CI without the top-1% (extreme days/trades)>. Plus:
  clustering by dates/months — naive t is inflated, cluster-SE is
  mandatory.

## [7] SIGN STABILITY: a monotone quantile ladder
  <q1 < q2 < q3 < q4> (or a strict decline). A zigzag = the effect
  is carried by the single extreme bucket = noise.

## VERDICT ASSEMBLY (measured numbers, no narrative)
  [1] <threshold: 4 values, alive? True/False>
  [2] <windows: 3 values, alive? True/False>
  [3] <thirds: 3/3 or k/3>
  [4] <LOO: k/k>
  [5] <placebo: percentile <...> (threshold > 95)>
  [6] <top-1%: survived? cluster-t <...> vs naive t <...>>
  [7] <ladder: q1..q4, monotone? True/False>
  FINAL: ROBUST | NOT ROBUST — <the True/False rundown>.
```

**Example output:** `runs/poc_robustness.log` — a genuine pass:
sections [0]-[9], placebo percentile **0.0** in both nulls
(unrestricted and symmetric-side) = genuine, thirds 3/3 with CI,
LOO 4/4, the near < mid < far ladder monotone -> "FINAL: ROBUST".
A genuine fail: `runs/exkurt_walkforward.log` — an in-sample spread
of −0.024 (t −4.54) shrank in walk-forward to −0.0046 (t −0.57) =
**~5x shrink = overfit**, the q1..q4 ladder non-monotone, 2 kill
rules fired -> GRAVEYARD.

---

## Template 4. POST-MORTEM — 5 diagnostics after death

The autopsy of a dead family: it answers "why did it look alive",
not "how to resurrect it". No gates, no candidates, no resurrection —
the family stays closed; the cause of death is fixed in one line.

```text
POST-MORTEM: <family> -- diagnostics only (post-hoc, declared)
NO gates, NO candidates, NO resurrection. The family stays CLOSED.
inputs: <data file> x <sealed outcome>, n=<...>

== [0] BASELINE REPRODUCTION (frozen rubric, n=<...>) ==
  <Metrics file vs re-run; the incremental t AFTER controls
  <list> — where it died. Naive i.i.d. SE alongside for comparison.>

== [1] PLACEBO: 200 shuffled draws (seed, declared post-hoc) ==
  <Real vs its own placebo distribution: mean/sd, p2.5/p97.5>.
  The real value sits at percentile <...>; <k>/200 draws
  match-or-beat. Scale check: i.i.d. sigma vs placebo sd
  (ratio <...>).

== [2] QUANTILE LADDER (buckets by score, n per bucket) ==
  <EV + CI per bucket>.
  Monotonicity (rank vs EV): Spearman <...>, Kendall <...>.
  Spread top-Q1 = <...>; middle third flat? (yes/no)

== [3] REGIME CELLS: which cell carries the effect? ==
  <The hi-score vs lo-score delta within terciles of <factor>;
  empty cells flagged. A delta living in one low-n cell = a cell
  artifact, not an effect>. Plus declared post-hoc conjunctions —
  explicitly flagged as post-hoc.

== [4] DE-CLUSTERING ==
  <Drop the top-k |fwd| observations from the top bucket: did the
  effect survive?> Clustering by month: naive t <...> -> cluster
  t <...>.

== [5] ATTRIBUTION: what do the top decile share? What killed the
   bottom decile? ==
  <Top-decile composition by regime/year; the fired terms top/rest;
  worst-3 with numbers>.
  <The rising/falling month comparison — in which months the effect
  flips and which known factor explains it.>

POST-MORTEM end -- diagnostics complete; the family stays CLOSED.
Cause of death (one line): <e.g. "the raw effect is real
(percentile 100), but the incremental is dead (t +0.73 after
controls) = an MVRV-momentum echo">.
```

**Example output:** `runs/glm_v2_postmortem.log` — all five
diagnostics on GLM v2: placebo percentile **100** (the raw Spearman
+0.215 is real), the ladder monotone (Spearman +1.00), regime cells
(mvrv HIGH carries a +36.2 delta with n=3 in the low bucket),
de-clustering (monthly clustering cuts t 3.09 -> 2.38), attribution
(top-decile = calm IV x MVRV>75; longs in falling months return
−12.2%). The death is the incremental: t +0.73 after the
iv/rv/mvrv/skew controls = **MVRV-momentum echo**; "family remains
CLOSED".

---

## Template 5. AGENT BRIEFING — a subagent briefing (7 mandatory blocks)

The standard task header for a marathon subagent: without these seven
blocks the agent either invents its own nulls or brings back an
ungated narrative. Death is a valid briefing outcome, not an agent
failure.

```text
# AGENT BRIEFING: <task name> (v<n>)

## BLOCK 1. ENVIRONMENT
- Repo: <absolute path>. Python: <version/environment>.
- Writing tool: <Write/log>; result log: runs/<name>.log.
- cwd resets between calls — absolute paths only.

## BLOCK 2. CONTEXT + DISCLOSURES (house findings relevant to the task)
(a) <what is already closed nearby: the family/sister, the numbers,
    the log path>;
(b) <acknowledged caveats: survivorship, funding coverage,
    period concentration — mandatory in any read>;
(c) <overlaps with parallel agents: SIBLINGS IN PARALLEL —
    adjacent tests belong to sibling agents, overlaps here are
    REPORT-ONLY>.

## BLOCK 3. DATA
- Exact paths: <data/...>; schema: <columns, cadence, date grid>;
  inclusion criterion: <e.g. >= 913 daily bars>.
- Limitations: <short windows LOUDLY: file, coverage, dates>;
  <costs/funding from which date>.

## BLOCK 4. TASK (numbered steps with measurable outputs)
1. <Build X> -> output: <a table/number in the log>.
2. <Test Y> -> output: <a number>.
3. <Assemble the verdict> -> output: <a SUMMARY section>.
Every step ends in a measurable artifact (a number in the log),
not the word "explore".

## BLOCK 5. NULLS
- Reviewer (verbatim): <the reviewer's null, word for word>.
- House-mandatory: shuffle/placebo 200 draws in EVERY factor/arm;
  per-year tables mandatory; cluster-SE (dates/months) alongside
  naive.

## BLOCK 6. KILL (criteria pre-declared, with thresholds)
- (a) turnover kill-line: > 20 RT/yr/symbol = automatic death;
- (b) <a statistical threshold: t < 2 / CI contains 0 / percentile
  < 95> = family closed;
- The kill-line decides by itself; bypassing or revisiting the
  kill-line after data = a protocol violation.

## BLOCK 7. EXPECTED HONEST OUTCOME (what death looks like — verbatim)
- If dead, report verbatim: "<factor>: EV <...>% t <...>, negative
  in <k/N> years, positive in <k/N> assets — grave".
- Death is a valid outcome (the graveyard grows); rescuing it with
  another window/threshold after the data = a briefing failure,
  not a success.
```

**Example output:** `runs/round11_academic_screens.log` — a real
marathon briefing header: origin = external review, HOUSE
DISCLOSURES (a)-(e) with numbers (the stablecoin lesson, the
20 RT/yr kill-line) and DATA with "SHORT WINDOW DISCLOSED LOUDLY"
(123.5d). `runs/round12_xs_academic.log` — a second live specimen:
binding discipline (a)-(e) (the kill-line decides; shuffle/placebo
200 in every factor/arm; per-year tables mandatory) plus the line
"SIBLINGS IN PARALLEL: ... overlaps here are REPORT-ONLY".

---

## How the templates chain together

The screen (template 2) finds a marginal effect without gates ->
the prereg (template 1) freezes the test before the data -> the
battery (template 3) checks whether it is a one-parameter sliver ->
death is autopsied by the post-mortem (template 4) -> the next agent
receives the finding through the briefing (template 5). No template
ever hands out a "grail" — only a verdict from measured numbers: a
grave or a dated prereg.
