# CASES AND FLAGS — red flags, case studies, the agent checklist

> Document v1.0 (2026-11-02). Part of the Atropos framework ("a system
> for killing crap hypotheses").
> Companions: FRAMEWORK.md (philosophy + the 7-stage pipeline + the
> 6 laws, §6), TEMPLATES.md (the prereg/screen/battery/post-mortem/
> briefing templates).
> The framework's red thread — §9 — is quoted at the end of section 3.

**Source.** Donor corpus: a closed internal repository (not included
in this distribution) — 16 rounds of the S10 review marathon, the
theory graveyard v1.1.1 (123 entries = 129 deaths + 1 retraction),
its preregs `PREREG_*.md` (including the OFI-shadow, the reference
7-gate design), its post-mortem and screen logs under the `runs/`
directory, and its deploy spec §8. All numbers are taken from those
files, not from memory. File names in this document are catalog
references into the donor corpus, not files in your workspace: the
`hypotheses/` layout (graveyard + registry) is scaffolded with
`atropos init` (files are generated, never overwritten).

**How to use.** Before launching any subagent: (1) run the hypothesis
through the 8 flags of section 1 — every flag that fires is a
mandatory line in the agent briefing; (2) check against the case
studies of section 2 — if the candidate resembles one of the five
deaths, the test is designed to kill exactly that mechanism;
(3) walk the checklist of section 3 — all 7 items, no exceptions.

---

## SECTION 1. RED FLAGS — 8 patterns that predict death BEFORE the test

Each flag follows the same scheme: the mechanism / a real example from
the graveyard / a one-line rule. A flag is not a verdict; it is a
prior. Only a single run against a frozen prereg produces a verdict.
But the 8 flags below fired dozens of times in the marathon: ignoring
them means paying for the candidate's death with a full run instead
of a single query.

### FLAG 1. Correlation with momentum > 0.5 → trend repackaging

**Mechanism.** The crypto cross-section is rank-one (PC1 = 66.8% of
variance). Any "new" feature strongly correlated with trend is, with
high probability, momentum retold in another language — ML,
nonlinearity and stacking add no information, because there was none
to add.

**Examples.** CTREND (an ML aggregation of multi-horizon trends,
Fieberg 2025 JFQA): OOS R² −0.00001, IC/d −0.007 (t −0.59), delta over
naive 30d momentum — zero, net −11.4%/yr against −1.2% for the naive
book, placebo 200 shuffled-feature p 0.74 — the real book lands BELOW
the placebo mean (`runs/round13_ctrend.log`, grave 100). GLM v2: the
mvrv_hi+10 rubric term mechanically pinned the score to the top on
60/63 mvrv-high days — the "prediction" is an MVRV-momentum echo,
with zero score variation inside the regime cell
(`runs/glm_v2_postmortem.log`, grave 118). A cautionary counterexample:
ETF flows had corr(flow_z, mom7) = 0.540 — the flag fired, but the
decisive incremental test gave t +2.96 AFTER mom7/mom30 → a
conditional YES (`runs/round16_etf_flows.log`). The flag sets the
prior; only the incremental kills.

**Rule.** |ρ(new feature, momentum)| > 0.5 → the default verdict is
"repackaging"; the candidate is alive only if the incremental t > 2
after momentum + known features.

### FLAG 2. Description without a predict test → describe-vs-predict (10/10)

**Mechanism.** Regime statistics computed on the past describe the
past. They are either degenerate (saturated, persistent) or follow
their own autocorrelation — and they do not predict the next period.
The family is closed in the graveyard at 10/10: SMA-tilt, vol-target,
Hurst, Shannon entropy, ADX, grid-Hurst, RV-regime switch, dispersion
conditional, graph density, Bayesian regimes.

**Examples.** Hurst: the H of shuffled series 0.627 is HIGHER than the
real 0.618 — the estimator measures its own bias (grave 90). Entropy:
median 0.994 out of max 1.0 — "chaos" is always on (92). SMA-tilt:
NW5 4.47 against base 4.89 — a degradation (58). Vol-target: CAGR
+299% → +16.5% — it eats an order of magnitude of return (57).
RV-regime: raw t 1.08, incremental t 0.78 (98). Graph density: lag-1
autocorr +0.988, ~15 independent days out of 2434 (123,
PERSISTENCE-DEATH). Bayesian regimes — the family's 7th member,
CONFIRMED describe-only (the pairs stack,
`runs/pairs_stack_components.log`).

**Rule.** No state statistic enters the system without a predictive
test of the next period against synthetic bands (GBM + shuffled) —
"describes" ≠ "predicts".

### FLAG 3. A threshold/window chosen after the data → in-sample artifact

**Mechanism.** A threshold taken from the same data (the full sample,
a percentile cut over the whole series) is fitted to noise.
Transplanted into walk-forward it shrinks the effect severalfold and
breaks the ladder's monotonicity.

**Example.** Ex-Kurt on the spread — "the only marginal predictor of
round-15": discovery spread −0.024, CI [−0.047, −0.0008] excludes 0,
consistency 3/4 years — but the p90/p10 thresholds were taken from
the full data (honestly disclosed). Walk-forward (rolling 180d train
→ frozen p25/p50/p75 → 30d test, 27 blocks, 6440 pair-days): spread
−0.00457 (t −0.57) — **a ~5× shrink**; the q1..q4 ladder
−0.003/−0.008/−0.003/−0.009 — a non-monotone zigzag, strictly
decreasing blocks 2/27 = a tail artifact. GRAVEYARD
(`runs/exkurt_walkforward.log`, `runs/pairs_stack_components.log`).

**Rule.** Zero parameters after the data: thresholds/windows are
frozen before the run; any candidate with in-sample thresholds goes
into a walk-forward prereg, not onto the shelf — it is never cited
as a "finding".

### FLAG 4. The effect lives in one year/regime → regime artifact

**Mechanism.** Positive EV concentrated in a single year/regime means
either a fit to a single episode or an edge that has already died
(the decay recognized only after the window closed).

**Examples.** BAC weekly: +39.4%/yr, beats 99% of shuffles — but
2021-22 carries **76%** of the positive EV and 2025-26 is −11%/yr;
it survived only as conditional with an acknowledged decay
(`runs/round12_xs_academic.log`, survivors section). Amihud
vol-of-illiq: thirds +83 / +55 / **−38**%/yr — FRAGILE-DEATH (103).
Friday-13th: z +2.70, positive in 8/9 years, but the whole effect =
the single day 2021-08-13 (84). C2 extreme flow: +9.9%/+4.7% in
2021-22, then zero — DECAY-DEATH (34). Gas → vol: the full-sample
1.45× is an artifact of the 2021-22 NFT mania, post-Merge ratio
1.01× (88).

**Rule.** A per-year split is mandatory in every screen; concentration
> ~70% in one year/regime = an artifact prior; the maximum status is
conditional with a disclosed decay, never a candidate.

### FLAG 5. A "known fact from equities" without a sign check → risk-premium inversion

**Mechanism.** In crypto the sign of the risk premium is negative:
investors PAY, in expected return, for lottery-likeness and high
down-risk. The academic direction (long calm / short lottery-like)
inverts — the book lands at the bottom of the null distribution.
Porting a fact from stocks without the cheap sign premise-check = a
guaranteed INVERTED-DEATH.

**Examples.** MAX effect (Bali-Cakici-Whitelaw 2011): LS net
**−47.2%**/yr weekly — beats 5.0% of shuffled factors (the bottom of
the distribution), Sharpe −0.56, DD 442%; the LONG-only leg (calm,
inverted) +38.1% (`runs/round12_xs_academic.log`, grave 101).
Downside beta (Ang-Chen-Xing 2006): LS −38.2%/yr weekly (beats
10.0%), reverse tail 90.5/94.5%; shorting low-down-beta
−85.2/−105.9%/yr; LONG-only +47.0/+58.9% (102). All the repackaging
controls were passed honestly (ρ(MAX, β) +0.502 < 0.7; overlap with
reversal max|ρ| 0.07; turnover 0.44-1.47 RT/yr) — the controls did
not die, the sign did.

**Rule.** Every port-in hypothesis from equities starts with a sign
premise-check on raw data (zero parameters, one query) — before any
book building, batteries and placebos.

### FLAG 6. EV amplitude < the cost line (10-20bp RT) → COST-DEATH a priori

**Mechanism.** The house has a cost line: ~10bp taker RT, a
conservative 20bp RT line. If the gross amplitude of the effect is
below the line, no mechanics, filtering or sizing can save it — the
arithmetic is shared by the whole family. Building infrastructure
before measuring the amplitude = paying for the candidate's death
with maximum resources.

**Examples.** Basis arb ("10-20%/yr, nearly risk-free"): max |basis|
for BTC over 123.5 days of 1m = **8.9bp < a single 20bp RT**;
P(|b|>10bp) = 0 minutes in 4 months; ETH — 3 tradable minutes
(grave 79). Stat-arb pairs: cost 0.20% RT against a pooled t −3.7
(an anti-edge) (55). Grid family: 4/5 arms auto-dead at the 20bp
line before any statistics (93-97). Weekend fade: max premium
+3.25pp < the 5pp kill bar (81). Hour-of-day: best hour +0.04% <
10bp RT (65). BTC/ETH IV spread: gross $9-19 against an RT haircut
of $296 (35).

**Rule.** The amplitude is measured with a single bounded probe
before any infrastructure; gross < the cost line → the family closes
without a prereg (the cause of death is recorded, no track is spent).

### FLAG 7. A yearly prior in scoring → calendar contamination

**Mechanism.** Any scoring/rubric term encoding "which year/regime we
are in now" (an explicit year prior, a multi-year lookback trend, the
author's knowledge of yearly regimes) makes the in-sample correlation
with the actual trajectory trivially high. OOS on fresh days —
collapse.

**Examples.** GLM v1: the "yearly regime" rubric term — pilot SPOT30
+0.53 (t +5.50), dir hit positive in every year (64/75/73/76/77/58);
on 189 fresh days **+0.053 (t +0.73)**, hit 52% against a 56% base.
Diagnosis: the selection of 80 stratified days + the author's
knowledge of yearly regimes (in-sample weights) — grave 118, strike
S2. CTREND: the aggregation of multi-horizon (up to yearly) trends —
the same calendar mechanism on top of repackaging (100). The
calendar class as a whole: Friday-13, hour-of-day, session windows,
CNY/Halloween — all dead (65, 83, 84).

**Rule.** No year/regime terms in scoring or the rubric; year dummies
in a regression are a necessary but insufficient control; replication
on fresh days the scoring's author never saw is mandatory.

### FLAG 8. Controlled turnover > 20 RT/yr → COST-DEATH by construction

**Mechanism.** Turnover is a design choice, not a market property.
The house line is 20 RT/yr: a candidate that needs more turnover to
live dies of costs before the statistics get a word in. The "reduce
the turnover" redesign often reveals that the edge was the refresh
frequency itself.

**Examples.** XS funding-carry hysteresis: two pre-declared kills —
(1) turnover did not drop below 20 (mean 20.7 / max 28.4 RT/yr; the
original weekly arm 24.8 / 33.4) and (2) Sharpe collapsed under
forced holding (NW5 −0.05, EV −1.88%/yr against a baseline of
+29.86) — **BOTH fired**; "the edge = the weekly refresh of
extremes, not a holdable signal"
(`runs/funding_carry_hysteresis.log`, grave 115). Basis term
structure: all 12 cells killed by turnover of 171-1105
RT/yr/symbol (104). Funding grid: up to 1.8M RT/yr (94).

**Rule.** Turnover is designed before the run; a forecast above
20 RT/yr = construction closed; if a hysteresis/filter kills the
Sharpe — there was no signal, only an overtrading frequency.

---

## SECTION 2. CASE STUDIES — the 5 most instructive deaths

Five deaths out of 130+, chosen by a single criterion: the lesson
generalizes to a whole class of future hypotheses. Each comes with a
mechanism, numbers and a tombstone.

### CASE 1. CVD → retest_delay: a beautiful raw signal, empty after controls (INCREMENTAL-DEATH)

**Hypothesis.** CVD divergence (aggressive flow against the price
move) confirms entries on the C'' leg — "the crowd sells, the smart
money buys". Round-6's first candidate, and it looked like a real
edge.

**What killed it.** Not the battery — the battery was ROBUST. The
incremental test against known features killed it. The raw picture:
long-side confirmation +0.217R [+0.083, +0.350], hit_tp +5.3pp,
positive in 5/6 years; full battery: no threshold cliff, invariance
to the normalizer, all 3 subperiods positive, LOO ±0.01R, placebo
percentile 100.0 (0/200 draws, p ≤ 0.005). Then the incremental on
top of OB controls: slope_z t +1.05 (needs > 2), ΔR² +0.0004; the
entire lift is absorbed by zone_width_atr (t −9.6) and retest_delay
(t −9.2). By symbol: BTC −0.41 / ETH +0.78 / SOL +0.94 — the
"signal" is a noisy proxy for retest_delay.

**Mechanism.** CVD is correlated with OB zone geometry: where the
zone is narrow and the retest fast, divergences "happen" more often —
and EV is systematically better there too. One regression pass
exposed this; without it, three years of observations would have
been wasted.

**Lesson (generalizes).** The robustness battery and placebo check
that the effect is not noise; only controls check that it is not
SOMEONE ELSE'S. From that day on, incremental controls against known
features are the mandatory START of every screen, not the final
check. The same mechanics repeated at least three times: trade-size
moments (skew −2.48 raw → −1.46 after controls), BTC.D (t 3.52 raw →
0.16 after momentum-leg controls), L/S ratio (hi_d t +1.86 after OB
controls). A separate sub-lesson: a raw regression t of ±10-14 = the
|BTC-move| confound, not a signal.

**Tombstone.** Graveyard section 5, "wave 3(b)"; the survivor
CVD-forward with a lowered prior; `runs/cvd_ob_independence.log`,
`runs/vpin_cvd_screen.log`.

### CASE 2. MAX effect: the risk premium in crypto is negative (INVERTED-DEATH)

**Hypothesis.** An academic port: long calm / short lottery-like
(low/high MAX30, Bali-Cakici-Whitelaw 2011). On equities the factor
has worked for decades — the sign was treated as given.

**What killed it.** The sign. Repackaging controls passed (ρ(MAX, β)
+0.502 < 0.7; overlap with the reversal agent max|ρ| 0.07; turnover
0.44-1.47 RT/yr — the kill-line was not even needed). A 200-draw
shuffled-factor placebo per arm. Result: LS net **−47.2%/yr** weekly
— at the 5.0th percentile of the shuffled distribution (the bottom),
Sharpe −0.56, DD 442%. The lottery coins underperformed. The monthly
arm +9.9% (beats 71.1% — below the 95% bar) → NO in both arms.
Sibling downside beta (Ang-Chen-Xing 2006): LS −38.2%/yr weekly,
reverse tail 90.5/94.5%; the inverted long legs of both books
+38..+59%/yr.

**Mechanism.** In the crypto cross-section risk is paid for: coins
with lottery tails and high down-side beta are a leveraged bet on
the market that overpays in bull regimes. The "calm" ones are the
underearning tail. The academic sign = a bet against the structure
of this market, and the data says so with a single number.

**Lesson (generalizes).** A known fact from equities ports over as a
QUESTION, not as a direction. A sign premise-check on raw data costs
one query and kills the inverted book before any of the work.
Flag 5 is the direct consequence of this case; the inversion is
priced in up front for all future factor screens. Note the
protocol's honesty: the death was recorded BEFORE the prereg, on a
battery-pre-mortem — the prereg slot's resources were never spent.

**Tombstone.** Graves 101-102, TAXONOMY: INVERTED-DEATH was
introduced as a term by this case; `runs/round12_xs_academic.log`.

### CASE 3. Hurst: shuffled above real — the birth of the anti-bias protocol (estimator bias)

**Hypothesis.** Rolling Hurst distinguishes trend/chaos and predicts
the next day of the A-vs-C'' spread — a regime filter/switch.

**What killed it.** The synthetic null. The real H sits inside the
95% bands 92-96% of the time. The decisive number: **the H of
shuffled series 0.627 against the real 0.618** — series whose
structure is destroyed by construction produce a MEDIAN Hurst HIGHER
than the market's. The estimator measures its own bias, not the
market. The predictive test: every CI spans zero. The rest of the
hypothesis's scaffolding collapsed as well: the H<0.4 arm — 0 days
at W=400; H>0.6 fires 42-57% of the time ("a trend day" = almost
always); the unconditional A−C'' spread = −0.98 — "switch to A" =
fighting the system's base rate.

**Mechanism.** The R/S Hurst estimator is positively biased on finite
heavy-tailed samples; a shuffle preserves the marginal distribution
and destroys the memory — and if the shuffled-H does not drop below
the real value, there is no "memory" in the data for this estimator.
The GBM + shuffled bands showed this before any predictive test.

**Lesson (generalizes).** The estimator is part of the hypothesis: it
must be audited with a synthetic null exactly like the signal (the
same logic as wave 1: the D.13g simulator bug = part of the
hypothesis). The anti-bias protocol: before any claim that "the
market is trending/chaotic by metric X" — compute X on GBM and
shuffled series, build the 95% bands; the real value inside the
bands = the metric is uninformative, there is nothing left to test.
The "describe-but-don't-predict" family was finished off by this
same protocol to 10/10 (flag 2).

**Tombstone.** Grave 90 (OVERFIT-DEATH); siblings entropy 92
(saturation 0.994/1.0), grid-Hurst 95 (ΔSharpe −2.22);
`runs/hurst_entropy_screen.log`.

### CASE 4. Basis 8.9bp: measuring the amplitude BEFORE the infrastructure (COST-DEATH a priori)

**Hypothesis.** The reviewer's claim: "spot/perp basis dislocations
are 10-20%/yr, near-riskless arbitrage".

**What killed it.** A single bounded probe with a fresh fetch:
123.5 days of 1m spot AND perp for BTC/ETH/SOL (2026-07..11). BTC's
entire |basis| range — max **8.9bp**, less than the cost of one
20bp RT; P(|b|>10bp) = **0 minutes in 4 months**. ETH: 3 tradable
minutes in 123 days. SOL: the "dislocation" is a persistent discount
of −5..−13bp — naive harvest −25.6%/yr; the median time to revert to
zero from −10bp is 5.9 days against a 48h exit. Real funding is
3-7%/yr, not 10-20%. The claim died on both legs: no amplitude and
no yield.

**Mechanism.** Arbitrage dislocations on the majors no longer exist —
arbitrage flow eats them faster than REST infrastructure can even
see them. What reads as "nearly riskless yield" in the telling turns
out, on the data, to be a distribution whose tail to the right of
the cost line is empty.

**Lesson (generalizes).** The order of operations for any "known
claim": (1) a bounded amplitude probe — hours; (2) the comparison
with the cost line — one line; (3) only if gross > the line —
infrastructure and a prereg. Flag 6 is the direct consequence.
Precedents with the same mechanics: IV spread gross $9-19 vs a
$296 haircut (35), SKEW-1 gross +2.86pt < the 3.0pt floor (37),
funding_carry v1 +1.59bp/d vs costs of 10.5bp/d (22). Note: in these
cases the predictor was REAL and STRONG (SKEW Spearman +0.473 — the
strongest in the house) — economics kills even working predictors.

**Tombstone.** Grave 79 (NULL-DEATH on the premise; the cost class in
the appendices); `runs/basis_dislocation_screen.log`.

### CASE 5. GLM v1: a yearly prior in scoring — masked momentum, OOS collapse (calendar contamination)

**Hypothesis.** A GLM context score (rubric: IV floor / MVRV
continuation / **yearly regime** / skew sign / term inversion)
predicts forward 30d spot. Pilot: blind scoring of 80 stratified
days from 2021-2026 before unsealing the sealed outcomes.

**What killed it.** The N-extension to 189 fresh days. Pilot: SPOT30
+0.53 (t +5.50), dir hit 64/75/73/76/77/58% — positive in EVERY
year; the momentum control "survived" (t +4.41 → +4.56), a holdout
fit 2021-24 → 2025-26 gave +0.657 (t +4.18) — the holdout was
stronger than in-sample. The candidate was re-ranked to the top and
the rubric frozen by a prereg. Then 189 new days: **SPOT30 +0.053
(t +0.73)** against +0.53; dir hit 52% against a 56% base; the 2026
flag confirmed (hit 42%). Diagnosis: the pilot = the selection of 80
stratified days + the author's knowledge of yearly regimes — the
"yearly regime" rubric term carried the calendar, which happened to
match the actual trajectory in the pilot and carried nothing on the
fresh days.

**Mechanism.** A score containing a year prior mechanically "guesses"
each year's regime: in a bull year it goes long, in a bear year it
does not. Cross-validation and holdout splits inside the pilot
sample do not catch this, because the year prior is valid across the
entire pilot history. Only a forward replication on days seen by
neither the score nor its author catches it.

**Lesson (generalizes).** A yearly prior in scoring = masked momentum
(flag 7). The house model is no exception to the discipline: GLM
died by the same incremental test as CVD (v2: t +3.10 formally,
t +0.73 after iv_pct/rv_pct/mvrv/skew; post-mortem — placebo
percentile 100, but the whole effect sits in the single
mid-IV × MVRV>75 cell, where the score is an MVRV-momentum echo;
declustering t 3.09 → 2.70; attribution = a levered beta tilt:
falling months −10.4% against +4.8% for rising). Three strikes (the
C'' repackaging, the calendar, the v3 conjunction: t +0.34, with the
mvrv_only control beating the conjunction by −3.68pp) →
GLM-as-feature closed for good; prompt tours are banned:
prompt-fitting = mining. The post-mortem log
(`runs/glm_v2_postmortem.log`) is the reference autopsy:
reproduction → placebo → quintiles → regime cells → declustering →
attribution.

**Tombstone.** Grave 118 (3-STRIKE PERMANENT CLOSE, PARADIGM-DEATH);
`PREREG_GLM_SPOT_2026-11-02.md`; `runs/glm_v2_postmortem.log`.

---

## SECTION 3. AGENT CHECKLIST — 7 items BEFORE launching a subagent

The slot frame is `DEPLOY_SPEC_V1.md` §8: 4 agents (A: core, B:
gateway, C: API/Telegram/Docker, D: presentation) + a lead auditor
per scope. Every checklist item is a binary question; a "no"/"not
disclosed" answer = the agent does not launch.

**1. SLOTS: at most 4-5 parallel agents.**
Per DEPLOY_SPEC §8 — 4 agents + a lead; no more. The marathon lesson:
with more slots, accounting breaks — a "phantom counter" was recorded
(ctrend/onchain listed as "in flight" while already closed), and 10
of 13 pairs-stack components sat in PENDING because agent-2 never
returned. The rule: a slot = an agent with a one-shot scope and a
dated verdict; "in flight" without a date = hunt it down, do not
launch anything new.

**2. GRAVEYARD check: is the hypothesis already dead?**
Before briefing — find the closest family in your workspace's
graveyard (`hypotheses/graveyard/`; scaffold with `atropos init`).
Precedent: 3 of the external reviewer's 49 directions had already
been closed by the house's own tests; re-testing was avoided only by
checking the history. If a proposal differs from a buried one only
by a parameter/wrapper — that is a rediscovery: it requires the
closed history to be disclosed explicitly in a new dated prereg,
otherwise it does not launch.

**3. DATA: is the data in the repo, or DATA-BLOCKED?**
A bounded probe of the free sources BEFORE briefing. The graveyard
holds 7+ DATA-BLOCKED deaths: miner reserves (Glassnode 401,
~$800/mo), whale front-running (no free labeled series + sub-second
mechanics), event NLP (a calendar without timestamps / CPI), XS
momentum (a delisted-inclusive universe), DeFi liquidations (the
auction closes within the block), Max Pain (OI-by-strike is not
free), price/new-addr (true AdrNewCnt behind a paywall). Unlock
artifacts (the fetch-parquet sets) are reused — but "the data is
paywalled" = the direction does not open.

**4. NULLS: 3-6 controls; momentum + known features at minimum.**
The CVD lesson is built in: incremental controls are the START of a
screen, not the finale. The minimal set: momentum (30d or the
candidate's horizon) + the domain's family of known features (OB
controls zone_width/retest_delay for microstructure candidates;
|BTC-move| for conditioning; beta for XS books). Plus the external
reviewer's null verbatim if one is claimed, and a 200-draw placebo
for the book. Without controls a candidate = descriptive statistics
(flag 2).

**5. KILL: criteria verbatim, thresholds frozen before the data.**
The reference design is `PREREG_OFI_SHADOW_2026-11-02.md` §7: gates
in a table with thresholds (executable econ ≥ 40bp; half-life >
2×(poll + RTT); t > 2 one-sided AND ΔR² ≥ 0.003 over the OB features;
the sign stable across terciles), the kill-line verbatim: "t < 2
incremental OR placebo p > 0.05", the file's immutable hash before
the first snapshot. The one-shot rule: a single run against the
frozen prereg; an edit after the run = a new version, the old test
void. Zero parameters after the data (flag 3).

**6. CALENDAR: a yearly prior in scoring? → remove it or control for it.**
The GLM v1 lesson: the "yearly regime" term in the scoring rubric
masked the momentum and produced a false pilot of +0.53 →
OOS +0.053. Check whether the scoring/filtering carries terms
encoding year/regime (explicit dummies, a multi-year lookback, the
author's "knowledge"). If removal is impossible — year dummies in
the regression are mandatory + replication on fresh days outside the
pilot sample. The calendar class of anomalies (holidays, hours,
sessions) — buried; do not launch without a new mechanism.

**7. HONEST FRAME: COST-DEAD base? survivorship? power? — disclosed before the run.**
Three mandatory disclosures in the briefing: (a) the economic frame —
if gross sits right on the cost line, say exactly that (SKEW-1: the
3.0pt floor was declared BEFORE the run); (b) survivorship — if the
universe is living symbols only, the book is inflated (XS momentum:
long-only = a survivorship mirage, the short leg minus in every
configuration); (c) power — on a small window, "UNDECIDABLE"
statuses are allowed and honest (basis term structure: a 123.5d
window, power honesty declared; sniper z>2.5: the class is empty,
n=9 undecidable). Plus year concentration (flag 4) and funding
coverage of the shorts (BAB: 54.6% of files from 2023-09) — all of
it in the briefing, not in a post-mortem.

### THE RED THREAD (§9 of the framework)

The one line that holds the entire pipeline — from idea to verdict:

> **Check the graves → run the pipeline → one run → one verdict →
> graveyard or prereg.**

Zero parameters after the data. Zero calendars in scoring. Zero
"might come in handy". Every hypothesis enters through the checklist
above and exits into exactly one of two outcomes; there is no third.

---

*Related framework documents: FRAMEWORK.md (§6 — the 6 systematic
laws of the marathon; §9 — the red thread), TEMPLATES.md (the agent
briefing template includes checklist items 2/4/5/7 as mandatory
fields).*
*Number sources: `docs/THEORY_GRAVEYARD.md` v1.1.1,
`RESPONSE_S10_PROPOSAL_2026-11-01.md` (16 rounds),
`PREREG_OFI_SHADOW_2026-11-02.md`, `runs/glm_v2_postmortem.log`,
`runs/funding_carry_hysteresis.log`, `runs/exkurt_walkforward.log`,
`runs/pairs_stack_components.log`, `runs/round16_etf_flows.log`,
`DEPLOY_SPEC_V1.md`.*
