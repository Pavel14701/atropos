# Known issues

Honest inventory of everything the test suite does not prove, and why.
None of these are silent: each is pinned in the suite (skips are documented
in code, xfails are `strict=True`, so an unexpected change breaks the build)
and each names what would resolve it. Last verified: 890 passed /
4 skipped / 2 xfailed.

## 1. Four skipped tests — chaos-matrix deduplication

Status: skipped (documented, by design).

Cause: the chaos matrices in `tests/test_chaos.py` skip 4 corruption
combos (`overflow_1e308` for `block_bootstrap_ci`,
`stationary_bootstrap_ci`, `apply_proportional_costs`,
`cost_death_test`) because each is already covered by a dedicated
finding test with a stricter, pinned assertion. Running both would
duplicate coverage, not add it.

Unblocks: nothing — this is deliberate deduplication. If a combo's
dedicated test is ever removed, its matrix skip must go with it.

## 2. xfail — fast-only synthetic edge dies on 2/10 sweep seeds

Status: `xfail(strict=True)`, `tests/test_pipeline_sweep.py`
(`test_genuine_edge_survives_on_every_sweep_seed`, `@slow`).

Cause: the FAST edge class (raw AR(0.5) driver) is killed by the
robustness battery on sweep seeds 2 and 7. The battery is RIGHT: the
driver's memory (~2 periods) is shorter than the battery's smoothing
windows (5/10/15), so window-smoothing monotonically inverts the
signal-to-return alignment (a smoothed high reads as "the state was
high over the whole window" — a mean-reversion anti-signal). The class
is genuinely not smoothing-robust; softening the window-sensitivity rule
was rejected as p-hacking the killer.

Unblocks: already compensated — the SLOW edge class
(`kind="slow_genuine_edge"`), whose information window overlaps the
smoothing instead of washing past it, passes the battery on all 10 swept
seeds (`test_edge_class_contract_fast_vs_slow`). The xfail stays as the
honest record that the fast class is battery-mortal.

## 3. xfail — no honest synthetic construction for NO_OOS_EDGE

Status: `xfail(strict=True)`, `tests/test_ensemble_pipeline.py`
(`test_first_half_edge_dies_as_no_oos_edge`, `@slow`).

Cause: `NO_OOS_EDGE` is the one verdict that cannot be built with honest
synthetic series data in the current tree. Structural reason: stage 3
(the robustness battery, which attacks the lagged long/flat rule —
harder) is stricter than stage 4 (walk-forward, which tests the
contemporaneous sign rule — easier), so any world that clears stage 3
still shows contemporaneous out-of-sample power; concentrated-edge
worlds die earlier as `ARTIFACT` or `DEAD` instead. Kill-gate thresholds
were deliberately not touched to force the verdict.

Unblocks: a genuinely honest series world whose walk-forward dies while
everything before it passes — then promote it into the demo matrix. The
cross-sectional `CS_NO_EDGE` path already covers the "no OOS edge"
family on panels.

## 4. Follow-ups (not blocking, tracked)

Status: planned.

- **Mutation testing** — waiting for external users before the suite
  grows another layer.
- **Two-level performance budgets** — per-module and whole-suite
  budgets instead of the current single level.
- **Real-data golden tests** — need a frozen data snapshot; the current
  golden tests are synthetic-only.
- **Cross-version matrix CI** — conflicts with the single pinned
  `uv.lock`; needs a separate resolution strategy first.
