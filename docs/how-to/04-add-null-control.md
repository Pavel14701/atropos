# How-to 04. Adding a new null control to stage 1

> **Estimated time:** 45–60 minutes (the control + tests + calibration)
>
> **You will need:** the atropos repository with an environment
> (`uv sync`), write access to `atropos/**` (unlike how-tos 01–03),
> clean `uv run ruff check .`, `uv run mypy` (strict), and
> `uv run pytest -q`. This is a contributor how-to: it changes library
> code.

## What a null control is and what its contract is

Stage 1 (`atropos.null_control_suite`) is six adversarial controls:
`shuffled_permutation`, `placebo_placebo`, `momentum_control`,
`known_features_control`, `subperiod_stability`, `calendar_control`.
Each returns a dict of exactly one shape (the schema, not code):

```text
{
    "stat": float,        # the control's statistic (t, correlation, share...)
    "p_value": float,     # the p-value of that statistic
    "verdict": "PASS" | "FAIL" | "INCONCLUSIVE",
    # ...control-specific keys (n, r2, HC0, ...)...
    "reason": str,        # optional: why it could not be computed
}
```

Semantics:

- **PASS** — the hypothesis survived this control. The threshold is
  strict: `PASS` requires `p_value < 0.05` strictly
  (`p_value == 0.05` is FAIL).
- **FAIL** — the control killed the hypothesis.
- **INCONCLUSIVE** — the control is **not computable**, most often a
  rank-deficient design (the signal is exactly collinear with a
  control). This is neither PASS nor FAIL: a pseudo-inverse would
  fabricate a plausible t-stat exactly in the cases the suite exists to
  catch. The current `reason` convention is `"collinear with controls"`.

The INCONCLUSIVE semantics also matter for an empty alignment:
`atropos._utils.align_inputs` returns empty arrays when the dropna
result is empty (it does not raise) — a public control must translate
this into an honest INCONCLUSIVE, not crash deep inside the regression
and not silently kill the hypothesis on zero observations.

## The steps to add one

1. **The control** — a function in `atropos/null_control_suite.py`,
   exported via `__all__` (alphabetical order).
2. **Registration in the orchestrator** — `atropos/ensemble_pipeline.py`,
   the `_stage_1_null_controls` function: the import, the call into the
   `results` dict, and three places where the old configuration is
   hard-coded:
   - the summary line hard-divides by 6:
     `f"{len(results) - len(failed)}/6 controls passed"` — update the
     divisor (or compute it from `len(results)`);
   - `_KILL_REASONS[1]` lists the control families in prose
     ("alignment / placebo / concentration / calendar") — extend it;
   - verdict refinement: if your control catches repackaging, its name
     must be added to the tuple
     `("momentum_control", "known_features_control")` in
     `repackaging = ...` — otherwise its FAIL yields `ARTIFACT`, not
     `REPACKAGING`.
3. **The demo asserts** — `run_all_examples()` at the end of
   `null_control_suite.py` is a layer-0 test: the planted edge prints
   PASS on every control and the demo tail **asserts** it, plus trash
   cases. Your control must participate in both blocks.
4. **Unit tests** — `tests/test_null_control_suite.py`: the planted
   case → PASS; a trash case → FAIL or INCONCLUSIVE; edge cases
   (threshold strictness, degenerate input).
5. **The calibration test (FPR under the null)** —
   `tests/test_calibration.py::test_null_controls_fail_fraction_under_pure_noise`:
   under pure noise the control must FAIL in roughly 95% of seeds (PASS
   requires p < 0.05), with the assert window `[0.85, 1.0]` over 150
   seeds. Add your control to the loop and to the `fails` dict. If your
   PASS gate works differently (like `subperiod_stability`, whose gate
   is sign consistency + a concentration cap), justify your own window
   in the test docstring and with a pinned measurement.

## Mini example: a control skeleton (actually run)

The example control `convexity_control`: kills an "edge" that is an
artifact of convexity — the regression `outcome ~ const + signal +
signal²`, the statistic is the t of the linear term after the quadratic.
The skeleton shows every required element of the contract: helpers from
`_utils`, INCONCLUSIVE on a degenerate input, the strict threshold.

```python
"""Skeleton example control: does a linear edge survive its own square?

outcome ~ const + signal + signal^2: if the "edge" is a convexity
artifact (big |moves| both pay), the linear term dies after the square
enters the model.  Follows the null_control_suite contract.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from atropos._utils import align_inputs, ols_hc0

_ALPHA = 0.05


def convexity_control(outcome: Any, signal: Any) -> dict[str, Any]:
    out, sig = align_inputs(outcome, signal)
    design = np.column_stack([np.ones(out.size), sig, sig**2])
    res = ols_hc0(out, design)
    t_lin = float(res["t"][1])
    p_lin = float(res["p"][1])
    degenerate = bool(res["rank_deficient"]) or not np.isfinite(t_lin)
    result: dict[str, Any] = {
        "stat": t_lin,
        "p_value": p_lin,
        "verdict": (
            "INCONCLUSIVE" if degenerate
            else ("PASS" if p_lin < _ALPHA else "FAIL")
        ),
        "n": int(res["n"]),
        "r2": float(res["r2"]),
        "HC0": True,
    }
    if degenerate:
        result["reason"] = "collinear with controls"
    return result
```

Its demo block (planted edge / trash / degenerate / calibration) —
example output of the run:

```text
linear edge   : PASS stat=11.40
convexity only: FAIL stat=-1.26
degenerate    : INCONCLUSIVE reason=collinear with controls
null FAIL fraction: 97/100 = 0.97
skeleton demo OK
```

What this shows:

- the linear edge passes (t=11.40 after the quadratic term);
- pure convexity dies (the linear term is not needed: t=-1.26 → FAIL);
- zero signal variance → a rank-deficient design → INCONCLUSIVE with a
  `reason`, not a fabricated t;
- under pure noise FAIL in 97/100 seeds — the ~0.95 calibration holds.

## Meta-traps (checked by CI)

- **No private copies of helpers.** Alignment and OLS come only from
  `atropos._utils` (`align_inputs`, `ols_hc0`). A `_align` copy in your
  module is behavioral drift between copies: the first alignment
  refactor will split your control from the rest. An empty alignment
  (see above) is translated into INCONCLUSIVE.
- **The demo asserts, it does not just print.** `run_all_examples()` is
  layer-0 verification: a regression breaks the demo instead of quietly
  printing a wrong verdict. Your control is added to both the prints and
  the asserts.
- **The docstring contract** (`tests/test_docstring_contract.py`): every
  function exported via `__all__` must carry a `>>>` example in its
  docstring; doctests are actually executed; `Raises: ValueError` must
  be backed by a real `raise`; `Args:` must match `inspect.signature`.
- **Names**: single-letter identifiers are banned everywhere except `_` —
  enforced by the AST test (`tests/test_conventions.py`); ruff does not
  catch this.
- **Style and types**: `ruff` (78 columns, pydocstyle, perflint, E741),
  `mypy` strict — zero warnings; tests of hypothesis code never touch
  the network and never depend on wall-clock time.
- **No lookahead lags**: lags are built with `shift` (the NaN head falls
  out in alignment), never `np.roll` — it wraps the tail into the head
  and leaks the future into the lag.

## Checklist

- [ ] The control returns `{stat, p_value, verdict, ...}` with
      INCONCLUSIVE on a non-computable input and
      `reason="collinear with controls"`.
- [ ] The threshold is strict: PASS ⇔ `p_value < 0.05`.
- [ ] Helpers from `atropos._utils`; no local copies.
- [ ] Registered in `_stage_1_null_controls` + the summary divisor + the
      `_KILL_REASONS[1]` text + (if it catches repackaging) the
      verdict-refinement tuple.
- [ ] The demo prints and **asserts** the new control; a trash case is
      included.
- [ ] Unit tests in `tests/test_null_control_suite.py`.
- [ ] FPR calibration in `tests/test_calibration.py`, ~0.95 under the
      null.
- [ ] `uv run ruff check .`, `uv run mypy`, `uv run pytest -q` — clean.

## What next

- How the stage-1 verdicts read in the pipeline report — how-to 02
  (`docs/how-to/02-interpret-verdicts.md`).
- Reference for the existing controls:
  `docs/api/atropos/null_control_suite.html`.
