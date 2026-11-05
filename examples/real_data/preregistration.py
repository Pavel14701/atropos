"""Preregistration for the SPY 12-1 TSMOM ML kill run (frozen spec).

This module holds the ``params`` dict handed to
``atropos.generate_prereg``.  It was frozen BEFORE the out-of-sample
walk-forward verdicts were looked at: the construction, the nulls, the
kill criteria (mirroring the engine's four gates with their frozen
thresholds) and the power floor below are all fixed constants of this
file.  The runner prints the rendered document BEFORE any result --
you call the framework, the framework validates whatever hypothesis text
you froze; keeping that order is the caller's discipline, and
``tests/test_real_data_example.py`` pins it.

The spec hash is computed from these constants (canonical JSON of the
params minus the hash itself), so editing any frozen field changes the
hash printed in the document.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from atropos import generate_prereg
from atropos.prereg_generator import validate_prereg

FREEZE_DATE = "2026-11-05"
"""The date this spec was frozen; explicit so reruns render identical
text instead of silently moving the stamp to today."""

# NOTE: run_example.py --quick may reduce bootstrap_b and the
# model set for CI speed; FULL mode is the preregistered contract.
PREREG_PARAMS: dict[str, Any] = {
    "name": "spy_tsmom_12_1_ml",
    "hypothesis": (
        "Twelve-minus-one momentum on SPY -- long the first trading "
        "day of every month whose prior month-end 12-1 momentum is "
        "positive, held 21 trading days -- leaves an executable edge "
        "net of 2 bps round-trip costs, and classifiers on volatility "
        "and moving-average conditioning take a book that clears the "
        "cost line out-of-sample, beats the linear momentum battery "
        "incrementally, ranks features consistently across folds and "
        "holds a deflated Sharpe above 0.95."
    ),
    "construction": [
        "data: committed SPY daily snapshot data/SPY.csv, adjusted "
        "close, 2005-01-03..2024-12-31 (one adjustment vintage, "
        "sha256-pinned)",
        "signal: monthly resample (ME); mom_12_1 = P[m-1]/P[m-13] - 1 "
        "(12-month return skipping the newest month)",
        "entries: first trading day of month m+1 when mom_12_1 > 0 "
        "(long/flat); exit 21 trading days later at the close",
        "features at the entry close: vol_20d, ma_gap_50, ma_gap_200; "
        "raw mom_12_1 feeds the linear battery only",
        "exec = 21-day forward close-to-close return - 2 bps RT cost; "
        "target = (exec > 0)",
        "engine: run_ml_kill_protocol, models gbm/rf/logit, "
        "battery_controls=[mom_12_1], k_blocks=6 expanding "
        "walk-forward, horizon 744h, purge 24h, embargo 744h, "
        "threshold grid 0.30..0.70 step 0.05, bootstrap_b=2000",
        "gates: incremental_t=2.0, profit_bar=0.0 (EV of taken trades "
        "must be positive net of costs), importance_rho_min=0.5, "
        "dsr_min=0.95, n_min_events=50",
    ],
    "nulls": [
        "the conditioning adds nothing: a linear battery on raw "
        "mom_12_1 trades an equal-or-better book than the classifiers",
        "the edge is repackaged equity drift: taken-trade EV does not "
        "exceed the unconditional drift of long months",
        "per-fold feature rankings are noise: importance stability at "
        "or below chance",
    ],
    "kill_criteria": [
        {
            "criterion": (
                "paired bootstrap t of taken-trade EV vs the linear "
                "mom_12_1 battery"
            ),
            "comparison": "<",
            "threshold": 2.0,
        },
        "OOS EV of taken trades < 0.0000 net of 2 bps RT costs",
        "median pairwise Spearman of per-fold importances < 0.5",
        "deflated Sharpe of the daily EV stream < 0.95",
    ],
    "power_calc": {
        "N_required": 50,
        "alpha": 0.05,
        "power": 0.80,
        "note": (
            "monthly book: 50 taken monthly events are the floor at "
            "which the gates resolve; they detect monthly effects of "
            "roughly 2 x 4.2% / sqrt(50) ~ 1.2% -- smaller true edges "
            "are killed, not confirmed"
        ),
    },
    "hash": "",
}


def spec_hash() -> str:
    """First 8 hex chars of sha256 over the frozen params (sans hash)."""
    frozen = {k: v for k, v in PREREG_PARAMS.items() if k != "hash"}
    canonical = json.dumps(frozen, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def build_prereg_text() -> str:
    """Validate the frozen params and render the prereg document.

    Returns:
        The markdown preregistration as produced by
        ``atropos.generate_prereg``, stamped with :data:`FREEZE_DATE`.

    Raises:
        ValueError: if the frozen params ever fail validation (the
            failure must be loud, never a softened spec).
    """
    params = {**PREREG_PARAMS, "hash": spec_hash()}
    problems = validate_prereg(params)
    if problems:
        detail = "\n  - ".join(problems)
        raise ValueError(f"frozen prereg params are invalid:\n  - {detail}")
    return generate_prereg(params, date=FREEZE_DATE)
