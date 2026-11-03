"""Shared pytest configuration: deterministic Hypothesis runs.

Project law: fixed seed, reproducible result. Hypothesis is pinned to
derandomized mode so property tests never flake in CI; a dev can opt
into random exploration locally with ATROPOS_HYPOTHESIS_RANDOM=1.
"""

import os

from hypothesis import settings

settings.register_profile(
    "derandomized", derandomize=True, max_examples=50, deadline=None
)
settings.register_profile(
    "exploratory", derandomize=False, max_examples=200, deadline=None
)

if os.environ.get("ATROPOS_HYPOTHESIS_RANDOM") == "1":
    settings.load_profile("exploratory")
else:
    settings.load_profile("derandomized")
