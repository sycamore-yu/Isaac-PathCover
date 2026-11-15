"""Keep test-only Numba callback signatures out of the flight process cache."""
import os
from pathlib import Path

os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".runtime" / "numba_tests")
)
