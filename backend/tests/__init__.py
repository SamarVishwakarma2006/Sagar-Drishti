"""Tests for Sagar Drishti Backend."""
import os
import sys
from pathlib import Path

# Ensure repository root and backend directory are always on sys.path
_TESTS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _TESTS_DIR.parent
_REPO_ROOT = _BACKEND_DIR.parent

for p in [str(_REPO_ROOT), str(_BACKEND_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)
