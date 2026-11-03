"""Puts this example's folder on the import path, so its tests import its package (as they do when run from here)
and `pytest` works from the repository's root too."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
