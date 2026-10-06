import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts", ROOT / "viz"):
    sys.path.insert(0, str(p))
