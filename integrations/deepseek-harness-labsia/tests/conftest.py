import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for plugin_dir in sorted((ROOT / "plugins").glob("deepseek-r1-*")):
    sys.path.insert(0, str(plugin_dir))
