"""
scripts/check_models.py
=======================
Pre-flight "Models Reached" check - prints which model files load and
whether each was trained with the library versions installed here.
Does NOT start the TCP server.

    python grim_hybrid/scripts/check_models.py [--models-dir DIR]

Exit code: 0 = every model reached and versions match,
           1 = something missing / failed / mismatched.
Run it after any pip install or retrain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from strategy_engine.utils import load_config                 # noqa: E402
from server.silver_bullet_server import ModelRegistry          # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--models-dir", default=None)
    args = ap.parse_args(argv)

    reg = ModelRegistry(load_config(), models_dir=args.models_dir, verbose=True)
    ok = all(r["status"] == "REACHED" for r in reg.status) and reg.all_versions_match
    print("RESULT:", "ALL MODELS REACHED, VERSIONS MATCH" if ok else "ATTENTION NEEDED (see table above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
