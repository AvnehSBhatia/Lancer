"""
Inference-only runner: invokes predict_from_strings.py once per vault personality (100).

No training subprocesses. Each call is a fresh Python process:
    python predict_from_strings.py <actor> <receiver> <context> <personality>

See PIPELINE_INFERENCE_AUDIT.md for the full dependency graph and where stages
may be skipped relative to main (23).pdf.

Usage (from repo root):
    python run_full_pipeline.py
    python run_full_pipeline.py "United States of America" "Iraq" "Year 2003. No dispute."
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREDICT_SCRIPT = ROOT / "predict_from_strings.py"

DEFAULT_ACTOR = "United States of America"
DEFAULT_RECEIVER = "Iraq"
DEFAULT_CONTEXT = (
    "Year 2003. United States of America CINC score 0.1518, "
    "United Arab Emirates CINC score 0.0022. No active dispute."
)


def _run_predict_subprocess(actor: str, receiver: str, context: str, personality: str) -> int:
    return subprocess.run(
        [
            sys.executable,
            str(PREDICT_SCRIPT),
            actor,
            receiver,
            context,
            personality,
        ],
        cwd=ROOT,
    ).returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run predict_from_strings.py once for each of the 100 vault personalities (no training).",
    )
    parser.add_argument("actor", nargs="?", default=DEFAULT_ACTOR, help="Actor string.")
    parser.add_argument("receiver", nargs="?", default=DEFAULT_RECEIVER, help="Receiver string.")
    parser.add_argument("context", nargs="?", default=DEFAULT_CONTEXT, help="Context string.")
    parser.add_argument(
        "--max-personalities",
        type=int,
        default=None,
        metavar="K",
        help="If set, only the first K vault entries (default: all 100).",
    )
    args = parser.parse_args()

    if not PREDICT_SCRIPT.is_file():
        raise SystemExit(f"Missing {PREDICT_SCRIPT}")

    from personality_bank import PERSONALITY_BANK

    names = list(PERSONALITY_BANK)
    if args.max_personalities is not None:
        names = names[: max(0, args.max_personalities)]

    print(f"Working directory: {ROOT}")
    print(f"Running {len(names)} subprocess calls to predict_from_strings.py")
    print("=" * 70)

    failures = 0
    for i, personality in enumerate(names):
        print()
        print(f"[{i + 1}/{len(names)}] personality: {personality[:72]}{'...' if len(personality) > 72 else ''}")
        print("-" * 70)
        rc = _run_predict_subprocess(args.actor, args.receiver, args.context, personality)
        if rc != 0:
            failures += 1
            print(f"  (exit code {rc})")

    print()
    print("=" * 70)
    if failures:
        raise SystemExit(f"Finished with {failures} failed run(s) out of {len(names)}.")
    print(f"All {len(names)} predict_from_strings.py runs completed successfully.")


if __name__ == "__main__":
    main()
