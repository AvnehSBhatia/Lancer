"""
speedtest_full_pipeline.py

Timing from cold start through 100 full pipeline runs.
Full pipeline: load embeddings + converters + model, then predict 100 times.
"""

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Start timer at very beginning
t_start = time.perf_counter()

# Imports (part of cold start)
from apollo.predict_from_strings import predict

# 100 full pipeline runs (first call loads everything; subsequent use cache)
actor = "United States of America"
receiver = "Iraq"
context = (
    "Year 2003. United States of America CINC score 0.1518, United Arab Emirates "
    "CINC score 0.0022. No active dispute between United States of America and "
    "United Arab Emirates. No alliance between United States of America and United Arab Emirates. "
    "Bilateral trade volume: 0.006. Diplomatic relations: unknown."
)
personality = "military analyst"

N = 100
for i in range(N):
    predict(actor=actor, receiver=receiver, context=context, personality=personality)

# End timer
t_end = time.perf_counter()
elapsed = t_end - t_start

print(f"Full pipeline: cold start + {N} runs")
print(f"  Total: {elapsed:.2f} s")
print(f"  Per run: {elapsed/N*1000:.2f} ms")
print(f"  Throughput: {N/elapsed:.0f} predictions/sec")
