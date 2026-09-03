#!/usr/bin/env python3
"""Debug resume matching logic."""
import json
import sys
sys.path.insert(0, '/home/cwq/carla-adversarial/scripts')

from collapse_configs import generate_experiment_matrix, condition_to_dirname

# 1. Show experiment matrix conditions
conds, info = generate_experiment_matrix(mode='h1_quick')
print("=== First 5 conditions ===")
for i, (gid, inten, duty) in enumerate(conds[:5]):
    print(f"  cond[{i}]: ({repr(gid)}, {repr(inten)}, {repr(duty)})")

# 2. Load actual progress file
progress_path = '/home/cwq/carla-adversarial/results/collapse_scan/h1_quick/scan_progress.json'
with open(progress_path, 'r') as f:
    progress = json.load(f)
print(f"\n=== Progress file ===")
print(f"  Raw: {progress}")

completed = set(tuple(c) for c in progress.get('completed', []))
print(f"  Completed set: {completed}")
for c in completed:
    print(f"    {c} -> types: ({type(c[0]).__name__}, {type(c[1]).__name__}, {type(c[2]).__name__})")

# 3. Check matching
print(f"\n=== Matching test ===")
for i, (gid, inten, duty) in enumerate(conds[:5]):
    cond_key = (gid, inten, duty)
    match = cond_key in completed
    print(f"  cond[{i}] {cond_key} -> in completed? {match}")

# 4. Check condition dirname matches
print(f"\n=== Dirname check ===")
cond_dirname = condition_to_dirname('single_front_main', 0.5, 1.0)
print(f"  condition_to_dirname('single_front_main', 0.5, 1.0) = {cond_dirname}")

import os
expected_dir = '/home/cwq/carla-adversarial/results/collapse_scan/h1_quick/single_front_main/' + cond_dirname
print(f"  Expected dir: {expected_dir}")
print(f"  Exists? {os.path.exists(expected_dir)}")
