#!/usr/bin/env python3
"""End-to-end test of resume logic (no CARLA needed)."""
import json
import os
import sys
sys.path.insert(0, '/home/cwq/carla-adversarial/scripts')

from collapse_configs import generate_experiment_matrix, condition_to_dirname

mode = 'h1_quick'
output_dir = os.path.expanduser(f'~/carla-adversarial/results/collapse_scan/{mode}')
progress_path = os.path.join(output_dir, 'scan_progress.json')

# 1. Generate conditions (same as run_collapse_experiment.py)
conditions, mode_info = generate_experiment_matrix(mode=mode)
print(f"Total conditions: {len(conditions)}")

# 2. Load progress (same as scan_batch)
completed = set()
if os.path.exists(progress_path):
    with open(progress_path, 'r') as f:
        progress = json.load(f)
    completed = set(tuple(c) for c in progress.get('completed', []))
    print(f"Resume: {len(completed)} conditions already completed")
else:
    print(f"NO progress file found at: {progress_path}")

# 3. Simulate scan_batch loop
skip_count = 0
run_count = 0
for idx, (gid, intensity, duty) in enumerate(conditions):
    cond_key = (gid, intensity, duty)
    cond_dirname = condition_to_dirname(gid, intensity, duty)
    
    if cond_key in completed:
        skip_count += 1
        print(f"  [{idx+1}/{len(conditions)}] SKIP: {cond_dirname}")
    else:
        run_count += 1
        if run_count <= 3:
            print(f"  [{idx+1}/{len(conditions)}] RUN:  {cond_dirname}")

print(f"\nResult: SKIP={skip_count}, RUN={run_count}")
if skip_count > 0:
    print("[PASS] Resume correctly skips completed conditions")
else:
    print("[FAIL] Resume did NOT skip any conditions!")
