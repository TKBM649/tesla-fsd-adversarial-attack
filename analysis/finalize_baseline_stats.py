#!/usr/bin/env python3
"""One-shot recompute of baseline_stats.json with robust frame-level attack
thresholds (supersedes the route-mean mu+-3sigma calibration).

Same algorithm as collect_baseline.compute_robust_thresholds(), kept
self-contained so it runs without the BEVFormer/CARLA dependency stack.
Existing route files predate the post_reset marker; their chimera frames
record cos < 0.99 and are excluded from calibration by the same core filter.
"""
import glob
import json
import os
import re

import numpy as np

RESULTS = os.path.expanduser('~/carla-adversarial/results')
PERSISTENCE = 15  # zero false alarms on the 2000-frame baseline (sweep-verified)

files = sorted(glob.glob(os.path.join(RESULTS, 'baseline_route_*.json')),
               key=lambda p: int(re.search(r'route_(\d+)', p).group(1)))
print(f'Loading {len(files)} route files')

per_route, all_route_frames = [], []
for path in files:
    with open(path) as f:
        data = json.load(f)
    data['summary']['route_id'] = data['route']
    per_route.append(data['summary'])
    all_route_frames.append(data['frames'])

# ---- route-level overall (same formulas as collect_baseline) ----
r = per_route
overall = {
    'num_routes': len(r),
    'total_frames': sum(x['num_frames'] for x in r),
    'score_thr': r[0].get('score_thr', 0.1),
    'det_count_mean': float(np.mean([x['det_count_mean'] for x in r])),
    'det_count_std': float(np.std([x['det_count_mean'] for x in r])),
    'det_score_mean': float(np.mean([x['det_score_mean'] for x in r])),
    'det_score_std': float(np.std([x['det_score_mean'] for x in r])),
    'det_score_max_mean': float(np.mean([x['det_score_max_mean'] for x in r])),
    'bev_cosine_mean': float(np.mean([x['bev_cosine_mean'] for x in r])),
    'bev_cosine_std': float(np.std([x['bev_cosine_mean'] for x in r])),
    'bev_l2_mean': float(np.mean([x['bev_l2_mean'] for x in r])),
    'bev_l2_std': float(np.std([x['bev_l2_mean'] for x in r])),
    'speed_mean': float(np.mean([x['speed_mean'] for x in r])),
    'legacy_threshold_bev_cosine': float(
        np.mean([x['bev_cosine_mean'] for x in r])
        - 3 * np.std([x['bev_cosine_mean'] for x in r])),
    'legacy_threshold_bev_l2': float(
        np.mean([x['bev_l2_mean'] for x in r])
        + 3 * np.std([x['bev_l2_mean'] for x in r])),
}

# ---- robust thresholds (identical algorithm to the collector) ----
cos_vals, l2_vals = [], []
jump_count = 0
for frames in all_route_frames:
    for f in frames:
        c = f['bev_self_sim']
        if c <= 0 or f.get('post_reset', False):
            continue
        if c < 0.99:
            jump_count += 1
            continue
        cos_vals.append(c)
        l2_vals.append(f['bev_l2_dist'])

cos_arr, l2_arr = np.asarray(cos_vals), np.asarray(l2_vals)


def _robust(x, sign):
    med = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - med)))
    if sign < 0:
        return max(med - 3 * mad, float(np.percentile(x, 0.5)))
    return min(med + 3 * mad, float(np.percentile(x, 99.5)))


cos_thr = _robust(cos_arr, -1)
l2_thr = _robust(l2_arr, +1)

fp_single = fp_persist = n_valid = 0
persist_events = []
for frames, rid in zip(all_route_frames, [x['route_id'] for x in per_route]):
    alarms = []
    for f in frames:
        c = f['bev_self_sim']
        if c <= 0 or f.get('post_reset', False):
            continue
        n_valid += 1
        alarms.append((c < cos_thr and f['bev_l2_dist'] > l2_thr, f['frame_idx']))
    fp_single += sum(a for a, _ in alarms)
    run = []
    for a, fi in alarms:
        if a:
            run.append(fi)
            if len(run) == PERSISTENCE:
                fp_persist += 1
                persist_events.append((rid, run[0]))
                run = []
        else:
            run = []

print(f'\nCore calibration set: {cos_arr.size} frames '
      f'({jump_count} scene-jump frames excluded)')
print(f'cos  median={np.median(cos_arr):.6f}  MAD-sigma={1.4826*np.median(np.abs(cos_arr-np.median(cos_arr))):.7f}')
print(f'L2   median={np.median(l2_arr):.6f}  MAD-sigma={1.4826*np.median(np.abs(l2_arr-np.median(l2_arr))):.7f}')
print(f'\nNEW attack thresholds:')
print(f'  BEV cosine < {cos_thr:.6f}  (legacy was {overall["legacy_threshold_bev_cosine"]:.6f})')
print(f'  BEV L2     > {l2_thr:.6f}  (legacy was {overall["legacy_threshold_bev_l2"]:.6f})')
print(f'  single-frame AND-rule FPs : {fp_single} ({100*fp_single/n_valid:.2f}%)')
print(f'  {PERSISTENCE}-frame persistence FPs: {fp_persist} {persist_events}')

# Persistence-k sweep: how long must an anomaly last before we call it an
# attack? Natural scene transients (route_1 start ramp, route_5 tail) are
# 5-25 frames; injected attacks persist for the whole attack window (100+).
print('\n  Persistence sweep (FP events):')
for k in (3, 5, 8, 10, 15, 20):
    cnt = 0
    for frames in all_route_frames:
        run = 0
        for f in frames:
            c = f['bev_self_sim']
            if c <= 0 or f.get('post_reset', False):
                continue
            run = run + 1 if (c < cos_thr and f['bev_l2_dist'] > l2_thr) else 0
            if run == k:
                cnt += 1
                run = 0
    print(f'    k={k:2d} frames ({k*0.05:4.1f}s): {cnt} false events')

overall['attack_threshold_bev_cosine'] = cos_thr
overall['attack_threshold_bev_l2'] = l2_thr
overall['threshold_rule'] = (f'BEV cosine < {cos_thr:.6f} AND L2 > {l2_thr:.6f} '
                             f'on {PERSISTENCE} consecutive frames '
                             '(robust median+-3*MAD on normal-driving core)')
overall['threshold_false_positives_single_frame'] = int(fp_single)
overall['threshold_false_positives_persisted'] = int(fp_persist)
overall['threshold_false_positive_events'] = \
    [[rid, fi] for rid, fi in persist_events]
overall['scene_jump_frames_excluded'] = int(jump_count)

out_path = os.path.join(RESULTS, 'baseline_stats.json')
with open(out_path, 'w') as f:
    json.dump({'overall': overall, 'per_route': per_route}, f, indent=2)
print(f'\nSaved: {out_path}')
