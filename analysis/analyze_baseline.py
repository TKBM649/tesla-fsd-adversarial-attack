#!/usr/bin/env python3
"""Deep-dive analysis of collected baseline: pooled frame-level stats,
jump-event forensics, and false-positive rate of current attack thresholds."""
import glob
import json
import re

import numpy as np

RESULTS = '/home/cwq/carla-adversarial/results'
COS_THR = 0.9907708940590155   # attack_threshold_bev_cosine
L2_THR = 0.031195992590858285  # attack_threshold_bev_l2

files = sorted(glob.glob(f'{RESULTS}/baseline_route_*.json'),
               key=lambda p: int(re.search(r'route_(\d+)', p).group(1)))
print(f'Loading {len(files)} route files\n')

all_cos, all_l2, all_det, all_spd = [], [], [], []
jump_events = []
fp_cos = fp_l2 = fp_both = 0
total_valid = 0

per_route_rows = []
for path in files:
    data = json.load(open(path))
    rid = data['route']
    frames = data['frames']
    cos = np.array([f['bev_self_sim'] for f in frames])
    l2 = np.array([f['bev_l2_dist'] for f in frames])
    det = np.array([f['det_count'] for f in frames])
    spd = np.array([f['speed'] for f in frames])
    idx = np.array([f['frame_idx'] for f in frames])

    valid = cos > 0  # first frame / post-reset frames record 0.0
    all_cos.append(cos[valid]); all_l2.append(l2[valid])
    all_det.append(det); all_spd.append(spd)
    total_valid += int(valid.sum())

    # frame-level stats for this route
    c, l = cos[valid], l2[valid]
    per_route_rows.append((rid, len(frames), int(valid.sum()),
                           c.mean(), c.std(), c.min(),
                           l.mean(), l.std(), l.max(),
                           det.mean(), (det == 0).mean() * 100, spd.mean()))

    # jump events: cosine dropped hard (below 0.99 = far outside any normal
    # driving variation at 20 Hz)
    jumps = np.where(cos < 0.99)[0]
    for j in jumps:
        jump_events.append((rid, int(idx[j]), cos[j], l2[j], spd[j], det[j]))

    fp_cos += int((cos[valid] < COS_THR).sum())
    fp_l2 += int((l2[valid] > L2_THR).sum())
    fp_both += int(((cos[valid] < COS_THR) & (l2[valid] > L2_THR)).sum())

c = np.concatenate(all_cos); l = np.concatenate(all_l2)
d = np.concatenate(all_det); s = np.concatenate(all_spd)

print('=== POOLED FRAME-LEVEL STATS (n=%d valid transitions) ===' % len(c))
print(f'BEV cosine : mean={c.mean():.6f}  std={c.std():.6f}  '
      f'min={c.min():.6f}  p01={np.percentile(c,1):.6f}  p05={np.percentile(c,5):.6f}')
print(f'BEV L2     : mean={l.mean():.6f}  std={l.std():.6f}  '
      f'max={l.max():.6f}  p95={np.percentile(l,95):.6f}  p99={np.percentile(l,99):.6f}')
print(f'Det count  : mean={d.mean():.2f}  zero-frames={100*(d==0).mean():.1f}%  '
      f'p90={np.percentile(d,90):.0f}  max={d.max():.0f}')
print(f'Speed m/s  : mean={s.mean():.2f}  '
      f'standing(<0.5m/s) frames={100*(s<0.5).mean():.1f}%')

print('\n=== FRAME-LEVEL 3-sigma (pooled, non-route-averaged) ===')
print(f'cosine thr (mu-3sigma): {c.mean()-3*c.std():.6f}')
print(f'L2 thr    (mu+3sigma): {l.mean()+3*l.std():.6f}')
print(f'L2 thr robust (median+3*MAD-sigma): '
      f'{np.median(l)+3*1.4826*np.median(np.abs(l-np.median(l))):.6f}')

print('\n=== CURRENT THRESHOLDS FALSE-POSITIVE RATE (normal driving only) ===')
print(f'cos < {COS_THR:.6f} alone : {fp_cos} frames ({100*fp_cos/len(c):.2f}%)')
print(f'L2  > {L2_THR:.6f} alone : {fp_l2} frames ({100*fp_l2/len(l):.2f}%)')
print(f'BOTH (AND logic)         : {fp_both} frames ({100*fp_both/len(c):.2f}%)')

print('\n=== JUMP EVENTS (cos < 0.99) ===')
print(f"{'route':9s} {'frame':>5s} {'cos':>10s} {'L2':>8s} {'spd':>5s} {'det':>3s}")
for rid, fi, cc, ll, ss, dd in jump_events:
    print(f'{rid:9s} {fi:5d} {cc:10.6f} {ll:8.4f} {ss:5.1f} {dd:3.0f}')

print('\n=== PER-ROUTE FRAME-LEVEL TABLE ===')
print(f"{'route':9s} {'n':>4s} {'cos_mean':>10s} {'cos_std':>9s} "
      f"{'L2_mean':>8s} {'L2_std':>8s} {'det':>5s} {'det0%':>6s} {'spd':>5s}")
for (rid, n, nv, cm, cs, cmin, lm, ls, lmax, dm, d0, sm) in per_route_rows:
    print(f'{rid:9s} {nv:4d} {cm:10.6f} {cs:9.6f} {lm:8.4f} {ls:8.4f} '
          f'{dm:5.2f} {d0:5.1f}% {sm:5.1f}')

# correlation: speed vs L2 (motion-induced drift), det vs cos
m = s[:len(c)] if len(s) >= len(c) else s
print('\n=== CORRELATIONS ===')
print(f'corr(speed, L2)  = {np.corrcoef(s[:len(c)], l)[0,1]:+.3f}')
print(f'corr(det_count, L2) = {np.corrcoef(d[:len(c)], l)[0,1]:+.3f}')
