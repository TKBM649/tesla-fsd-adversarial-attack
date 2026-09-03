#!/usr/bin/env python3
"""Verify the teleport chimera-frame fix in a smoke collection JSON.

Old-bug signature: a frame within ~6 frames AFTER a post_reset frame
shows an intermediate cosine jump (0 < sim < 0.99).
Fixed behavior: post_reset frame and the one after it report sim == 0
(noise floor ~1e-6), and every subsequent frame is > 0.99 (or == 0 for
route starts). sim == 0.0 exactly at the immediate next frame is the
expected no-history state.

Exit code 0 = PASS, 1 = FAIL.
"""
import json
import sys

CHIMERA_TAIL = 6  # frames after a teleport within which jumps are suspects

def main(path):
    with open(path, 'r', encoding='utf-8') as f:
        frames = json.load(f)['frames']

    n = len(frames)
    sims = [fr.get('bev_self_sim', 0.0) for fr in frames]
    reset_idx = [i for i, fr in enumerate(frames) if fr.get('post_reset')]

    suspects = []  # (idx, sim, reason)
    for i, s in enumerate(sims):
        if i > 0 and 0.0 < s < 0.99:
            recent_reset = any(0 < i - r <= CHIMERA_TAIL for r in reset_idx)
            suspects.append((i, s, 'chimera' if recent_reset else 'scene_jump'))

    chimera = [t for t in suspects if t[2] == 'chimera']
    next_ok = all(sims[min(r + 1, n - 1)] == 0.0 for r in reset_idx) if reset_idx else False

    print(f'file                : {path}')
    print(f'frames              : {n}')
    print(f'teleport resets     : {len(reset_idx)} at frames {reset_idx}  (need >= 1)')
    print(f'reset tails sim==0  : {next_ok}  (next frame after each reset must be 0)')
    print(f'suspect frames >0.99: {len(suspects)}')
    for i, s, why in suspects:
        print(f'   frame {i:4d}  sim={s:.6f}  [{why}]')

    passed = bool(reset_idx) and next_ok and not chimera
    print(f'CHIMERA FRAMES      : {len(chimera)}  (must be 0)')
    print('RESULT: ' + ('PASS' if passed else 'FAIL'))
    return 0 if passed else 1

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
