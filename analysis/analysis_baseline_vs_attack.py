#!/usr/bin/env python3
"""Baseline vs Attack comparative analysis for Tesla FSD adversarial attack experiment.

Loads all baseline and attack route data, computes statistics,
generates comparison visualizations, and outputs a summary report.
"""

import json
import os
import glob
import numpy as np

# ---- matplotlib setup (save to file, no display) ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

RESULTS_DIR = os.path.expanduser('~/carla-adversarial/results')
BASELINE_DIR = RESULTS_DIR
ATTACK_DIR = os.path.join(RESULTS_DIR, 'attack_sign_patch_front')
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Attack detection threshold (from baseline robust calibration)
THR_COS = 0.999740
THR_L2 = 0.027718
THR_CONSEC = 15  # consecutive frames


def load_routes(pattern):
    """Load all route JSON files matching a glob pattern."""
    files = sorted(glob.glob(pattern))
    routes = []
    for f in files:
        with open(f) as fp:
            routes.append(json.load(fp))
    return routes


def extract_frame_arrays(routes, field, attack_only=False, off_only=False):
    """Extract per-frame values from routes, returning list of arrays."""
    arrays = []
    for r in routes:
        vals = []
        for fr in r['frames']:
            if attack_only and not fr.get('attack_active', False):
                continue
            if off_only and fr.get('attack_active', False):
                continue
            vals.append(fr.get(field, 0.0))
        arrays.append(np.array(vals))
    return arrays


def main():
    print("=" * 70)
    print("  BASELINE vs ATTACK  COMPARATIVE ANALYSIS")
    print("  Attack: sign_patch_front (patch_sticker on front_main)")
    print("=" * 70)

    # ---- Load data ----
    baseline_routes = load_routes(os.path.join(BASELINE_DIR, 'baseline_route_*.json'))
    attack_routes = load_routes(os.path.join(ATTACK_DIR, 'attack_sign_patch_front_route_*.json'))
    print(f"\nLoaded: {len(baseline_routes)} baseline routes, {len(attack_routes)} attack routes")

    # ---- Overall stats ----
    bl_total_frames = sum(r['num_frames'] for r in baseline_routes)
    at_total_frames = sum(r['num_frames'] for r in attack_routes)
    print(f"Baseline total frames: {bl_total_frames}")
    print(f"Attack total frames:   {at_total_frames}")

    # ---- Per-route summary table ----
    print("\n" + "=" * 100)
    print(f"{'Route':>8} | {'Condition':>10} | {'Frames':>6} | {'DetCnt':>8} | {'DetScore':>10} | {'BEV_Cos':>12} | {'BEV_L2':>10} | {'Speed':>8}")
    print("-" * 100)

    for i, r in enumerate(baseline_routes):
        s = r.get('summary', {})
        print(f"{'R'+str(i):>8} | {'BASELINE':>10} | {r['num_frames']:>6} | "
              f"{s.get('det_count_mean',0):>8.2f} | {s.get('det_score_mean',0):>10.5f} | "
              f"{s.get('bev_cosine_mean',0):>12.8f} | {s.get('bev_l2_mean',0):>10.6f} | "
              f"{s.get('speed_mean',0):>8.3f}")

    for i, r in enumerate(attack_routes):
        s = r.get('summary', {})
        print(f"{'R'+str(i):>8} | {'ATTACK':>10} | {r['num_frames']:>6} | "
              f"{s.get('det_count_mean',0):>8.2f} | {s.get('det_score_mean',0):>10.5f} | "
              f"{s.get('bev_cosine_mean',0):>12.8f} | {s.get('bev_l2_mean',0):>10.6f} | "
              f"{s.get('speed_mean',0):>8.3f} | "
              f"atk={s.get('attack_frames',0)}/{r['num_frames']} ({s.get('attack_frame_ratio',0)*100:.1f}%)")

    # ---- Aggregate comparison ----
    print("\n" + "=" * 70)
    print("  AGGREGATE COMPARISON")
    print("=" * 70)

    # Baseline aggregates
    bl_det = [fr['det_count'] for r in baseline_routes for fr in r['frames']]
    bl_cos = [fr['bev_self_sim'] for r in baseline_routes for fr in r['frames'] if fr.get('bev_embed_available', False)]
    bl_l2 = [fr['bev_l2_dist'] for r in baseline_routes for fr in r['frames'] if fr.get('bev_embed_available', False)]
    bl_score = [fr['det_score_max'] for r in baseline_routes for fr in r['frames']]
    bl_speed = [fr['speed'] for r in baseline_routes for fr in r['frames']]

    # Attack aggregates - all frames
    at_det = [fr['det_count'] for r in attack_routes for fr in r['frames']]
    at_cos = [fr['bev_self_sim'] for r in attack_routes for fr in r['frames'] if fr.get('bev_embed_available', False)]
    at_l2 = [fr['bev_l2_dist'] for r in attack_routes for fr in r['frames'] if fr.get('bev_embed_available', False)]
    at_score = [fr['det_score_max'] for r in attack_routes for fr in r['frames']]
    at_speed = [fr['speed'] for r in attack_routes for fr in r['frames']]

    # Attack ON only
    at_on_det = [fr['det_count'] for r in attack_routes for fr in r['frames'] if fr.get('attack_active', False)]
    at_on_cos = [fr['bev_self_sim'] for r in attack_routes for fr in r['frames'] if fr.get('attack_active', False) and fr.get('bev_embed_available', False)]
    at_on_l2 = [fr['bev_l2_dist'] for r in attack_routes for fr in r['frames'] if fr.get('attack_active', False) and fr.get('bev_embed_available', False)]
    at_on_score = [fr['det_score_max'] for r in attack_routes for fr in r['frames'] if fr.get('attack_active', False)]

    # Attack OFF only
    at_off_det = [fr['det_count'] for r in attack_routes for fr in r['frames'] if not fr.get('attack_active', False)]
    at_off_cos = [fr['bev_self_sim'] for r in attack_routes for fr in r['frames'] if not fr.get('attack_active', False) and fr.get('bev_embed_available', False)]
    at_off_l2 = [fr['bev_l2_dist'] for r in attack_routes for fr in r['frames'] if not fr.get('attack_active', False) and fr.get('bev_embed_available', False)]

    metrics = [
        ("Det Count (filtered)", bl_det, at_det, at_on_det, at_off_det),
        ("BEV Cosine (self-sim)", bl_cos, at_cos, at_on_cos, at_off_cos),
        ("BEV L2 Distance", bl_l2, at_l2, at_on_l2, at_off_l2),
        ("Det Score (max)", bl_score, at_score, at_on_score, None),
    ]

    print(f"\n{'Metric':>25} | {'Baseline':>15} | {'Attack(all)':>15} | {'Attack(ON)':>15} | {'Attack(OFF)':>15} | {'Delta(ON-BL)':>15}")
    print("-" * 115)

    for name, bl, at, at_on, at_off in metrics:
        bl_arr = np.array(bl) if bl else np.array([0])
        at_arr = np.array(at) if at else np.array([0])
        at_on_arr = np.array(at_on) if at_on else np.array([0])
        at_off_arr = np.array(at_off) if at_off else np.array([0])

        bl_m = np.mean(bl_arr)
        at_m = np.mean(at_arr)
        at_on_m = np.mean(at_on_arr)
        at_off_m = np.mean(at_off_arr) if len(at_off_arr) > 0 else float('nan')
        delta = at_on_m - bl_m

        print(f"{name:>25} | {bl_m:>15.6f} | {at_m:>15.6f} | {at_on_m:>15.6f} | {at_off_m:>15.6f} | {delta:>+15.6f}")

    # ---- Attack detection analysis ----
    print("\n" + "=" * 70)
    print("  ATTACK DETECTION ANALYSIS")
    print("  Threshold: BEV cos < {:.6f} AND L2 > {:.6f}, consecutive {} frames".format(THR_COS, THR_L2, THR_CONSEC))
    print("=" * 70)

    total_attack_frames = 0
    total_detected = 0
    total_false_positives = 0

    for ri, r in enumerate(attack_routes):
        frames = r['frames']
        n = len(frames)
        consec = 0
        detected = False
        detection_frame = -1
        fp_count = 0

        for fi, fr in enumerate(frames):
            cos = fr.get('bev_self_sim', 1.0)
            l2 = fr.get('bev_l2_dist', 0.0)
            is_anomalous = (cos < THR_COS) and (l2 > THR_L2)

            if is_anomalous:
                consec += 1
                if consec >= THR_CONSEC and not detected:
                    detected = True
                    detection_frame = fi - THR_CONSEC + 1
            else:
                if consec > 0 and not detected:
                    pass  # transient, no alarm
                consec = 0

            # Count false positives: anomaly triggered when attack is NOT active
            if not fr.get('attack_active', False) and is_anomalous:
                fp_count += 1

        attack_frames = sum(1 for fr in frames if fr.get('attack_active', False))
        total_attack_frames += attack_frames

        status = "DETECTED" if detected else "MISSED"
        det_info = f"at frame {detection_frame}" if detected else "-"
        print(f"  Route {ri}: {status:>8} {det_info:>15} | attack_frames={attack_frames:>4} | FP_frames={fp_count:>4}")

        if detected:
            total_detected += 1
        total_false_positives += fp_count

    print(f"\n  Detection rate: {total_detected}/{len(attack_routes)} routes ({100*total_detected/len(attack_routes):.1f}%)")
    print(f"  Total attack frames: {total_attack_frames}")
    print(f"  Total false positive frames (attack-OFF): {total_false_positives}")

    # ---- Baseline false positive check ----
    bl_fp_events = 0
    for ri, r in enumerate(baseline_routes):
        frames = r['frames']
        consec = 0
        for fr in frames:
            cos = fr.get('bev_self_sim', 1.0)
            l2 = fr.get('bev_l2_dist', 0.0)
            if (cos < THR_COS) and (l2 > THR_L2):
                consec += 1
                if consec >= THR_CONSEC:
                    bl_fp_events += 1
                    break
            else:
                consec = 0
    print(f"  Baseline false positive events: {bl_fp_events}/{len(baseline_routes)}")

    # ---- Visualization ----
    print("\n" + "=" * 70)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 70)

    # ---- Figure 1: Time series comparison (Route 0) ----
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle('Baseline vs Attack: Time Series Comparison (Route 0)', fontsize=14, fontweight='bold')

    bl_r0 = baseline_routes[0]['frames']
    at_r0 = attack_routes[0]['frames']

    # Plot 1: Detection count
    ax = axes[0]
    bl_x = [fr['frame_idx'] for fr in bl_r0]
    bl_y = [fr['det_count'] for fr in bl_r0]
    at_x = [fr['frame_idx'] for fr in at_r0]
    at_y = [fr['det_count'] for fr in at_r0]
    at_colors = ['red' if fr.get('attack_active', False) else 'gray' for fr in at_r0]

    ax.plot(bl_x, bl_y, 'b-', alpha=0.7, label='Baseline', linewidth=1)
    ax.scatter(at_x, at_y, c=at_colors, s=8, alpha=0.6, label='Attack (ON/OFF)')
    ax.set_ylabel('Det Count')
    ax.legend(loc='upper right')
    ax.set_title('Detection Count (filtered, score > 0.05)')
    ax.grid(True, alpha=0.3)

    # Plot 2: BEV Cosine
    ax = axes[1]
    bl_cos_y = [fr['bev_self_sim'] for fr in bl_r0]
    at_cos_y = [fr['bev_self_sim'] for fr in at_r0]
    ax.plot(bl_x, bl_cos_y, 'b-', alpha=0.7, label='Baseline')
    ax.scatter(at_x, at_cos_y, c=at_colors, s=8, alpha=0.6)
    ax.axhline(y=THR_COS, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({THR_COS})')
    ax.set_ylabel('BEV Cosine')
    ax.legend(loc='lower left')
    ax.set_title('BEV Self-Similarity (cosine)')
    ax.grid(True, alpha=0.3)

    # Plot 3: BEV L2
    ax = axes[2]
    bl_l2_y = [fr['bev_l2_dist'] for fr in bl_r0]
    at_l2_y = [fr['bev_l2_dist'] for fr in at_r0]
    ax.plot(bl_x, bl_l2_y, 'b-', alpha=0.7, label='Baseline')
    ax.scatter(at_x, at_l2_y, c=at_colors, s=8, alpha=0.6)
    ax.axhline(y=THR_L2, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({THR_L2})')
    ax.set_ylabel('BEV L2')
    ax.legend(loc='upper right')
    ax.set_title('BEV L2 Distance')
    ax.grid(True, alpha=0.3)

    # Plot 4: Speed
    ax = axes[3]
    bl_spd = [fr['speed'] for fr in bl_r0]
    at_spd = [fr['speed'] for fr in at_r0]
    ax.plot(bl_x, bl_spd, 'b-', alpha=0.7, label='Baseline')
    ax.scatter(at_x, at_spd, c=at_colors, s=8, alpha=0.6)
    ax.set_ylabel('Speed (m/s)')
    ax.set_xlabel('Frame Index')
    ax.legend(loc='upper right')
    ax.set_title('Vehicle Speed')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_timeseries_route0.png'), dpi=150, bbox_inches='tight')
    print("  Saved: fig1_timeseries_route0.png")
    plt.close()

    # ---- Figure 2: Distribution comparison (all routes) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline vs Attack: Distribution Comparison (All Routes)', fontsize=14, fontweight='bold')

    # Det count histogram
    ax = axes[0, 0]
    ax.hist(bl_det, bins=30, alpha=0.5, color='blue', label='Baseline', density=True)
    ax.hist(at_on_det, bins=30, alpha=0.5, color='red', label='Attack ON', density=True)
    ax.set_xlabel('Detection Count')
    ax.set_ylabel('Density')
    ax.set_title('Det Count Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # BEV cosine histogram
    ax = axes[0, 1]
    bins_cos = np.linspace(min(min(bl_cos), min(at_cos)), 1.0, 50)
    ax.hist(bl_cos, bins=bins_cos, alpha=0.5, color='blue', label='Baseline', density=True)
    ax.hist(at_on_cos, bins=bins_cos, alpha=0.5, color='red', label='Attack ON', density=True)
    ax.axvline(x=THR_COS, color='r', linestyle='--', alpha=0.7, label='Threshold')
    ax.set_xlabel('BEV Cosine')
    ax.set_ylabel('Density')
    ax.set_title('BEV Cosine Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # BEV L2 histogram
    ax = axes[1, 0]
    max_l2 = max(max(bl_l2), max(at_l2))
    bins_l2 = np.linspace(0, min(max_l2, 0.1), 50)
    ax.hist(bl_l2, bins=bins_l2, alpha=0.5, color='blue', label='Baseline', density=True)
    ax.hist(at_on_l2, bins=bins_l2, alpha=0.5, color='red', label='Attack ON', density=True)
    ax.axvline(x=THR_L2, color='r', linestyle='--', alpha=0.7, label='Threshold')
    ax.set_xlabel('BEV L2 Distance')
    ax.set_ylabel('Density')
    ax.set_title('BEV L2 Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Detection score histogram
    ax = axes[1, 1]
    max_score = max(max(bl_score), max(at_score))
    bins_score = np.linspace(0, min(max_score, 0.15), 50)
    ax.hist(bl_score, bins=bins_score, alpha=0.5, color='blue', label='Baseline', density=True)
    ax.hist(at_on_score, bins=bins_score, alpha=0.5, color='red', label='Attack ON', density=True)
    ax.set_xlabel('Max Detection Score')
    ax.set_ylabel('Density')
    ax.set_title('Detection Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_distributions.png'), dpi=150, bbox_inches='tight')
    print("  Saved: fig2_distributions.png")
    plt.close()

    # ---- Figure 3: Per-route comparison bar chart ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Per-Route Comparison: Baseline vs Attack', fontsize=14, fontweight='bold')

    route_ids = list(range(len(baseline_routes)))
    bl_det_means = [np.mean([fr['det_count'] for fr in r['frames']]) for r in baseline_routes]
    at_det_means = [np.mean([fr['det_count'] for fr in r['frames']]) for r in attack_routes]
    at_on_det_means = []
    for r in attack_routes:
        on_dets = [fr['det_count'] for fr in r['frames'] if fr.get('attack_active', False)]
        at_on_det_means.append(np.mean(on_dets) if on_dets else 0)

    bl_cos_means = [np.mean([fr['bev_self_sim'] for fr in r['frames'] if fr.get('bev_embed_available', False)]) for r in baseline_routes]
    at_cos_means = [np.mean([fr['bev_self_sim'] for fr in r['frames'] if fr.get('bev_embed_available', False)]) for r in attack_routes]

    bl_l2_means = [np.mean([fr['bev_l2_dist'] for fr in r['frames'] if fr.get('bev_embed_available', False)]) for r in baseline_routes]
    at_l2_means = [np.mean([fr['bev_l2_dist'] for fr in r['frames'] if fr.get('bev_embed_available', False)]) for r in attack_routes]

    x = np.arange(len(route_ids))
    w = 0.35

    # Det count
    ax = axes[0]
    ax.bar(x - w/2, bl_det_means, w, label='Baseline', color='steelblue')
    ax.bar(x + w/2, at_det_means, w, label='Attack', color='indianred')
    ax.set_xlabel('Route')
    ax.set_ylabel('Mean Det Count')
    ax.set_title('Detection Count')
    ax.set_xticks(x)
    ax.set_xticklabels([f'R{i}' for i in route_ids])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # BEV Cosine
    ax = axes[1]
    ax.bar(x - w/2, bl_cos_means, w, label='Baseline', color='steelblue')
    ax.bar(x + w/2, at_cos_means, w, label='Attack', color='indianred')
    ax.axhline(y=THR_COS, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Route')
    ax.set_ylabel('Mean BEV Cosine')
    ax.set_title('BEV Self-Similarity')
    ax.set_xticks(x)
    ax.set_xticklabels([f'R{i}' for i in route_ids])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # BEV L2
    ax = axes[2]
    ax.bar(x - w/2, bl_l2_means, w, label='Baseline', color='steelblue')
    ax.bar(x + w/2, at_l2_means, w, label='Attack', color='indianred')
    ax.axhline(y=THR_L2, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Route')
    ax.set_ylabel('Mean BEV L2')
    ax.set_title('BEV L2 Distance')
    ax.set_xticks(x)
    ax.set_xticklabels([f'R{i}' for i in route_ids])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_per_route_comparison.png'), dpi=150, bbox_inches='tight')
    print("  Saved: fig3_per_route_comparison.png")
    plt.close()

    # ---- Figure 4: Attack activation timeline ----
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title('Attack Activation Timeline (All Routes)', fontsize=13, fontweight='bold')

    for ri, r in enumerate(attack_routes):
        frames = r['frames']
        for fr in frames:
            if fr.get('attack_active', False):
                ax.scatter(fr['frame_idx'], ri, c='red', s=10, alpha=0.5)
            else:
                ax.scatter(fr['frame_idx'], ri, c='gray', s=5, alpha=0.3)

    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Route')
    ax.set_yticks(range(len(attack_routes)))
    ax.set_yticklabels([f'Route {i}' for i in range(len(attack_routes))])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_attack_timeline.png'), dpi=150, bbox_inches='tight')
    print("  Saved: fig4_attack_timeline.png")
    plt.close()

    # ---- Figure 5: BEV drift scatter (cos vs L2) ----
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('BEV Feature Space: Baseline vs Attack ON', fontsize=13, fontweight='bold')

    ax.scatter(bl_l2, bl_cos, c='blue', s=10, alpha=0.3, label='Baseline')
    ax.scatter(at_on_l2, at_on_cos, c='red', s=10, alpha=0.3, label='Attack ON')

    # Threshold lines
    ax.axhline(y=THR_COS, color='blue', linestyle='--', alpha=0.5, label=f'Cos threshold ({THR_COS})')
    ax.axvline(x=THR_L2, color='green', linestyle='--', alpha=0.5, label=f'L2 threshold ({THR_L2})')

    # Danger zone
    ax.fill_between([0, max(max(bl_l2), max(at_l2), THR_L2*2)],
                     [0, 0], [THR_COS, THR_COS],
                     alpha=0.1, color='red', label='Anomaly zone')

    ax.set_xlabel('BEV L2 Distance')
    ax.set_ylabel('BEV Cosine Self-Similarity')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(0.1, max(max(bl_l2), max(at_l2)) * 1.2))

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_bev_scatter.png'), dpi=150, bbox_inches='tight')
    print("  Saved: fig5_bev_scatter.png")
    plt.close()

    # ---- Save analysis report ----
    report = {
        'experiment': 'sign_patch_front baseline vs attack',
        'baseline': {
            'num_routes': len(baseline_routes),
            'total_frames': bl_total_frames,
            'det_count_mean': float(np.mean(bl_det)),
            'det_count_std': float(np.std(bl_det)),
            'bev_cosine_mean': float(np.mean(bl_cos)),
            'bev_cosine_std': float(np.std(bl_cos)),
            'bev_l2_mean': float(np.mean(bl_l2)),
            'bev_l2_std': float(np.std(bl_l2)),
            'speed_mean': float(np.mean(bl_speed)),
        },
        'attack_all': {
            'num_routes': len(attack_routes),
            'total_frames': at_total_frames,
            'det_count_mean': float(np.mean(at_det)),
            'det_count_std': float(np.std(at_det)),
            'bev_cosine_mean': float(np.mean(at_cos)),
            'bev_cosine_std': float(np.std(at_cos)),
            'bev_l2_mean': float(np.mean(at_l2)),
            'bev_l2_std': float(np.std(at_l2)),
        },
        'attack_on': {
            'total_frames': len(at_on_det),
            'det_count_mean': float(np.mean(at_on_det)),
            'det_count_std': float(np.std(at_on_det)),
            'bev_cosine_mean': float(np.mean(at_on_cos)) if at_on_cos else None,
            'bev_l2_mean': float(np.mean(at_on_l2)) if at_on_l2 else None,
        },
        'attack_off': {
            'total_frames': len(at_off_det),
            'det_count_mean': float(np.mean(at_off_det)) if at_off_det else None,
        },
        'detection': {
            'threshold_cos': THR_COS,
            'threshold_l2': THR_L2,
            'threshold_consecutive': THR_CONSEC,
            'routes_detected': total_detected,
            'routes_total': len(attack_routes),
            'detection_rate': total_detected / len(attack_routes),
            'baseline_false_positive_events': bl_fp_events,
            'attack_false_positive_frames': total_false_positives,
        },
        'key_findings': [],
    }

    # Key findings
    bl_det_m = np.mean(bl_det)
    at_on_det_m = np.mean(at_on_det) if at_on_det else 0
    bl_cos_m = np.mean(bl_cos)
    at_on_cos_m = np.mean(at_on_cos) if at_on_cos else 1.0
    bl_l2_m = np.mean(bl_l2)
    at_on_l2_m = np.mean(at_on_l2) if at_on_l2 else 0.0

    report['key_findings'].append(
        f"Detection count: baseline={bl_det_m:.2f}, attack_ON={at_on_det_m:.2f} "
        f"(delta={at_on_det_m - bl_det_m:+.2f})"
    )
    report['key_findings'].append(
        f"BEV cosine: baseline={bl_cos_m:.8f}, attack_ON={at_on_cos_m:.8f} "
        f"(delta={at_on_cos_m - bl_cos_m:+.8f})"
    )
    report['key_findings'].append(
        f"BEV L2: baseline={bl_l2_m:.6f}, attack_ON={at_on_l2_m:.6f} "
        f"(delta={at_on_l2_m - bl_l2_m:+.6f})"
    )
    report['key_findings'].append(
        f"Attack detection rate: {total_detected}/{len(attack_routes)} routes "
        f"({100*total_detected/len(attack_routes):.1f}%) using robust threshold"
    )
    report['key_findings'].append(
        f"Baseline false positives: {bl_fp_events} events | "
        f"Attack false positives: {total_false_positives} frames"
    )

    # Analysis conclusion
    bev_drift_detected = (at_on_cos_m < bl_cos_m - 1e-6) or (at_on_l2_m > bl_l2_m + 1e-5)
    if bev_drift_detected:
        report['conclusion'] = "BEV drift detected under attack. The image-space patch attack on front_main camera causes measurable changes in BEV features."
    else:
        report['conclusion'] = "No significant BEV drift detected. The image-space patch attack does NOT cause measurable BEV feature changes at the current attack intensity. The BEVFormer representation appears robust to this attack."

    report['conclusion'] += " Note: The extremely high BEV cosine (>0.9999) in both conditions suggests the surrogate model's BEV representation is highly stable, possibly due to the domain gap between CARLA and training data dominating the feature space."

    with open(os.path.join(OUTPUT_DIR, 'analysis_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print("\n  Saved: analysis_report.json")

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("  KEY FINDINGS")
    print("=" * 70)
    for finding in report['key_findings']:
        print(f"  • {finding}")

    print(f"\n  CONCLUSION: {report['conclusion']}")

    print(f"\n  Output directory: {OUTPUT_DIR}")
    print("  Files: fig1-5 PNGs + analysis_report.json")
    print("=" * 70)


if __name__ == '__main__':
    main()
