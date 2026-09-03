#!/usr/bin/env python3
"""
analyze_collapse_points.py -- Phase 3B analysis and visualization

Loads collapse scan results, detects collapse points, generates 6 figures,
and outputs a comprehensive report.

Usage:
  python analyze_collapse_points.py --scan-dir ~/carla-adversarial/results/collapse_scan/h1_full
  python analyze_collapse_points.py --scan-dir ~/carla-adversarial/results/collapse_scan/h1_quick
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

# ---- matplotlib setup ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---- Project imports (no CARLA needed) ----
from collapse_configs import (
    CAMERA_GROUPS, NEGATIVE_CONTROL_CAMS, BEV_CONSUMED_CAMERAS,
    condition_to_dirname, dirname_to_condition,
)


# ============================================================================
# Data loading
# ============================================================================

def load_scan_results(scan_dir):
    """
    Load all condition summaries from a scan directory.

    Expected structure:
      scan_dir/
        group_id/
          group_id__i0.00__d1.0/
            condition_summary.json
            route_0.json, route_1.json, ...

    Returns:
        list of condition dicts with keys:
          group_id, intensity, duty_cycle, per_route_summary, per_route
    """
    conditions = []

    # Find all condition_summary.json files
    pattern = os.path.join(scan_dir, '*', '*', 'condition_summary.json')
    for fpath in sorted(glob.glob(pattern)):
        with open(fpath, 'r') as f:
            summary = json.load(f)

        # Parse directory name for condition parameters
        cond_dir = os.path.dirname(fpath)
        group_dir = os.path.dirname(cond_dir)
        group_id = os.path.basename(group_dir)

        # Load per-route detailed data
        route_files = sorted(glob.glob(os.path.join(cond_dir, 'route_*.json')))
        per_route = []
        for rf in route_files:
            with open(rf, 'r') as f:
                per_route.append(json.load(f))

        summary['group_id'] = group_id
        summary['per_route'] = per_route
        conditions.append(summary)

    print(f"Loaded {len(conditions)} conditions from {scan_dir}")
    return conditions


# ============================================================================
# Statistics computation
# ============================================================================

def compute_condition_stats(conditions):
    """
    Compute per-condition aggregate statistics.

    For each condition, compute:
      - det_count_median: median of per-route median det_counts (attack-ON frames)
      - det_count_iqr: IQR of per-route median det_counts
      - bev_cosine_median: median BEV cosine (attack-ON frames)
      - baseline_det: median det_count at intensity=0

    Returns:
        stats_dict: {
            (group_id, duty_cycle): {
                'intensities': [...],
                'det_medians': [...],
                'det_q25': [...],
                'det_q75': [...],
                'bev_cos_medians': [...],
                'baseline_det': float,
                'baseline_std': float,
            }
        }
    """
    # Group conditions by (group_id, duty_cycle)
    grouped = {}
    for cond in conditions:
        gid = cond['group_id']
        duty = cond['duty_cycle']
        inten = cond['intensity']
        key = (gid, duty)

        if key not in grouped:
            grouped[key] = []
        grouped[key].append(cond)

    stats_dict = {}
    for (gid, duty), conds in grouped.items():
        conds.sort(key=lambda c: c['intensity'])

        intensities = []
        det_medians = []
        det_q25 = []
        det_q75 = []
        bev_cos_medians = []

        for cond in conds:
            inten = cond['intensity']
            intensities.append(inten)

            # Compute attack-ON frame statistics across routes
            route_det_medians = []
            route_bev_cos = []
            for rd in cond['per_route']:
                # Filter to attack-ON frames
                det = np.array(rd['det_count'])
                atk = np.array(rd['attack_active'])
                bev = np.array(rd['bev_self_sim'])

                if atk.sum() > 0:
                    route_det_medians.append(float(np.median(det[atk])))
                    bev_on = bev[atk]
                    bev_on = bev_on[bev_on > 0]
                    if len(bev_on) > 0:
                        route_bev_cos.append(float(np.median(bev_on)))
                else:
                    route_det_medians.append(float(np.median(det)))

            det_arr = np.array(route_det_medians)
            det_medians.append(float(np.median(det_arr)))
            det_q25.append(float(np.percentile(det_arr, 25)))
            det_q75.append(float(np.percentile(det_arr, 75)))

            if route_bev_cos:
                bev_cos_medians.append(float(np.median(route_bev_cos)))
            else:
                bev_cos_medians.append(1.0)

        # Baseline (intensity=0)
        baseline_idx = np.argmin(np.array(intensities))
        baseline_det = det_medians[baseline_idx]
        baseline_dets = []
        for cond in conds:
            if cond['intensity'] == intensities[baseline_idx]:
                for rd in cond['per_route']:
                    baseline_dets.extend(rd['det_count'])
        baseline_std = float(np.std(baseline_dets)) if baseline_dets else 0.0

        stats_dict[(gid, duty)] = {
            'intensities': intensities,
            'det_medians': det_medians,
            'det_q25': det_q25,
            'det_q75': det_q75,
            'bev_cos_medians': bev_cos_medians,
            'baseline_det': baseline_det,
            'baseline_std': baseline_std,
        }

    return stats_dict


# ============================================================================
# Collapse point detection
# ============================================================================

def detect_collapse_point(intensities, det_medians, baseline_det,
                          baseline_std, method='derivative'):
    """
    Detect the collapse intensity from a det_count vs intensity curve.

    Methods:
      'derivative': max first derivative (steepest increase)
      'threshold': first crossing of baseline + 3*max(std, 1.0)
      'ratio': first intensity where det > max(baseline * 10, 10)

    Returns:
        collapse_intensity: float or None
    """
    intensities = np.array(intensities)
    det_medians = np.array(det_medians)

    # Skip baseline point
    mask = intensities > 0
    if mask.sum() == 0:
        return None
    ints = intensities[mask]
    dets = det_medians[mask]

    if method == 'derivative':
        if len(ints) < 2:
            return None
        diffs = np.diff(dets) / np.diff(ints)
        if len(diffs) == 0:
            return None
        idx = np.argmax(diffs)
        # Collapse point = midpoint between idx and idx+1
        collapse = (ints[idx] + ints[idx + 1]) / 2
        return float(collapse)

    elif method == 'threshold':
        thr = baseline_det + 3 * max(baseline_std, 1.0)
        above = dets > thr
        if not above.any():
            return None
        idx = np.argmax(above)
        return float(ints[idx])

    elif method == 'ratio':
        thr = max(baseline_det * 10, 10)
        above = dets > thr
        if not above.any():
            return None
        idx = np.argmax(above)
        return float(ints[idx])

    return None


def compute_all_collapse_points(stats_dict, methods=None):
    """
    Compute collapse points for all conditions using multiple methods.

    Returns:
        collapse_dict: {
            (group_id, duty_cycle): {
                method: collapse_intensity or None
            }
        }
    """
    if methods is None:
        methods = ['derivative', 'threshold', 'ratio']

    collapse_dict = {}
    for (gid, duty), stats in stats_dict.items():
        collapse_dict[(gid, duty)] = {}
        for method in methods:
            cp = detect_collapse_point(
                stats['intensities'], stats['det_medians'],
                stats['baseline_det'], stats['baseline_std'],
                method=method,
            )
            collapse_dict[(gid, duty)][method] = cp

    return collapse_dict


# ============================================================================
# Visualization
# ============================================================================

def _get_color(group_id):
    """Consistent color for each camera group."""
    colors = {
        'front_main': '#e41a1c', 'front_wide': '#377eb8',
        'side_front_left': '#4daf4a', 'side_front_right': '#984ea3',
        'side_rear_left': '#ff7f00', 'rear': '#a65628',
        'front_narrow': '#999999', 'side_rear_right': '#666666',
    }
    # Extract camera name from group_id
    for cam, color in colors.items():
        if cam in group_id:
            return color
    return '#000000'


def _get_marker(n_cams):
    if n_cams == 1:
        return 'o'
    elif n_cams == 2:
        return 's'
    return '^'


def fig_h1_single_camera(stats_dict, output_dir):
    """Fig 1: Single camera Det Count vs Intensity curves."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title('H1: Single Camera Attack - Detection Count vs Intensity',
                 fontsize=13, fontweight='bold')

    for (gid, duty), stats in sorted(stats_dict.items()):
        if not gid.startswith('single_'):
            continue
        if duty != 1.0:
            continue

        cams = CAMERA_GROUPS.get(gid, ())
        n_cams = len(cams)
        color = _get_color(gid)
        marker = _get_marker(n_cams)
        is_neg = any(c in NEGATIVE_CONTROL_CAMS for c in cams)

        ls = '--' if is_neg else '-'
        alpha = 0.5 if is_neg else 1.0
        lw = 1.5 if is_neg else 2.0

        ax.plot(stats['intensities'], stats['det_medians'],
                marker=marker, color=color, linestyle=ls,
                linewidth=lw, alpha=alpha, markersize=6,
                label=gid.replace('single_', ''))

        # Fill IQR
        ax.fill_between(stats['intensities'],
                        stats['det_q25'], stats['det_q75'],
                        alpha=0.1, color=color)

    ax.set_xlabel('Attack Intensity (patch_frac)', fontsize=12)
    ax.set_ylabel('Detection Count (median, attack-ON)', fontsize=12)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.05)

    plt.tight_layout()
    fpath = os.path.join(output_dir, 'fig1_h1_single_camera.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")
    return fpath


def fig_h2_multi_camera(stats_dict, collapse_dict, output_dir):
    """Fig 2: Single vs Dual vs Triple camera collapse comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title('H2: Multi-Camera Synergy - Collapse Point Comparison',
                 fontsize=13, fontweight='bold')

    # Group by number of cameras
    categories = {'single': {}, 'dual': {}, 'triple': {}}
    for (gid, duty), stats in stats_dict.items():
        if duty != 1.0:
            continue
        cams = CAMERA_GROUPS.get(gid, ())
        # Skip negative controls
        if any(c in NEGATIVE_CONTROL_CAMS for c in cams):
            continue
        n = len(cams)
        if n == 1:
            cat = 'single'
        elif n == 2:
            cat = 'dual'
        else:
            cat = 'triple'

        cp = collapse_dict.get((gid, duty), {}).get('derivative')
        if cp is not None:
            categories[cat][gid] = cp

    # Bar chart
    x_pos = 0
    tick_positions = []
    tick_labels = []
    colors_cat = {'single': '#4daf4a', 'dual': '#377eb8', 'triple': '#e41a1c'}

    for cat_name in ['single', 'dual', 'triple']:
        data = categories[cat_name]
        if not data:
            continue
        for gid, cp in sorted(data.items()):
            short_name = gid.replace('single_', '').replace('dual_', '').replace('triple_', '')
            ax.bar(x_pos, cp, color=colors_cat[cat_name], alpha=0.7, width=0.6)
            tick_positions.append(x_pos)
            tick_labels.append(short_name)
            x_pos += 1
        x_pos += 0.5  # gap between categories

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Collapse Intensity (patch_frac)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_cat[c], alpha=0.7, label=c)
                       for c in ['single', 'dual', 'triple']]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    fpath = os.path.join(output_dir, 'fig2_h2_multi_camera.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")
    return fpath


def fig_h3_duty_cycle(stats_dict, output_dir):
    """Fig 3: Duty cycle effect on collapse curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle('H3: Duty Cycle Effect on Attack Vulnerability',
                 fontsize=13, fontweight='bold')

    # Find groups that have multiple duty cycles
    duty_groups = {}
    for (gid, duty) in stats_dict.keys():
        if gid not in duty_groups:
            duty_groups[gid] = []
        duty_groups[gid].append(duty)

    # Pick groups with multiple duty cycles
    multi_duty_groups = {gid: duties for gid, duties in duty_groups.items()
                         if len(duties) > 1}

    if not multi_duty_groups:
        for ax in axes:
            ax.text(0.5, 0.5, 'No multi-duty data', ha='center', va='center',
                    transform=ax.transAxes)
        plt.tight_layout()
        fpath = os.path.join(output_dir, 'fig3_h3_duty_cycle.png')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close()
        return fpath

    # Plot up to 3 groups
    for ax_idx, (gid, duties) in enumerate(
            sorted(multi_duty_groups.items())[:3]):
        ax = axes[ax_idx]
        duties.sort()

        for duty in duties:
            key = (gid, duty)
            if key not in stats_dict:
                continue
            stats = stats_dict[key]
            ax.plot(stats['intensities'], stats['det_medians'],
                    marker='o', linewidth=2, markersize=5,
                    label=f'duty={duty}')
            ax.fill_between(stats['intensities'],
                            stats['det_q25'], stats['det_q75'],
                            alpha=0.1)

        short_name = gid.replace('single_', '').replace('dual_', '')
        ax.set_title(short_name, fontsize=11)
        ax.set_xlabel('Intensity', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Det Count (median)', fontsize=11)

    plt.tight_layout()
    fpath = os.path.join(output_dir, 'fig3_h3_duty_cycle.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")
    return fpath


def fig_collapse_heatmap(stats_dict, collapse_dict, output_dir):
    """Fig 4: Camera group x Intensity heatmap of Det Count."""
    # Build matrix for duty_cycle=1.0 only
    groups = []
    intensities_set = set()
    for (gid, duty), stats in stats_dict.items():
        if duty != 1.0:
            continue
        if any(c in NEGATIVE_CONTROL_CAMS for c in CAMERA_GROUPS.get(gid, ())):
            continue
        groups.append(gid)
        intensities_set.update(stats['intensities'])

    intensities_sorted = sorted(intensities_set)
    if not groups or not intensities_sorted:
        print("  [WARN] Not enough data for heatmap")
        return None

    # Build det_count matrix
    matrix = np.full((len(groups), len(intensities_sorted)), np.nan)
    inten_idx = {v: i for i, v in enumerate(intensities_sorted)}

    for gi, gid in enumerate(groups):
        key = (gid, 1.0)
        if key not in stats_dict:
            continue
        stats = stats_dict[key]
        for inten, det in zip(stats['intensities'], stats['det_medians']):
            if inten in inten_idx:
                matrix[gi, inten_idx[inten]] = det

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title('Collapse Heatmap: Detection Count by Camera Group x Intensity',
                 fontsize=12, fontweight='bold')

    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest')
    ax.set_xticks(range(len(intensities_sorted)))
    ax.set_xticklabels([f'{v:.2f}' for v in intensities_sorted],
                       rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(groups)))
    short_labels = [g.replace('single_', 's_').replace('dual_', 'd_').replace('triple_', 't_')
                    for g in groups]
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel('Attack Intensity (patch_frac)', fontsize=11)
    ax.set_ylabel('Camera Group', fontsize=11)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Det Count (median)', fontsize=10)

    # Mark collapse points
    for gi, gid in enumerate(groups):
        cp = collapse_dict.get((gid, 1.0), {}).get('derivative')
        if cp is not None and cp in inten_idx:
            ax.plot(inten_idx[cp], gi, 'k*', markersize=10)

    plt.tight_layout()
    fpath = os.path.join(output_dir, 'fig4_collapse_heatmap.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")
    return fpath


def fig_bev_stability(stats_dict, output_dir):
    """Fig 5: BEV Cosine vs Intensity (verify encoder robustness)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title('BEV Encoder Stability: Cosine Self-Similarity vs Attack Intensity',
                 fontsize=12, fontweight='bold')

    for (gid, duty), stats in sorted(stats_dict.items()):
        if duty != 1.0:
            continue
        cams = CAMERA_GROUPS.get(gid, ())
        if any(c in NEGATIVE_CONTROL_CAMS for c in cams):
            continue

        color = _get_color(gid)
        n_cams = len(cams)
        marker = _get_marker(n_cams)
        short = gid.replace('single_', '').replace('dual_', '').replace('triple_', '')

        ax.plot(stats['intensities'], stats['bev_cos_medians'],
                marker=marker, color=color, linewidth=1.5, markersize=5,
                alpha=0.8, label=short)

    ax.axhline(y=0.999, color='r', linestyle='--', alpha=0.5,
               label='Threshold (0.999)')
    ax.set_xlabel('Attack Intensity (patch_frac)', fontsize=12)
    ax.set_ylabel('BEV Cosine Self-Similarity (median)', fontsize=12)
    ax.legend(loc='lower left', fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0.990, 1.001)

    plt.tight_layout()
    fpath = os.path.join(output_dir, 'fig5_bev_stability.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")
    return fpath


def fig_negative_control(stats_dict, output_dir):
    """Fig 6: Negative control cameras should show no response."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Negative Control: Non-BEV Cameras Should Show No Response',
                 fontsize=12, fontweight='bold')

    # Plot negative control cameras
    for (gid, duty), stats in sorted(stats_dict.items()):
        if duty != 1.0:
            continue
        cams = CAMERA_GROUPS.get(gid, ())
        if not any(c in NEGATIVE_CONTROL_CAMS for c in cams):
            continue

        color = _get_color(gid)
        short = gid.replace('single_', '')
        ax.plot(stats['intensities'], stats['det_medians'],
                marker='x', color=color, linewidth=2, markersize=8,
                label=short)

    # Also plot front_main for comparison
    key = ('single_front_main', 1.0)
    if key in stats_dict:
        stats = stats_dict[key]
        ax.plot(stats['intensities'], stats['det_medians'],
                marker='o', color='#e41a1c', linewidth=2, markersize=6,
                alpha=0.5, label='front_main (reference)')

    ax.set_xlabel('Attack Intensity (patch_frac)', fontsize=12)
    ax.set_ylabel('Det Count (median)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.05)

    plt.tight_layout()
    fpath = os.path.join(output_dir, 'fig6_negative_control.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")
    return fpath


# ============================================================================
# Report generation
# ============================================================================

def generate_report(stats_dict, collapse_dict, output_dir):
    """Generate collapse_report.json with collapse point values."""
    report = {
        'conditions_analyzed': len(stats_dict),
        'collapse_points': {},
        'hypothesis_tests': {},
    }

    # Collapse points
    for (gid, duty), methods in sorted(collapse_dict.items()):
        key = f'{gid}__d{duty:.1f}'
        report['collapse_points'][key] = {
            method: (float(cp) if cp is not None else None)
            for method, cp in methods.items()
        }

    # H1: front cameras most critical
    h1_collapse = {}
    for (gid, duty), methods in collapse_dict.items():
        if duty != 1.0 or not gid.startswith('single_'):
            continue
        cams = CAMERA_GROUPS.get(gid, ())
        if any(c in NEGATIVE_CONTROL_CAMS for c in cams):
            continue
        cp = methods.get('derivative')
        if cp is not None:
            h1_collapse[gid] = cp
    report['hypothesis_tests']['H1_front_critical'] = {
        'collapse_by_camera': h1_collapse,
        'most_vulnerable': min(h1_collapse, key=h1_collapse.get) if h1_collapse else None,
        'least_vulnerable': max(h1_collapse, key=h1_collapse.get) if h1_collapse else None,
    }

    # H2: multi-camera synergy
    h2_collapse = {'single': [], 'dual': [], 'triple': []}
    for (gid, duty), methods in collapse_dict.items():
        if duty != 1.0:
            continue
        cams = CAMERA_GROUPS.get(gid, ())
        if any(c in NEGATIVE_CONTROL_CAMS for c in cams):
            continue
        cp = methods.get('derivative')
        if cp is None:
            continue
        n = len(cams)
        if n == 1:
            h2_collapse['single'].append(cp)
        elif n == 2:
            h2_collapse['dual'].append(cp)
        else:
            h2_collapse['triple'].append(cp)

    report['hypothesis_tests']['H2_multi_synergy'] = {
        'median_collapse_single': float(np.median(h2_collapse['single'])) if h2_collapse['single'] else None,
        'median_collapse_dual': float(np.median(h2_collapse['dual'])) if h2_collapse['dual'] else None,
        'median_collapse_triple': float(np.median(h2_collapse['triple'])) if h2_collapse['triple'] else None,
        'confirmed': (np.median(h2_collapse['dual']) < np.median(h2_collapse['single'])
                      if h2_collapse['dual'] and h2_collapse['single'] else None),
    }

    # Save
    report_path = os.path.join(output_dir, 'collapse_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")
    return report


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Phase 3B: Analyze collapse scan results')
    parser.add_argument('--scan-dir', type=str, required=True,
                        help='Scan output directory (e.g. ~/carla-adversarial/results/collapse_scan/h1_full)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Analysis output directory (default: scan-dir/analysis)')
    args = parser.parse_args()

    scan_dir = os.path.expanduser(args.scan_dir)
    if not os.path.isdir(scan_dir):
        print(f"[FATAL] Scan directory not found: {scan_dir}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(scan_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  PHASE 3B COLLAPSE ANALYSIS")
    print(f"  Scan dir: {scan_dir}")
    print(f"  Output:   {output_dir}")
    print("=" * 70)

    # 1. Load data
    print("\n[1] Loading scan results...")
    conditions = load_scan_results(scan_dir)
    if not conditions:
        print("[FATAL] No conditions found. Check scan directory structure.")
        sys.exit(1)

    # 2. Compute statistics
    print("\n[2] Computing statistics...")
    stats_dict = compute_condition_stats(conditions)
    print(f"  {len(stats_dict)} (group, duty) combinations")

    # 3. Detect collapse points
    print("\n[3] Detecting collapse points...")
    collapse_dict = compute_all_collapse_points(stats_dict)
    for (gid, duty), methods in sorted(collapse_dict.items()):
        cp_str = ', '.join(f'{m}={v:.2f}' if v else f'{m}=None'
                           for m, v in methods.items())
        print(f"  {gid:<25} d={duty:.1f} | {cp_str}")

    # 4. Generate figures
    print("\n[4] Generating figures...")
    fig_paths = []
    fig_paths.append(fig_h1_single_camera(stats_dict, output_dir))
    fig_paths.append(fig_h2_multi_camera(stats_dict, collapse_dict, output_dir))
    fig_paths.append(fig_h3_duty_cycle(stats_dict, output_dir))
    fig_paths.append(fig_collapse_heatmap(stats_dict, collapse_dict, output_dir))
    fig_paths.append(fig_bev_stability(stats_dict, output_dir))
    fig_paths.append(fig_negative_control(stats_dict, output_dir))

    # 5. Generate report
    print("\n[5] Generating report...")
    report = generate_report(stats_dict, collapse_dict, output_dir)

    # 6. Summary
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  Figures: {len([f for f in fig_paths if f])}")
    print(f"  Report:  {os.path.join(output_dir, 'collapse_report.json')}")

    # Print key findings
    h1 = report['hypothesis_tests'].get('H1_front_critical', {})
    if h1.get('most_vulnerable'):
        print(f"\n  H1: Most vulnerable camera = {h1['most_vulnerable']} "
              f"(collapse at {h1['collapse_by_camera'].get(h1['most_vulnerable'], 'N/A')})")

    h2 = report['hypothesis_tests'].get('H2_multi_synergy', {})
    if h2.get('confirmed') is not None:
        status = "CONFIRMED" if h2['confirmed'] else "NOT CONFIRMED"
        print(f"  H2: Multi-camera synergy: {status}")
        print(f"      Single median collapse: {h2.get('median_collapse_single')}")
        print(f"      Dual median collapse:   {h2.get('median_collapse_dual')}")

    print(f"\n  Output directory: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
