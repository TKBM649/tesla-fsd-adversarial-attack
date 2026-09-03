#!/usr/bin/env python3
"""
run_collapse_experiment.py -- Phase 3B CLI entry point

Orchestrates the collapse point scanning experiment:
  1. Load BEVFormer model
  2. Connect to CARLA, spawn vehicle + cameras
  3. Run collapse scanning across experiment matrix
  4. Save results and progress

Usage:
  # Quick H1 validation (24 conditions, ~2h)
  python run_collapse_experiment.py --mode h1_quick

  # Full H1 scan (64 conditions, ~8h)
  python run_collapse_experiment.py --mode h1_full

  # Full H2 scan (120 conditions, ~15h)
  python run_collapse_experiment.py --mode h2_full

  # H3 duty cycle scan (60 conditions, ~8h)
  python run_collapse_experiment.py --mode h3_full

  # Full matrix (600 conditions, ~50h)
  python run_collapse_experiment.py --mode full

  # Resume interrupted scan
  python run_collapse_experiment.py --mode h1_full --resume

  # Custom parameters
  python run_collapse_experiment.py --mode h1_quick --num-routes 5 --frames-per-route 150
"""

import argparse
import json
import os
import sys
import time

# ---- Project imports ----
from run_bevformer_carla import setup_bevformer_path
from collapse_configs import (
    EXPERIMENT_MODES, generate_experiment_matrix,
    condition_to_dirname, NUM_ROUTES_DEFAULT, FRAMES_PER_ROUTE_DEFAULT,
)
from collapse_point_scanner import (
    CollapsePointScanner, setup_carla_environment,
    teardown_carla_environment,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Phase 3B: Collapse point scanning experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiment modes:
  h1_quick   8 single cameras x 3 intensities x 5 routes    (~2h)
  h1_full    8 single cameras x 8 intensities x 10 routes   (~8h)
  h2_full    15 groups x 8 intensities x 10 routes          (~15h)
  h3_full    3 cameras x 4 intensities x 5 duty cycles      (~8h)
  full       15 groups x 8 intensities x 5 duty cycles      (~50h)
        """)

    # Experiment mode
    parser.add_argument('--mode', type=str, required=True,
                        choices=list(EXPERIMENT_MODES.keys()),
                        help='Experiment mode (see below for details)')

    # Route parameters
    parser.add_argument('--num-routes', type=int, default=None,
                        help='Routes per condition (default: from mode)')
    parser.add_argument('--frames-per-route', type=int, default=None,
                        help='Frames per route (default: 200)')
    parser.add_argument('--warmup-frames', type=int, default=8)
    parser.add_argument('--score-thr', type=float, default=0.05)

    # Output
    parser.add_argument('--output-dir', type=str,
                        default=os.path.expanduser(
                            '~/carla-adversarial/results/collapse_scan'))

    # Resume
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume from last completed condition')
    parser.add_argument('--no-resume', action='store_true', default=False,
                        help='Ignore previous progress, start fresh')

    # CARLA connection
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=2000)

    # BEVFormer
    parser.add_argument('--config', type=str, default=None,
                        help='BEVFormer config file path')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='BEVFormer checkpoint file path')

    # Dry run
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Print experiment plan without running')

    return parser.parse_args()


def print_experiment_plan(conditions, mode_info, output_dir):
    """Print the experiment plan summary."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT PLAN")
    print("=" * 70)
    print(f"  Mode:           {mode_info['mode']}")
    print(f"  Description:    {EXPERIMENT_MODES.get(mode_info['mode'], {}).get('description', 'custom')}")
    print(f"  Camera groups:  {mode_info['num_groups']}")
    print(f"  Intensities:    {mode_info['num_intensities']}")
    print(f"  Duty cycles:    {mode_info['num_duty_cycles']}")
    print(f"  Conditions:     {mode_info['num_conditions']}")
    print(f"  Routes/cond:    {mode_info['num_routes']}")
    print(f"  Total routes:   {mode_info['total_routes']}")
    print(f"  Output:         {output_dir}")

    est_minutes = mode_info['total_routes'] * 1.2  # ~1.2 min per route
    print(f"  Est. time:      ~{est_minutes:.0f} min ({est_minutes/60:.1f} h)")
    print()

    # Print condition list
    print(f"  {'#':>4}  {'Group':<25}  {'Intensity':>9}  {'Duty':>5}  {'Dirname'}")
    print("  " + "-" * 80)
    for idx, (gid, inten, duty) in enumerate(conditions):
        dname = condition_to_dirname(gid, inten, duty)
        print(f"  {idx+1:>4}  {gid:<25}  {inten:>9.2f}  {duty:>5.1f}  {dname}")
    print()


def main():
    args = parse_args()

    # ---- Resolve mode defaults ----
    mode = EXPERIMENT_MODES[args.mode]
    num_routes = args.num_routes or mode['num_routes']
    frames_per_route = args.frames_per_route or FRAMES_PER_ROUTE_DEFAULT

    # ---- Generate experiment matrix ----
    conditions, mode_info = generate_experiment_matrix(mode=args.mode)
    mode_info['num_routes'] = num_routes
    mode_info['total_routes'] = mode_info['num_conditions'] * num_routes

    # Output directory: output_dir/mode_name/
    output_dir = os.path.join(args.output_dir, args.mode)

    # ---- Print plan ----
    print_experiment_plan(conditions, mode_info, output_dir)

    if args.dry_run:
        print("[DRY RUN] Exiting without running.")
        return

    # ---- Confirm ----
    print(f"About to run {mode_info['num_conditions']} conditions x "
          f"{num_routes} routes = {mode_info['total_routes']} total routes.")
    print("Press Ctrl+C within 5 seconds to abort...")
    time.sleep(5)

    # ---- Setup BEVFormer ----
    setup_bevformer_path()
    print("=" * 60)
    print("Loading BEVFormer model...")
    from run_bevformer_carla import load_bevformer_model
    model, cfg = load_bevformer_model(args.config, args.checkpoint)
    print("[DONE] Model loaded")

    # ---- Setup CARLA ----
    client, world, vehicle, cameras, routes = setup_carla_environment(
        host=args.host, port=args.port,
        num_routes=max(num_routes, 10),  # need enough spawn points
    )

    # ---- Create scanner ----
    scanner = CollapsePointScanner(
        world, vehicle, cameras, model,
        warmup_frames=args.warmup_frames,
        score_thr=args.score_thr,
    )
    scanner.attach_callbacks()

    # ---- Save experiment metadata ----
    os.makedirs(output_dir, exist_ok=True)
    meta = {
        'mode': args.mode,
        'mode_info': mode_info,
        'conditions': [(g, i, d) for g, i, d in conditions],
        'warmup_frames': args.warmup_frames,
        'frames_per_route': frames_per_route,
        'score_thr': args.score_thr,
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    meta_path = os.path.join(output_dir, 'experiment_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Experiment metadata saved: {meta_path}")

    # ---- Run scan ----
    resume = args.resume and not args.no_resume
    t_start = time.time()

    try:
        results = scanner.scan_batch(
            conditions, output_dir,
            num_routes=num_routes,
            frames_per_route=frames_per_route,
            resume=resume,
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Scan paused. Progress saved.")
        print(f"Resume with: python run_collapse_experiment.py "
              f"--mode {args.mode} --resume")
        results = []
    except Exception as e:
        print(f"\n[FATAL] {e}")
        results = []
        raise
    finally:
        # ---- Teardown ----
        teardown_carla_environment(vehicle, cameras)

        # Save end time
        elapsed = time.time() - t_start
        meta['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        meta['elapsed_seconds'] = elapsed
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"\nTotal time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        print(f"Results in: {output_dir}")

    # ---- Final summary ----
    if results:
        print("\n" + "=" * 70)
        print("  SCAN SUMMARY")
        print("=" * 70)
        for r in results:
            all_det = []
            for rd in r['per_route']:
                all_det.extend(rd['det_count'])
            import numpy as np
            print(f"  {r['group_id']:<25} i={r['intensity']:.2f} "
                  f"d={r['duty_cycle']:.1f} | "
                  f"det median={np.median(all_det):.1f} "
                  f"max={np.max(all_det)}")

    print(f"\n[DONE] Collapse scan complete.")
    print(f"Next: python analyze_collapse_points.py --scan-dir {output_dir}")


if __name__ == '__main__':
    main()
