#!/usr/bin/env python3
"""
collect_baseline.py — Normal-scenario BEVFormer detection baseline collection

Connects to CARLA, drives Tesla 8-camera layout on multiple routes with autopilot,
and records per-frame detection stats + BEV feature embeddings.

Output:
  results/baseline_stats.json          — overall + per-route summary
  results/baseline_route_N.json        — per-route detailed frame data

Usage:
  # Quick test (1 route, 50 frames, ~2 min):
  python collect_baseline.py --num-routes 1 --frames-per-route 50

  # Full collection (10 routes, 200 frames each, ~2-3 hours):
  python collect_baseline.py --num-routes 10 --frames-per-route 200

  # Custom output directory:
  python collect_baseline.py --num-routes 5 --output-dir ~/carla-adversarial/results
"""

import argparse
import json
import os
import sys
import time
import threading
import numpy as np

# ============================================================================
# Path setup — share modules with run_bevformer_carla.py
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from run_bevformer_carla import (
    setup_bevformer_path,
    load_bevformer_model,
    run_inference,
    decode_results,
    compute_bev_similarity,
)
from carla_bev_adapter import (
    prepare_img_tensor,
    build_img_metas,
    TESLA_TO_NUSCENES,
)
from tesla_camera_layout import TESLA_CAMERAS
from setup_tesla_cameras import spawn_tesla_cameras


# ============================================================================
# Route definition
# ============================================================================

def define_routes(world, num_routes, start_id=0):
    """Generate routes from CARLA map spawn points.

    Each route uses a different spawn point as starting position.
    Autopilot handles the actual driving.

    Args:
        start_id: numeric offset for route IDs (enables resumable collection,
                  e.g. --start-route-id 3 names files route_3.json, ...)

    Returns:
        list[dict]: route configs with 'route_id' and 'spawn_point'
    """
    spawn_points = world.get_map().get_spawn_points()
    n_available = len(spawn_points)
    print(f"  Available spawn points: {n_available}")

    # Shuffle and pick num_routes unique starting points
    indices = np.random.permutation(n_available)[:min(num_routes, n_available)]

    routes = []
    for i, idx in enumerate(indices):
        sp = spawn_points[idx]
        routes.append({
            'route_id': f'route_{start_id + i}',
            'spawn_index': int(idx),
            'spawn_point': sp,
        })
        print(f"  Route {start_id + i}: spawn_point[{idx}] "
              f"loc=({sp.location.x:.1f}, {sp.location.y:.1f}, {sp.location.z:.1f})")

    return routes


# ============================================================================
# Per-frame data collection
# ============================================================================

def _normalize_bev(bev_embed):
    """Flatten a BEV embedding tensor/array to an L2-normalized numpy vector."""
    if bev_embed is None:
        return None
    if hasattr(bev_embed, 'cpu'):
        vec = bev_embed.detach().cpu().numpy().astype(np.float64).flatten()
    else:
        vec = np.array(bev_embed, dtype=np.float64).flatten()
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    return vec


def collect_one_frame(model, images_snapshot, vehicle_transform, velocity,
                      prev_bev_feat, score_thr=0.1):
    """Run inference on one frame and extract stats.

    NOTE: BEVFormer's NMS-free head always emits top-300 slots regardless of
    confidence (config has no score_threshold). We therefore apply our own
    `score_thr` filter here — otherwise det_count is pinned at 300 and the
    mean score is diluted by hundreds of near-zero noise boxes.

    Args:
        model: BEVFormer model (eval mode)
        images_snapshot: dict mapping Tesla camera name → BGR image
        vehicle_transform: CARLA Transform (vehicle global pose)
        velocity: scalar forward speed (m/s)
        prev_bev_feat: previous frame's normalized BEV vector (or None)
        score_thr: confidence threshold to keep detections

    Returns:
        stats: dict with per-frame metrics
        current_bev: normalized BEV vector for next frame's comparison
    """
    img_tensor = prepare_img_tensor(images_snapshot)
    img_metas = build_img_metas(images_snapshot, TESLA_CAMERAS,
                                vehicle_transform, velocity=velocity)

    # Inference — returns (result, bev_embed)
    result, bev_embed = run_inference(model, img_tensor, img_metas)
    decoded = decode_results(result)

    all_scores = np.asarray(decoded['scores_3d'])
    n_raw = int(len(all_scores))
    keep = all_scores > score_thr
    scores = all_scores[keep]
    n_det = int(len(scores))

    # BEV frame-to-frame similarity (cosine + L2 on normalized vectors)
    current_bev = _normalize_bev(bev_embed)
    bev_self_sim = 0.0
    bev_l2_dist = 0.0
    if prev_bev_feat is not None and current_bev is not None:
        bev_self_sim = compute_bev_similarity(prev_bev_feat, current_bev)
        bev_l2_dist = float(np.linalg.norm(current_bev - prev_bev_feat))

    stats = {
        'score_thr': float(score_thr),
        'det_count': n_det,
        'det_count_raw': n_raw,
        'det_score_mean': float(np.mean(scores)) if n_det > 0 else 0.0,
        'det_score_std': float(np.std(scores)) if n_det > 0 else 0.0,
        'det_score_median': float(np.median(scores)) if n_det > 0 else 0.0,
        'det_score_min': float(np.min(scores)) if n_det > 0 else 0.0,
        'det_score_max': float(np.max(all_scores)) if n_raw > 0 else 0.0,
        'det_score_above_thr': float(n_det / max(n_raw, 1)),
        'bev_self_sim': float(bev_self_sim),
        'bev_l2_dist': float(bev_l2_dist),
        'bev_embed_available': bev_embed is not None,
    }
    return stats, current_bev


# ============================================================================
# Single route execution
# ============================================================================

def run_one_route_reuse(model, world, vehicle, cameras, route_config,
                        num_frames, warmup_frames=8, score_thr=0.1):
    """Execute one route with an ALREADY-SPAWNED vehicle + cameras.

    The caller is responsible for teleporting the vehicle to the route start
    and keeping the world in sync mode (enabled ONCE per process by main —
    per-route apply_settings churn and per-route TM port rotation were the
    suspected server-hang triggers). This function:
      1. enables autopilot (fixed TM port 8000, retry on bind error)
      2. attaches camera listeners, warms up
      3. collects `num_frames` frames
      4. stops listeners + autopilot (sync mode stays ON for the next route)

    Args:
        model: BEVFormer model
        world: CARLA world
        vehicle: CARLA vehicle actor (already spawned, already teleported)
        cameras: dict of camera name → CARLA camera sensor (already attached)
        route_config: dict with 'spawn_index'
        num_frames: number of frames to collect (after warmup)
        warmup_frames: frames to skip before collecting (camera warmup)
        score_thr: confidence threshold for detection filtering

    Returns:
        list[dict]: per-frame stats
    """
    import carla

    # Enable autopilot. FIXED TM port for all routes: a different port per
    # route creates a new TrafficManager instance server-side each time
    # (accumulated instances are a known hang vector).
    tm_port = 8000
    for attempt in range(3):
        try:
            vehicle.set_autopilot(True, tm_port)
            break
        except RuntimeError as e:
            if 'bind error' in str(e) and attempt < 2:
                print(f"    TM port {tm_port} busy, trying {tm_port+1}...")
                tm_port += 1
                time.sleep(1.0)
            else:
                raise

    # All world-mutating work runs under try/finally so the world is ALWAYS
    # restored to async mode — a leaked sync-mode server poisons every
    # subsequent route (apply_settings/ticks time out).
    #
    # Frame buffer — accumulating mode (same as run_bevformer_carla.py online)
    frame_buffer = {}
    buffer_lock = threading.Lock()
    frame_stats = []
    skipped = 0
    recoveries = 0

    try:
        def make_callback(name):
            def callback(image):
                arr = np.frombuffer(image.raw_data, dtype=np.uint8)
                arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
                with buffer_lock:
                    frame_buffer[name] = arr.copy()
            return callback

        for name, cam in cameras.items():
            cam.listen(make_callback(name))

        # Sync mode is already ON (enabled once by main). Wait briefly so the
        # newly attached listeners get their first images on the next ticks.
        time.sleep(1.0)

        # Warmup: tick a few frames to let all cameras start streaming
        for i in range(warmup_frames):
            try:
                world.tick()
            except RuntimeError:
                print(f"    Warmup tick {i} failed, retrying...")
                time.sleep(1.0)
                world.tick()
            time.sleep(0.15)

        # Inference loop
        prev_bev = None
        stuck_anchor = None
        stuck_frames = 0
        post_reset = False

        for frame_idx in range(num_frames):
            try:
                world.tick()
            except Exception as e:
                print(f"    Frame {frame_idx}: tick failed: {e}")
                skipped += 1
                continue

            time.sleep(0.2)

            # Snapshot the frame buffer
            with buffer_lock:
                n_received = len(frame_buffer)
                if n_received < 8:
                    skipped += 1
                    continue
                snapshot = {k: v.copy() for k, v in frame_buffer.items()}

            # Get vehicle state
            transform = vehicle.get_transform()
            vel = vehicle.get_velocity()
            speed = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

            # Stuck-vehicle recovery: autopilot often jams after collisions.
            # Position-based (no net displacement > 1.5m within 60 frames
            # ~ 3s sim time) instead of pure velocity, which would
            # false-positive on normal red-light stops at route start.
            loc = transform.location
            if stuck_anchor is None or loc.distance(stuck_anchor) > 1.5:
                stuck_anchor = loc
                stuck_frames = 0
            else:
                stuck_frames += 1
            if stuck_frames >= 60 and recoveries < 5:
                vehicle.set_transform(route_config['spawn_point'])
                for _ in range(12):
                    world.tick()
                with buffer_lock:
                    frame_buffer.clear()  # drop every pre-teleport image
                stuck_anchor = None
                stuck_frames = 0
                prev_bev = None   # don't pollute BEV drift stats with the jump
                recoveries += 1
                post_reset = True
                print(f"    Frame {frame_idx}: vehicle stuck -> teleported "
                      f"to route start (recovery #{recoveries})")

            # Collect frame data
            stats, prev_bev = collect_one_frame(
                model, snapshot, transform, speed, prev_bev, score_thr=score_thr)
            stats['frame_idx'] = frame_idx
            stats['speed'] = float(speed)
            if post_reset:
                # This frame's snapshot predates the teleport: its BEV belongs
                # to the OLD scene. Keeping it as the drift-chain head made the
                # NEXT frame report a fake ~0.5 cosine jump (measured in the
                # 2000-frame baseline). Drop it from the chain and mark it.
                stats['post_reset'] = True
                stats['bev_self_sim'] = 0.0
                stats['bev_l2_dist'] = 0.0
                prev_bev = None
                post_reset = False
            frame_stats.append(stats)

            # Progress report every 20 frames
            if (frame_idx + 1) % 20 == 0:
                print(f"    Frame {frame_idx+1}/{num_frames}: "
                      f"det={stats['det_count']} (raw={stats['det_count_raw']}), "
                      f"score={stats['det_score_mean']:.3f}, "
                      f"max={stats['det_score_max']:.3f}, "
                      f"bev_sim={stats['bev_self_sim']:.6f}, "
                      f"bev_l2={stats['bev_l2_dist']:.4f}, "
                      f"speed={speed:.1f}m/s")
    finally:
        # Cleanup: stop cameras + autopilot, restore async mode — runs even
        # if an exception propagated out of the loop above
        for cam in cameras.values():
            try:
                cam.stop()
            except Exception:
                pass
        try:
            vehicle.set_autopilot(False, tm_port)
        except Exception:
            pass
        time.sleep(0.5)
        # NOTE: sync mode intentionally stays ON — main restores async once
        # after ALL routes finish (per-route mode switching removed).

    if skipped > 0:
        print(f"    Skipped {skipped} frames (buffer incomplete or tick error)")
    if recoveries > 0:
        print(f"    Stuck-vehicle recoveries: {recoveries}")

    return frame_stats


# ============================================================================
# Statistics computation
# ============================================================================

def compute_route_summary(frame_stats):
    """Compute summary statistics for a single route.

    Returns:
        dict: summary metrics
    """
    if not frame_stats:
        return {
            'num_frames': 0,
            'det_count_mean': 0.0, 'det_count_std': 0.0,
            'det_score_mean': 0.0, 'det_score_std': 0.0,
            'det_score_max_mean': 0.0,
            'bev_cosine_mean': 0.0, 'bev_cosine_std': 0.0, 'bev_cosine_min': 0.0,
            'bev_l2_mean': 0.0, 'bev_l2_max': 0.0,
            'speed_mean': 0.0, 'speed_max': 0.0,
        }

    det_counts = [s['det_count'] for s in frame_stats]
    det_scores = [s['det_score_mean'] for s in frame_stats]
    det_score_maxs = [s['det_score_max'] for s in frame_stats]
    bev_sims = [s['bev_self_sim'] for s in frame_stats if s['bev_self_sim'] > 0]
    bev_l2s = [s['bev_l2_dist'] for s in frame_stats if s['bev_l2_dist'] > 0]
    speeds = [s['speed'] for s in frame_stats]

    return {
        'num_frames': len(frame_stats),
        'score_thr': frame_stats[0].get('score_thr', 0.1),
        'det_count_mean': float(np.mean(det_counts)),
        'det_count_std': float(np.std(det_counts)),
        'det_count_min': int(np.min(det_counts)),
        'det_count_max': int(np.max(det_counts)),
        'det_count_raw_mean': float(np.mean([s['det_count_raw'] for s in frame_stats])),
        'det_score_mean': float(np.mean(det_scores)),
        'det_score_std': float(np.std(det_scores)),
        'det_score_median': float(np.median(det_scores)),
        'det_score_max_mean': float(np.mean(det_score_maxs)),
        'bev_cosine_mean': float(np.mean(bev_sims)) if bev_sims else 0.0,
        'bev_cosine_std': float(np.std(bev_sims)) if bev_sims else 0.0,
        'bev_cosine_min': float(np.min(bev_sims)) if bev_sims else 0.0,
        'bev_l2_mean': float(np.mean(bev_l2s)) if bev_l2s else 0.0,
        'bev_l2_max': float(np.max(bev_l2s)) if bev_l2s else 0.0,
        'speed_mean': float(np.mean(speeds)),
        'speed_max': float(np.max(speeds)),
    }


def compute_overall_summary(route_summaries):
    """Compute cross-route aggregate statistics.

    Returns:
        dict: overall summary
    """
    if not route_summaries:
        return {}

    return {
        'num_routes': len(route_summaries),
        'total_frames': sum(r['num_frames'] for r in route_summaries),
        'score_thr': route_summaries[0].get('score_thr', 0.1),
        'det_count_mean': float(np.mean([r['det_count_mean'] for r in route_summaries])),
        'det_count_std': float(np.std([r['det_count_mean'] for r in route_summaries])),
        'det_score_mean': float(np.mean([r['det_score_mean'] for r in route_summaries])),
        'det_score_std': float(np.std([r['det_score_mean'] for r in route_summaries])),
        'det_score_max_mean': float(np.mean([r['det_score_max_mean'] for r in route_summaries])),
        'bev_cosine_mean': float(np.mean([r['bev_cosine_mean'] for r in route_summaries])),
        'bev_cosine_std': float(np.std([r['bev_cosine_mean'] for r in route_summaries])),
        'bev_l2_mean': float(np.mean([r['bev_l2_mean'] for r in route_summaries])),
        'bev_l2_std': float(np.std([r['bev_l2_mean'] for r in route_summaries])),
        'speed_mean': float(np.mean([r['speed_mean'] for r in route_summaries])),
    }


def compute_robust_thresholds(all_route_frames, route_ids=None, persistence=15):
    """Robust frame-level attack thresholds, replacing route-mean mu+-3sigma.

    Rationale (measured on the 2000-frame baseline): the per-frame BEV
    transition distribution is strongly bimodal — clean driving sits at
    cos ~0.99985-0.99999 (frame std ~2e-5), while ~12 scene-jump/chimera
    frames at cos ~0.5 poison any mean/std estimate, making mu-3sigma both
    arbitrary and ~0.6% false-positive per frame.

    Method:
      1. Pool all valid transitions (sim > 0), exclude post-reset frames.
      2. Calibrate on the normal-driving core only (cos >= 0.99); scene-jump
         frames are counted and reported, not fitted.
      3. Threshold = median -+ 3 * 1.4826 * MAD (outlier-immune), clamped by
         the core distribution's 0.5/99.5 percentiles so it can never be
         tighter than 1-in-200 natural variation.
      4. Report empirical false positives for single-frame AND 3-consecutive-
         frame AND-rule alarm logic (the monitor requires persistence).
    """
    cos_vals, l2_vals = [], []
    jump_count = 0

    for frames in all_route_frames:
        for f in frames:
            c = f['bev_self_sim']
            if c <= 0 or f.get('post_reset', False):
                continue  # first frame / chain-restart marker: no comparison
            if c < 0.99:
                jump_count += 1
                continue
            cos_vals.append(c)
            l2_vals.append(f['bev_l2_dist'])

    cos_arr = np.asarray(cos_vals)
    l2_arr = np.asarray(l2_vals)
    if cos_arr.size == 0:
        return {}

    def _robust(x, sign):
        med = float(np.median(x))
        mad = 1.4826 * float(np.median(np.abs(x - med)))
        if sign < 0:  # cosine floor
            raw = med - 3 * mad
            guard = float(np.percentile(x, 0.5))
            return max(raw, guard)
        raw = med + 3 * mad  # L2 ceiling
        guard = float(np.percentile(x, 99.5))
        return min(raw, guard)

    cos_thr = _robust(cos_arr, -1)
    l2_thr = _robust(l2_arr, +1)

    # Empirical false positives (single-frame and N-frame persistence)
    fp_single = 0
    fp_persist = 0
    n_valid = 0
    persist_events = []
    for ri, frames in enumerate(all_route_frames):
        rid = (route_ids[ri] if route_ids and ri < len(route_ids)
               else f'route_{ri}')
        alarms = []
        for f in frames:
            c = f['bev_self_sim']
            if c <= 0 or f.get('post_reset', False):
                continue
            n_valid += 1
            alarms.append((c < cos_thr and f['bev_l2_dist'] > l2_thr,
                           f['frame_idx']))
        fp_single += sum(a for a, _ in alarms)
        run = []
        for a, fi in alarms:
            if a:
                run.append(fi)
                if len(run) == persistence:
                    fp_persist += 1
                    persist_events.append([rid, run[0]])
                    run = []
            else:
                run = []

    print(f"\n  Robust threshold calibration ({cos_arr.size} core frames, "
          f"{jump_count} scene-jump frames excluded):")
    print(f"    Attack threshold (BEV cosine <): {cos_thr:.6f}")
    print(f"    Attack threshold (BEV L2 dist >): {l2_thr:.6f}")
    print(f"    False positives single-frame: {fp_single} "
          f"({100.0 * fp_single / max(n_valid, 1):.2f}%)")
    print(f"    False positives {persistence}-frame persistence: {fp_persist}")

    return {
        'attack_threshold_bev_cosine': cos_thr,
        'attack_threshold_bev_l2': l2_thr,
        'threshold_rule': (f'BEV cosine < {cos_thr:.6f} AND L2 > {l2_thr:.6f} '
                           f'on {persistence} consecutive frames '
                           '(robust median+-3*MAD on normal-driving core)'),
        'threshold_false_positives_single_frame': int(fp_single),
        'threshold_false_positives_persisted': int(fp_persist),
        'threshold_false_positive_events': persist_events,
        'scene_jump_frames_excluded': int(jump_count),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect BEVFormer detection baseline in CARLA')
    parser.add_argument('--num-routes', type=int, default=10,
                        help='Number of routes to collect (default: 10)')
    parser.add_argument('--start-route-id', type=int, default=0,
                        help='Numeric offset for route IDs, for resumable '
                             'collection (e.g. 3 continues route_3.json+ '
                             'after a crash; final stats merge ALL routes '
                             'found on disk)')
    parser.add_argument('--frames-per-route', type=int, default=200,
                        help='Frames per route after warmup (default: 200)')
    parser.add_argument('--warmup-frames', type=int, default=8,
                        help='Warmup frames before collecting (default: 8)')
    parser.add_argument('--score-thr', type=float, default=0.1,
                        help='Confidence threshold for counting detections '
                             '(BEVFormer NMS-free head always emits 300 slots; '
                             'default: 0.1)')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.expanduser('~/carla-adversarial/results'))
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--config', type=str, default=None,
                        help='BEVFormer config file path')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='BEVFormer checkpoint file path')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load model ----
    print("=" * 60)
    print("BEVFormer Baseline Collection")
    print("=" * 60)

    setup_bevformer_path()
    model, cfg = load_bevformer_model(args.config, args.checkpoint)

    # ---- Connect to CARLA ----
    import carla
    print(f"\nConnecting to CARLA at {args.host}:{args.port}...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    print(f"  Connected. Map: {world.get_map().name}")

    # ---- Recover server to a clean state ----
    # Previous crashed runs may leave the server in sync mode with zombie
    # actors. Restore async mode FIRST (actor RPCs can hang while the server
    # waits for ticks in sync mode), then clear leftover actors.
    settings = world.get_settings()
    if settings.synchronous_mode:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print("  Restored async mode (server was stuck in sync mode)")
        time.sleep(0.5)

    print("\nClearing leftover actors (sensors/vehicles/walkers)...")
    leftovers = [a for a in world.get_actors()
                 if a.type_id.startswith(('sensor.', 'vehicle.', 'walker.'))]
    # Destroy highest IDs first (attached sensors have higher IDs than parents)
    for actor in sorted(leftovers, key=lambda a: a.id, reverse=True):
        try:
            actor.destroy()
        except Exception:
            pass
    print(f"  Attempted cleanup of {len(leftovers)} actors")
    time.sleep(1.0)

    # ---- Define routes ----
    print(f"\nDefining {args.num_routes} routes "
          f"(route_{args.start_route_id}...):")
    routes = define_routes(world, args.num_routes, start_id=args.start_route_id)

    # ---- Spawn vehicle + cameras ONCE ----
    print("\nSpawning vehicle and attaching 8 cameras...")
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
    vehicle = world.spawn_actor(vehicle_bp, routes[0]['spawn_point'])
    print(f"  Vehicle spawned: {vehicle.type_id}")
    time.sleep(1.0)

    cameras = spawn_tesla_cameras(vehicle, world)
    print(f"  {len(cameras)} cameras attached")

    # ---- Enable sync mode ONCE for the whole run ----
    # (Per-route apply_settings switching was a suspected server-hang trigger.)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    print("  Synchronous mode enabled for entire run (dt=0.05s)")

    # ---- Run each route ----
    all_route_summaries = []

    for route_idx, route in enumerate(routes):
        print(f"\n{'=' * 60}")
        print(f"Route: {route['route_id']} "
              f"(spawn_index={route['spawn_index']}, "
              f"frames={args.frames_per_route})")
        print(f"{'=' * 60}")

        # Teleport vehicle to route start (sync mode: transform applies on
        # the next ticks — settle with a few idle ticks)
        try:
            vehicle.set_transform(route['spawn_point'])
            for _ in range(8):
                world.tick()

            # Run route (autopilot, listeners, warmup, inference, cleanup)
            frame_stats = run_one_route_reuse(
                model, world, vehicle, cameras, route,
                num_frames=args.frames_per_route,
                warmup_frames=args.warmup_frames,
                score_thr=args.score_thr,
            )
        except Exception as e:
            print(f"  [ERROR] {route['route_id']} aborted: {e}")
            print("  Pausing 10s, then continuing with next route...")
            time.sleep(10.0)
            continue

        # Compute route summary
        summary = compute_route_summary(frame_stats)
        summary['route_id'] = route['route_id']
        summary['spawn_index'] = route['spawn_index']
        all_route_summaries.append(summary)

        print(f"\n  Route summary:")
        print(f"    Frames: {summary['num_frames']}")
        print(f"    Det count (score>{summary['score_thr']:.2f}): "
              f"{summary['det_count_mean']:.1f} "
              f"± {summary['det_count_std']:.1f} "
              f"[{summary['det_count_min']}, {summary['det_count_max']}]")
        print(f"    Det score (kept): {summary['det_score_mean']:.3f} "
              f"± {summary['det_score_std']:.3f} "
              f"(raw max: {summary['det_score_max_mean']:.3f})")
        print(f"    BEV cosine: {summary['bev_cosine_mean']:.6f} "
              f"± {summary['bev_cosine_std']:.6f} "
              f"(min={summary['bev_cosine_min']:.6f})")
        print(f"    BEV L2 dist: {summary['bev_l2_mean']:.4f} "
              f"(max={summary['bev_l2_max']:.4f})")
        print(f"    Speed: {summary['speed_mean']:.1f} m/s "
              f"(max={summary['speed_max']:.1f})")

        # Save per-route data
        route_path = os.path.join(
            args.output_dir, f"baseline_{route['route_id']}.json")
        with open(route_path, 'w') as f:
            json.dump({
                'route': route['route_id'],
                'spawn_index': route['spawn_index'],
                'num_frames': summary['num_frames'],
                'summary': summary,
                'frames': frame_stats,
            }, f, indent=2)
        print(f"  Saved: {route_path}")

    # ---- Final cleanup ----
    print(f"\nCleaning up vehicle and cameras...")
    for cam in cameras.values():
        try:
            cam.stop()
        except Exception:
            pass
        cam.destroy()
    try:
        vehicle.set_autopilot(False)
    except Exception:
        pass
    vehicle.destroy()

    # Restore async mode (sync was enabled once for the whole run)
    try:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
    except Exception as e:
        print(f"  [WARN] could not restore async mode: {e}")

    # ---- Merge ALL route files on disk ----
    # Supports resumable collection: stats always aggregate every completed
    # route in the output dir, not just the ones from this process run.
    import glob
    import re
    print(f"\nMerging all baseline_route_*.json in {args.output_dir}...")
    merged = []
    for path in glob.glob(os.path.join(args.output_dir, 'baseline_route_*.json')):
        m = re.search(r'baseline_route_(\d+)\.json$', path)
        if not m:
            continue
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            s = data['summary']
            s.setdefault('route_id', f"route_{m.group(1)}")
            merged.append((int(m.group(1)), s, data.get('frames', [])))
        except Exception as e:
            print(f"  [WARN] skipping {os.path.basename(path)}: {e}")
    merged.sort(key=lambda t: t[0])
    all_route_summaries = [s for _, s, _ in merged]
    all_route_frames = [fr for _, _, fr in merged]
    print(f"  Found {len(all_route_summaries)} complete routes: "
          f"{[s['route_id'] for s in all_route_summaries]}")

    # ---- Overall summary ----
    overall = compute_overall_summary(all_route_summaries)

    print(f"\n{'=' * 60}")
    print("OVERALL BASELINE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Routes: {overall['num_routes']}, "
          f"Total frames: {overall['total_frames']}")
    print(f"  Score threshold: {overall['score_thr']:.2f}")
    print(f"  Det count: {overall['det_count_mean']:.1f} "
          f"± {overall['det_count_std']:.1f}")
    print(f"  Det score (kept): {overall['det_score_mean']:.3f} "
          f"± {overall['det_score_std']:.3f} "
          f"(raw max: {overall['det_score_max_mean']:.3f})")
    print(f"  BEV cosine: {overall['bev_cosine_mean']:.6f} "
          f"± {overall['bev_cosine_std']:.6f}")
    print(f"  BEV L2 dist: {overall['bev_l2_mean']:.4f} "
          f"± {overall['bev_l2_std']:.4f}")
    print(f"  Speed: {overall['speed_mean']:.1f} m/s")

    # Attack detection thresholds — robust frame-level calibration.
    # (The old route-mean μ±3σ method was superseded: the frame distribution
    # is bimodal, its std is dominated by ~12 scene-jump outliers, and the
    # resulting thresholds yielded ~0.6% single-frame false positives.)
    overall['legacy_threshold_bev_cosine'] = float(
        overall['bev_cosine_mean'] - 3 * overall['bev_cosine_std'])
    overall['legacy_threshold_bev_l2'] = float(
        overall['bev_l2_mean'] + 3 * overall['bev_l2_std'])
    if all_route_frames:
        overall.update(compute_robust_thresholds(
            all_route_frames,
            route_ids=[s.get('route_id', '') for s in all_route_summaries]))

    # Save overall summary
    overall_path = os.path.join(args.output_dir, 'baseline_stats.json')
    with open(overall_path, 'w') as f:
        json.dump({
            'overall': overall,
            'per_route': all_route_summaries,
        }, f, indent=2)
    print(f"\n  Saved overall summary to {overall_path}")
    print(f"\n[DONE] Baseline collection complete.")


if __name__ == '__main__':
    main()
