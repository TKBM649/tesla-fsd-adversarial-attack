#!/usr/bin/env python3
"""
collect_attack.py — 对抗攻击采集脚本（Phase 2 Step 3）

以 collect_baseline.py 为模板，集成 AttackInjector，在帧循环中注入攻击。

核心改动：
- 新增 --attack-config 参数（场景名或 JSON 文件路径）
- 输出隔离：attack_{scenario_id}/ 独立目录
- 帧统计扩展：每帧记录 attack_active + active_cameras
- 继承全部防崩资产（单同步模式、TM 8000、断点续采、post_reset 标记）
"""

import argparse
import json
import os
import sys
import time
import threading
import numpy as np

# ---- PIL import (optional, for image saving) ----
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ---- CARLA imports ----
try:
    import carla
except ImportError:
    print("[FATAL] carla module not found. Make sure CARLA Python is installed.")
    sys.exit(1)

# ---- Project imports ----
from run_bevformer_carla import run_inference, compute_bev_similarity, decode_results, setup_bevformer_path, load_bevformer_model
from carla_bev_adapter import (
    prepare_img_tensor, build_img_metas, draw_3d_detections_on_images,
    NUSCENES_CAM_ORDER, CLASS_NAMES, TESLA_TO_NUSCENES,
)
from tesla_camera_layout import TESLA_CAMERAS
from setup_tesla_cameras import spawn_tesla_cameras
from attack_configs import AttackConfig, ATTACK_SCENARIOS, load_json
from attack_injector import AttackInjector


# ============================================================================
# Route definitions (same as collect_baseline.py)
# ============================================================================

def define_routes(world, num_routes=10, start_id=0):
    """Define routes with random spawn points from the map."""
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
# Frame collection (same as collect_baseline.py, with attack injection added)
# ============================================================================

def collect_one_frame(model, images_snapshot, vehicle_transform, velocity,
                      prev_bev_feat, score_thr=0.1):
    """Run inference on one frame and extract stats."""
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

    # BEV frame-to-frame similarity
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


def _normalize_bev(bev_embed):
    """Normalize BEV embedding vector."""
    if bev_embed is None:
        return None
    if hasattr(bev_embed, 'cpu'):
        bev_embed = bev_embed.detach().cpu().numpy()
    bev_flat = np.asarray(bev_embed).flatten()
    norm = np.linalg.norm(bev_flat)
    if norm < 1e-8:
        return None
    return bev_flat / norm


# ============================================================================
# Route execution (with attack injection)
# ============================================================================

def save_comparison_frame(original_snapshot, attacked_snapshot, frame_idx,
                          save_dir, route_id, attack_meta, cameras_to_save=None):
    """Save side-by-side comparison grid: original (top) vs attacked (bottom).

    Creates a single PNG with:
    - Top row: original camera images
    - Bottom row: attacked camera images
    - Labels and attack metadata overlay
    """
    if not HAS_PIL:
        return

    if cameras_to_save is None:
        # Default: save front 3 cameras for comparison
        cameras_to_save = ['front_main', 'front_wide', 'side_front_left']

    # Filter to cameras that exist in both snapshots
    cams = [c for c in cameras_to_save
            if c in original_snapshot and c in attacked_snapshot]
    if not cams:
        return

    n_cams = len(cams)
    # Get image dimensions from first available camera
    sample = original_snapshot[cams[0]]
    h, w = sample.shape[:2]

    # Scale down for manageable file sizes (half resolution)
    scale = 0.5
    sh, sw = int(h * scale), int(w * scale)

    # Grid: n_cams columns × 2 rows (original + attacked)
    gap = 4
    grid_w = n_cams * sw + (n_cams - 1) * gap
    grid_h = 2 * sh + gap + 30  # 30px for labels
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    pil_grid = Image.fromarray(grid)
    draw = ImageDraw.Draw(pil_grid)

    # Try to get a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_small = font

    # Row labels
    attack_active = attack_meta.get('attack_active', False)
    draw.text((5, 2), f"Route: {route_id}  Frame: {frame_idx}  Attack: {'ON' if attack_active else 'OFF'}",
              fill=(255, 255, 255), font=font)

    for col_idx, cam_name in enumerate(cams):
        x0 = col_idx * (sw + gap)

        # Original (top row)
        orig_img = original_snapshot[cam_name]
        orig_resized = Image.fromarray(orig_img).resize((sw, sh), Image.BILINEAR)
        pil_grid.paste(orig_resized, (x0, 25))

        # Attacked (bottom row)
        atkd_img = attacked_snapshot[cam_name]
        atkd_resized = Image.fromarray(atkd_img).resize((sw, sh), Image.BILINEAR)
        pil_grid.paste(atkd_resized, (x0, 25 + sh + gap))

        # Camera name label
        is_target = cam_name in attack_meta.get('active_cameras', [])
        label_color = (255, 100, 100) if is_target else (200, 200, 200)
        draw.text((x0 + 2, sh + gap + 25 + 2), cam_name,
                  fill=label_color, font=font_small)

    # Save
    fname = f"{route_id}_frame{frame_idx:04d}_comparison.png"
    fpath = os.path.join(save_dir, fname)
    pil_grid.save(fpath, optimize=True)


def run_one_route_reuse(world, vehicle, cameras, model, route_config,
                        num_frames, warmup_frames, score_thr, injector,
                        frame_buffer, buffer_lock,
                        save_images=False, save_interval=20, image_save_dir=None):
    """Execute one route with attack injection."""
    # Clear frame buffer for this route
    with buffer_lock:
        frame_buffer.clear()

    # Warmup
    print(f"  Warming up {warmup_frames} frames...")
    for _ in range(warmup_frames):
        world.tick()
        time.sleep(0.05)

    # Collect frames
    frame_stats = []
    prev_bev = None
    stuck_anchor = None
    stuck_frames = 0
    recoveries = 0
    post_reset = False
    skipped = 0

    print(f"  Collecting {num_frames} frames...")
    for frame_idx in range(num_frames):
        world.tick()

        # Snapshot the frame buffer
        with buffer_lock:
            n_received = len(frame_buffer)
            if n_received < 8:
                skipped += 1
                continue
            snapshot = {k: v.copy() for k, v in frame_buffer.items()}

        # >>> ATTACK INJECTION POINT <<<
        # Save original snapshot BEFORE attack (for comparison)
        if save_images and image_save_dir and (frame_idx % save_interval == 0):
            original_snapshot = {k: v.copy() for k, v in snapshot.items()}

        snapshot, attack_meta = injector.apply(snapshot, frame_idx)

        # Save comparison image AFTER attack
        if save_images and image_save_dir and (frame_idx % save_interval == 0):
            save_comparison_frame(
                original_snapshot, snapshot, frame_idx,
                image_save_dir, route_config['route_id'], attack_meta)
            del original_snapshot  # free memory

        # Get vehicle state
        transform = vehicle.get_transform()
        vel = vehicle.get_velocity()
        speed = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # Stuck-vehicle recovery
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
                frame_buffer.clear()
            stuck_anchor = None
            stuck_frames = 0
            prev_bev = None
            recoveries += 1
            post_reset = True
            print(f"    Frame {frame_idx}: vehicle stuck -> teleported "
                  f"(recovery #{recoveries})")

        # Collect frame data
        stats, prev_bev = collect_one_frame(
            model, snapshot, transform, speed, prev_bev, score_thr=score_thr)
        stats['frame_idx'] = frame_idx
        stats['speed'] = float(speed)

        # >>> RECORD ATTACK METADATA <<<
        stats['attack_active'] = attack_meta['attack_active']
        stats['active_cameras'] = attack_meta['active_cameras']

        if post_reset:
            stats['post_reset'] = True
            stats['bev_self_sim'] = 0.0
            stats['bev_l2_dist'] = 0.0
            prev_bev = None
            post_reset = False

        frame_stats.append(stats)

    if skipped > 0:
        print(f"    Skipped {skipped} frames (incomplete camera buffer)")

    # Compute summary
    summary = compute_route_summary(frame_stats)
    summary['recoveries'] = recoveries

    return summary, frame_stats


def camera_callback_with_name(image, cam_name, frame_buffer, buffer_lock):
    """Camera callback that stores image in frame_buffer."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))[:, :, :3]  # RGBA -> RGB
    with buffer_lock:
        frame_buffer[cam_name] = arr


def compute_route_summary(frame_stats):
    """Compute per-route summary statistics."""
    if not frame_stats:
        return {'num_frames': 0}

    n = len(frame_stats)
    det_counts = [f['det_count'] for f in frame_stats]
    bev_sims = [f['bev_self_sim'] for f in frame_stats if f['bev_self_sim'] > 0]
    bev_l2s = [f['bev_l2_dist'] for f in frame_stats if f['bev_l2_dist'] > 0]
    speeds = [f['speed'] for f in frame_stats]

    summary = {
        'num_frames': n,
        'det_count_mean': float(np.mean(det_counts)),
        'det_count_std': float(np.std(det_counts)),
        'det_count_min': int(np.min(det_counts)),
        'det_count_max': int(np.max(det_counts)),
        'bev_cosine_mean': float(np.mean(bev_sims)) if bev_sims else 0.0,
        'bev_cosine_std': float(np.std(bev_sims)) if bev_sims else 0.0,
        'bev_cosine_min': float(np.min(bev_sims)) if bev_sims else 0.0,
        'bev_l2_mean': float(np.mean(bev_l2s)) if bev_l2s else 0.0,
        'bev_l2_std': float(np.std(bev_l2s)) if bev_l2s else 0.0,
        'bev_l2_max': float(np.max(bev_l2s)) if bev_l2s else 0.0,
        'speed_mean': float(np.mean(speeds)),
        'speed_max': float(np.max(speeds)),
    }

    # Attack statistics
    attack_frames = [f for f in frame_stats if f.get('attack_active')]
    summary['attack_frames'] = len(attack_frames)
    summary['attack_frame_ratio'] = len(attack_frames) / n if n > 0 else 0.0

    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Collect BEVFormer attack experiment in CARLA')
    parser.add_argument('--num-routes', type=int, default=10,
                        help='Number of routes to collect (default: 10)')
    parser.add_argument('--start-route-id', type=int, default=0,
                        help='Numeric offset for route IDs')
    parser.add_argument('--frames-per-route', type=int, default=200,
                        help='Frames per route after warmup (default: 200)')
    parser.add_argument('--warmup-frames', type=int, default=8,
                        help='Warmup frames before collecting (default: 8)')
    parser.add_argument('--score-thr', type=float, default=0.05,
                        help='Confidence threshold for counting detections (default: 0.05)')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.expanduser('~/carla-adversarial/results'))
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--config', type=str, default=None,
                        help='BEVFormer config file path')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='BEVFormer checkpoint file path')

    # >>> ATTACK CONFIG PARAMETER <<<
    parser.add_argument('--attack-config', type=str, required=True,
                        help='Attack scenario name (from ATTACK_SCENARIOS) or JSON file path')

    # >>> IMAGE SAVING PARAMETERS <<<
    parser.add_argument('--save-images', action='store_true', default=False,
                        help='Save camera images for before/after attack comparison')
    parser.add_argument('--save-interval', type=int, default=20,
                        help='Save comparison images every N frames (default: 20)')

    args = parser.parse_args()

    # ---- Load attack config ----
    if args.attack_config in ATTACK_SCENARIOS:
        attack_cfg = ATTACK_SCENARIOS[args.attack_config]
        print(f"Loaded attack scenario: {attack_cfg.attack_id}")
    elif os.path.isfile(args.attack_config):
        attack_cfg = load_json(args.attack_config)
        print(f"Loaded attack config from file: {args.attack_config}")
    else:
        print(f"[FATAL] --attack-config '{args.attack_config}' not found")
        sys.exit(1)

    injector = AttackInjector(attack_cfg)

    # Output directory: attack_{scenario_id}/
    attack_dir = os.path.join(args.output_dir, f'attack_{attack_cfg.attack_id}')
    os.makedirs(attack_dir, exist_ok=True)
    print(f"Output directory: {attack_dir}")

    # ---- Setup BEVFormer paths ----
    setup_bevformer_path()

    # ---- Load model ----
    print("=" * 60)
    print("Loading BEVFormer model...")
    model, cfg = load_bevformer_model(args.config, args.checkpoint)
    print("[DONE] Model loaded")

    # ---- Connect to CARLA ----
    print("=" * 60)
    print(f"Connecting to CARLA at {args.host}:{args.port}...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    # Cleanup residual actors
    print("Cleaning up residual actors...")
    settings = world.get_settings()
    if settings.synchronous_mode:
        settings.synchronous_mode = False
        world.apply_settings(settings)
    for actor in world.get_actors():
        if 'vehicle' in actor.type_id or 'sensor' in actor.type_id:
            actor.destroy()
    time.sleep(1.0)

    # Define routes (need world to get spawn points)
    routes = define_routes(world, args.num_routes, args.start_route_id)

    # Spawn vehicle and cameras
    print("Spawning vehicle and cameras...")
    blueprint_lib = world.get_blueprint_library()
    vehicle_bp = blueprint_lib.filter('vehicle.tesla.model3')[0]
    vehicle = world.spawn_actor(vehicle_bp, routes[0]['spawn_point'])
    print(f"  Vehicle spawned: {vehicle.type_id}")
    time.sleep(1.0)

    cameras = spawn_tesla_cameras(vehicle, world)
    print(f"  {len(cameras)} cameras attached")

    # Wait for sensors to initialize (async mode, no tick needed)
    print("  Initializing sensors...")
    time.sleep(1.0)

    # Enable synchronous mode (once for entire run)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    print("  Synchronous mode enabled for entire run (dt=0.05s)")

    # Attach camera callbacks (once for entire run, after sync mode)
    frame_buffer = {}
    buffer_lock = threading.Lock()
    for cam_name, cam in cameras.items():
        cam.listen(lambda img, cn=cam_name: camera_callback_with_name(img, cn, frame_buffer, buffer_lock))
    print("  Camera callbacks attached")

    # ---- Image saving setup ----
    image_save_dir = None
    if args.save_images:
        if not HAS_PIL:
            print("[WARN] --save-images requires PIL/Pillow, but it's not installed. Skipping image saving.")
        else:
            image_save_dir = os.path.join(attack_dir, 'images')
            os.makedirs(image_save_dir, exist_ok=True)
            print(f"Image saving enabled: interval={args.save_interval}, dir={image_save_dir}")

    # ---- Run routes ----
    print(f"\nRunning {len(routes)} routes with attack: {attack_cfg.attack_id}")

    for route in routes:
        print(f"\n{'=' * 60}")
        print(f"Route: {route['route_id']}")
        print(f"{'=' * 60}")

        # Teleport to spawn point
        vehicle.set_transform(route['spawn_point'])
        for _ in range(8):
            world.tick()

        summary, frame_stats = run_one_route_reuse(
            world, vehicle, cameras, model, route,
            args.frames_per_route, args.warmup_frames, args.score_thr, injector,
            frame_buffer, buffer_lock,
            save_images=args.save_images, save_interval=args.save_interval,
            image_save_dir=image_save_dir)

        print(f"  Frames: {summary['num_frames']}")
        print(f"  Det count: {summary['det_count_mean']:.1f} "
              f"± {summary['det_count_std']:.1f}")
        print(f"  BEV cosine: {summary['bev_cosine_mean']:.6f} "
              f"± {summary['bev_cosine_std']:.6f}")
        print(f"  Attack frames: {summary['attack_frames']} "
              f"({summary['attack_frame_ratio']:.1%})")

        # Save per-route data
        route_path = os.path.join(
            attack_dir, f"attack_{attack_cfg.attack_id}_{route['route_id']}.json")
        with open(route_path, 'w') as f:
            json.dump({
                'route': route['route_id'],
                'attack_id': attack_cfg.attack_id,
                'num_frames': summary['num_frames'],
                'summary': summary,
                'frames': frame_stats,
            }, f, indent=2)
        print(f"  Saved: {route_path}")

    # ---- Final cleanup ----
    print(f"\nCleaning up vehicle and cameras...")
    for cam in cameras.values():
        cam.stop()
        cam.destroy()
    vehicle.destroy()

    # ---- Merge ALL route files on disk ----
    import glob
    import re
    print(f"\nMerging all attack_{attack_cfg.attack_id}_route_*.json in {attack_dir}...")
    merged = []
    pattern = f'attack_{attack_cfg.attack_id}_route_(\\d+)\\.json$'
    for path in glob.glob(os.path.join(attack_dir, f'attack_{attack_cfg.attack_id}_route_*.json')):
        m = re.search(pattern, path)
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
    print(f"  Found {len(all_route_summaries)} complete routes")

    # Save overall summary
    overall_path = os.path.join(attack_dir, f'attack_stats_{attack_cfg.attack_id}.json')
    with open(overall_path, 'w') as f:
        json.dump({
            'attack_id': attack_cfg.attack_id,
            'attack_config': attack_cfg.to_dict(),
            'num_routes': len(all_route_summaries),
            'per_route': all_route_summaries,
        }, f, indent=2)
    print(f"\n  Saved overall summary to {overall_path}")
    print(f"\n[DONE] Attack collection complete.")


if __name__ == '__main__':
    # Self-test block (3.2)
    if '--self-test' in sys.argv:
        print('=' * 70)
        print('collect_attack.py 自测')
        print('=' * 70)

        # 1. 验证配置加载
        print('\n[1] 配置加载验证')
        cfg = ATTACK_SCENARIOS['sign_patch_front']
        print(f'  Loaded: {cfg.attack_id}, type={cfg.attack_type}')
        print(f'  blind_to_monitor: {cfg.blind_to_monitor}')

        # 2. 验证注入器集成（不跑 CARLA，只验证接口）
        print('\n[2] 注入器接口验证')
        injector = AttackInjector(cfg)
        dummy_snapshot = {
            cam: np.zeros((900, 1600, 3), dtype=np.uint8)
            for cam in TESLA_CAMERAS.keys()
        }
        mod, meta = injector.apply(dummy_snapshot, cfg.onset_frame + 10)
        assert meta['attack_active'], 'Should be active'
        print(f'  [PASS] Injector integrated, active_cameras={meta["active_cameras"]}')

        # 3. 验证输出目录创建
        print('\n[3] 输出目录验证')
        attack_dir = f'attack_{cfg.attack_id}'
        print(f'  Will create: {attack_dir}/')
        print(f'  Route files: {attack_dir}/attack_{cfg.attack_id}_route_N.json')
        print(f'  Merged stats: {attack_dir}/attack_stats_{cfg.attack_id}.json')

        print('\n' + '=' * 70)
        print('[PASS] All interface tests passed')
        print('Note: Full CARLA integration test requires running with --attack-config')
        sys.exit(0)
    else:
        main()
