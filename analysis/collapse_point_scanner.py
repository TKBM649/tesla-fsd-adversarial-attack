"""
collapse_point_scanner.py -- Phase 3B collapse point scanning engine

Reuses collect_attack.py's frame loop (run_one_route_reuse) to execute
systematic attack intensity x camera combination experiments.

Dependencies: CARLA + collect_attack + collapse_configs (WSL only)
"""

import json
import os
import sys
import time
import threading
import numpy as np

# ---- CARLA imports ----
try:
    import carla
except ImportError:
    print("[FATAL] carla module not found. This script must run in WSL with CARLA.")
    sys.exit(1)

# ---- Project imports ----
from run_bevformer_carla import setup_bevformer_path, load_bevformer_model
from tesla_camera_layout import TESLA_CAMERAS
from setup_tesla_cameras import spawn_tesla_cameras
from attack_injector import AttackInjector
from collect_attack import (
    define_routes, run_one_route_reuse, camera_callback_with_name,
    compute_route_summary,
)
from collapse_configs import (
    CAMERA_GROUPS, INTENSITY_LEVELS, DUTY_CYCLES,
    make_attack_config, generate_experiment_matrix,
    condition_to_dirname, dirname_to_condition, EXPERIMENT_MODES,
)


# ============================================================================
# Scanner class
# ============================================================================

class CollapsePointScanner:
    """
    Systematically scans collapse points by varying attack intensity,
    camera combinations, and duty cycles.

    Usage:
        scanner = CollapsePointScanner(world, vehicle, cameras, model)
        scanner.scan_batch(conditions, output_dir, num_routes=10)
    """

    def __init__(self, world, vehicle, cameras, model,
                 warmup_frames=8, score_thr=0.05):
        self.world = world
        self.vehicle = vehicle
        self.cameras = cameras
        self.model = model
        self.warmup_frames = warmup_frames
        self.score_thr = score_thr

        # Shared frame buffer (camera callbacks attached once)
        self.frame_buffer = {}
        self.buffer_lock = threading.Lock()

    def attach_callbacks(self):
        """Attach camera callbacks (call once after sync mode is enabled)."""
        for cam_name, cam in self.cameras.items():
            cam.listen(lambda img, cn=cam_name:
                       camera_callback_with_name(img, cn,
                                                 self.frame_buffer,
                                                 self.buffer_lock))
        print(f"  Camera callbacks attached ({len(self.cameras)} cameras)")

    def scan_one_condition(self, group_id, intensity, duty_cycle,
                           num_routes=10, frames_per_route=200,
                           output_dir=None, start_route_id=0):
        """
        Execute one experimental condition.

        Returns:
            condition_result dict with per-route frame data
        """
        # 1. Create attack config and injector
        config = make_attack_config(group_id, intensity, duty_cycle)
        injector = AttackInjector(config)

        # 2. Define routes
        routes = define_routes(self.world, num_routes, start_route_id)

        # 3. Run each route
        per_route = []
        for route in routes:
            # Teleport to spawn point
            self.vehicle.set_transform(route['spawn_point'])
            for _ in range(8):
                self.world.tick()

            # Run frame loop (reuses collect_attack.py infrastructure)
            summary, frame_stats = run_one_route_reuse(
                self.world, self.vehicle, self.cameras, self.model,
                route, frames_per_route, self.warmup_frames, self.score_thr,
                injector, self.frame_buffer, self.buffer_lock,
                save_images=False,
            )

            # Extract per-frame arrays (compact format)
            route_data = {
                'route_id': route['route_id'],
                'num_frames': summary['num_frames'],
                'frame_idx': [f['frame_idx'] for f in frame_stats],
                'det_count': [f['det_count'] for f in frame_stats],
                'det_score_max': [f['det_score_max'] for f in frame_stats],
                'bev_self_sim': [f['bev_self_sim'] for f in frame_stats],
                'bev_l2_dist': [f['bev_l2_dist'] for f in frame_stats],
                'attack_active': [f.get('attack_active', False)
                                  for f in frame_stats],
                'speed': [f['speed'] for f in frame_stats],
                'recoveries': summary.get('recoveries', 0),
            }
            per_route.append(route_data)

        # 4. Build condition result
        condition_result = {
            'group_id': group_id,
            'intensity': intensity,
            'duty_cycle': duty_cycle,
            'attack_id': config.attack_id,
            'num_routes': len(per_route),
            'frames_per_route': frames_per_route,
            'per_route': per_route,
        }

        # 5. Save if output_dir specified
        if output_dir:
            cond_dir = os.path.join(
                output_dir, group_id,
                condition_to_dirname(group_id, intensity, duty_cycle))
            os.makedirs(cond_dir, exist_ok=True)

            # Save per-route files
            for rd in per_route:
                route_idx = rd['route_id'].split('_')[-1]
                fpath = os.path.join(
                    cond_dir, f'route_{route_idx}.json')
                with open(fpath, 'w') as f:
                    json.dump(rd, f, indent=2)

            # Save condition summary
            summary_path = os.path.join(cond_dir, 'condition_summary.json')
            with open(summary_path, 'w') as f:
                json.dump({
                    'group_id': group_id,
                    'intensity': intensity,
                    'duty_cycle': duty_cycle,
                    'num_routes': len(per_route),
                    'per_route_summary': [
                        {
                            'route_id': rd['route_id'],
                            'num_frames': rd['num_frames'],
                            'det_count_mean': float(np.mean(rd['det_count'])),
                            'det_count_median': float(np.median(rd['det_count'])),
                            'det_count_max': int(np.max(rd['det_count'])),
                            'attack_active_ratio': float(np.mean(rd['attack_active'])),
                            'recoveries': rd['recoveries'],
                        }
                        for rd in per_route
                    ],
                }, f, indent=2)

        return condition_result

    def scan_batch(self, conditions, output_dir, num_routes=10,
                   frames_per_route=200, resume=True):
        """
        Scan multiple conditions with progress tracking and resume support.

        Args:
            conditions: list of (group_id, intensity, duty_cycle) tuples
            output_dir: base output directory
            num_routes: routes per condition
            frames_per_route: frames per route
            resume: if True, skip already-completed conditions

        Returns:
            list of condition_result dicts
        """
        os.makedirs(output_dir, exist_ok=True)

        # Progress tracking
        progress_path = os.path.join(output_dir, 'scan_progress.json')
        completed = set()
        if resume and os.path.exists(progress_path):
            with open(progress_path, 'r') as f:
                progress = json.load(f)
            completed = set(tuple(c) for c in progress.get('completed', []))
            print(f"  Resuming: {len(completed)} conditions already completed")

        results = []
        total = len(conditions)

        for idx, (gid, intensity, duty) in enumerate(conditions):
            cond_key = (gid, intensity, duty)
            cond_dirname = condition_to_dirname(gid, intensity, duty)

            # Skip completed
            if cond_key in completed:
                print(f"\n[{idx+1}/{total}] SKIP (already done): {cond_dirname}")
                continue

            print(f"\n{'='*60}")
            print(f"[{idx+1}/{total}] Scanning: {cond_dirname}")
            print(f"  cameras={CAMERA_GROUPS[gid]}, "
                  f"intensity={intensity}, duty={duty}")
            print(f"{'='*60}")

            t0 = time.time()
            try:
                result = self.scan_one_condition(
                    gid, intensity, duty,
                    num_routes=num_routes,
                    frames_per_route=frames_per_route,
                    output_dir=output_dir,
                )
                elapsed = time.time() - t0

                # Print quick summary
                all_det = []
                for rd in result['per_route']:
                    all_det.extend(rd['det_count'])
                print(f"  [DONE] {elapsed:.0f}s | "
                      f"det_count median={np.median(all_det):.1f}, "
                      f"max={np.max(all_det)}")

                results.append(result)

                # Mark completed
                completed.add(cond_key)
                with open(progress_path, 'w') as f:
                    json.dump({
                        'completed': [list(c) for c in completed],
                        'total': total,
                        'last_update': time.strftime('%Y-%m-%d %H:%M:%S'),
                    }, f, indent=2)

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  [FAIL] {elapsed:.0f}s | Error: {e}")
                # Save error info
                err_dir = os.path.join(output_dir, gid, cond_dirname)
                os.makedirs(err_dir, exist_ok=True)
                with open(os.path.join(err_dir, 'error.json'), 'w') as f:
                    json.dump({
                        'group_id': gid,
                        'intensity': intensity,
                        'duty_cycle': duty,
                        'error': str(e),
                    }, f, indent=2)

        print(f"\n[DONE] Scan complete: {len(results)} new conditions, "
              f"{len(completed)} total completed")
        return results


# ============================================================================
# CARLA environment setup (shared with run_collapse_experiment.py)
# ============================================================================

def setup_carla_environment(host='localhost', port=2000, num_routes=10):
    """
    Initialize CARLA environment: connect, spawn vehicle + cameras,
    enable sync mode, attach callbacks.

    Returns:
        (client, world, vehicle, cameras, routes)
    """
    print("=" * 60)
    print(f"Connecting to CARLA at {host}:{port}...")
    client = carla.Client(host, port)
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

    # Define routes
    routes = define_routes(world, num_routes)

    # Spawn vehicle
    print("Spawning vehicle and cameras...")
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
    vehicle = world.spawn_actor(vehicle_bp, routes[0]['spawn_point'])
    print(f"  Vehicle: {vehicle.type_id}")
    time.sleep(1.0)

    # Spawn cameras
    cameras = spawn_tesla_cameras(vehicle, world)
    print(f"  {len(cameras)} cameras attached")
    time.sleep(1.0)

    # Enable sync mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    print("  Synchronous mode enabled (dt=0.05s)")

    return client, world, vehicle, cameras, routes


def teardown_carla_environment(vehicle, cameras):
    """Clean up CARLA actors."""
    print("Cleaning up...")
    for cam in cameras.values():
        cam.stop()
        cam.destroy()
    vehicle.destroy()


# ============================================================================
# Self-test (no CARLA required for basic validation)
# ============================================================================

if __name__ == '__main__':
    if '--self-test' in sys.argv:
        print('=' * 70)
        print('collapse_point_scanner.py self-test')
        print('=' * 70)

        # 1. Import validation
        print('\n[1] Import validation')
        print('  [PASS] All imports successful')

        # 2. Config generation
        print('\n[2] Config generation for scan conditions')
        for gid in ['single_front_main', 'dual_front', 'triple_front']:
            for inten in [0.0, 0.5, 1.0]:
                cfg = make_attack_config(gid, inten, 1.0)
                print(f'  {cfg.attack_id:<55} '
                      f'cams={cfg.target_cameras} patch={cfg.patch_frac}')

        # 3. Experiment matrix
        print('\n[3] Experiment matrix (h1_quick)')
        conds, info = generate_experiment_matrix(mode='h1_quick')
        print(f'  {info["num_conditions"]} conditions x '
              f'{info["num_routes"]} routes = {info["total_routes"]} total')
        for gid, inten, duty in conds[:5]:
            print(f'  ({gid}, {inten}, {duty})')
        print(f'  ... ({len(conds)} total)')

        print('\n[PASS] All self-tests passed')
        print('Note: Full scan requires CARLA. Use run_collapse_experiment.py')
        sys.exit(0)
    else:
        print("This module provides CollapsePointScanner.")
        print("Use run_collapse_experiment.py as the CLI entry point.")
        print("Or run with --self-test for basic validation.")
