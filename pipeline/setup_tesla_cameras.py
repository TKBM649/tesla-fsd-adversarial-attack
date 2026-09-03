#!/usr/bin/env python3
"""
Spawn Tesla FSD 8-camera layout on a vehicle in CARLA and capture verification images.

Usage:
  # Basic: spawn cameras + capture 8 images
  python setup_tesla_cameras.py

  # With BEV stitching (requires opencv-python):
  python setup_tesla_cameras.py --bev

  # From WSL:
  wsl.exe -d Ubuntu -- bash -lc "source ~/carla-env/bin/activate && python ~/carla-adversarial/scripts/setup_tesla_cameras.py --bev"
"""
import sys
import os
import time
import argparse
import carla

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from tesla_camera_layout import TESLA_CAMERAS

OUTPUT_DIR = os.path.join(os.path.expanduser('~'), 'carla-adversarial', 'output')


def spawn_tesla_cameras(vehicle, world):
    """Attach 8 cameras to the given vehicle."""
    blueprint_library = world.get_blueprint_library()
    cameras = {}

    for name, config in TESLA_CAMERAS.items():
        bp = blueprint_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(config['width']))
        bp.set_attribute('image_size_y', str(config['height']))
        bp.set_attribute('fov', str(config['fov']))
        bp.set_attribute('sensor_tick', str(1.0 / config['fps']))

        # CARLA attaches child actors with rotation relative to parent.
        # For side/rear cameras (|yaw| >= 45°), the yaw direction is inverted
        # when attached to a vehicle, so we negate it to face outward.
        raw_yaw = config['rotation'][1]
        carla_yaw = -raw_yaw if abs(raw_yaw) >= 45 else raw_yaw

        transform = carla.Transform(
            carla.Location(
                x=config['location'][0],
                y=config['location'][1],
                z=config['location'][2]
            ),
            carla.Rotation(
                pitch=config['rotation'][0],
                yaw=carla_yaw,
                roll=config['rotation'][2]
            )
        )

        camera = world.spawn_actor(bp, transform, attach_to=vehicle)
        cameras[name] = camera
        print(f"[OK] {name:16s} pos=({config['location'][0]:5.1f},{config['location'][1]:5.1f},{config['location'][2]:5.1f}) "
              f"raw_yaw={raw_yaw:>6.1f} carla_yaw={carla_yaw:>6.1f}  fov={config['fov']:.0f}")

    return cameras


def capture_one_frame(cameras, output_dir):
    """Capture one frame from each camera and save as PNG."""
    os.makedirs(output_dir, exist_ok=True)
    images = {}

    for name, camera in cameras.items():
        result = []

        def callback(image):
            result.append(image)

        camera.listen(callback)
        time.sleep(0.5)
        camera.stop()

        if result:
            filepath = os.path.join(output_dir, f'{name}.png')
            result[0].save_to_disk(filepath)
            print(f"[SAVED] {name:16s} -> {filepath}")

            # Convert to numpy for BEV
            import numpy as np
            img_data = np.frombuffer(result[0].raw_data, dtype=np.uint8)
            img_array = img_data.reshape((result[0].height, result[0].width, 4))
            images[name] = img_array[:, :, :3]
        else:
            print(f"[WARN] {name:16s} no image captured")
            images[name] = None

    return images


def run_bev_stitching(images):
    """Run BEV stitching on captured images."""
    try:
        from bev_stitching import stitch_bev_simple, stitch_bev, create_grid_view, save_individual_warps
    except ImportError:
        print("[ERROR] bev_stitching module not found. Skipping BEV.")
        return

    n_valid = sum(1 for v in images.values() if v is not None)
    if n_valid == 0:
        print("[ERROR] No valid images for BEV stitching.")
        return

    # Grid view
    print("\n--- Grid view ---")
    grid_path = os.path.join(OUTPUT_DIR, 'camera_grid.png')
    create_grid_view(images, grid_path)

    # BEV simple
    print("\n--- BEV stitching (simple overlay) ---")
    bev_simple = stitch_bev_simple(images)
    import cv2
    path = os.path.join(OUTPUT_DIR, 'bev_surround_simple.png')
    cv2.imwrite(path, bev_simple)
    print(f"[SAVED] BEV simple -> {path}")

    # BEV blend
    print("\n--- BEV stitching (weighted blend) ---")
    bev_blend = stitch_bev(images)
    path = os.path.join(OUTPUT_DIR, 'bev_surround_blend.png')
    cv2.imwrite(path, bev_blend)
    print(f"[SAVED] BEV blend -> {path}")

    # Individual warps
    print("\n--- Individual BEV warps ---")
    save_individual_warps(images, OUTPUT_DIR)


def main():
    parser = argparse.ArgumentParser(description='Tesla 8-camera setup for CARLA')
    parser.add_argument('--bev', action='store_true',
                        help='Run BEV stitching after capture (requires opencv-python)')
    args = parser.parse_args()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    print(f"Connected to CARLA. Map: {world.get_map().name}")

    # Find or spawn a vehicle
    vehicles = world.get_actors().filter('vehicle.*')
    if vehicles:
        vehicle = vehicles[0]
        print(f"Using existing vehicle: {vehicle.type_id}")
    else:
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Spawned vehicle: {vehicle.type_id}")
        time.sleep(2.0)

    # Attach cameras
    cameras = spawn_tesla_cameras(vehicle, world)
    print(f"\nTotal cameras attached: {len(cameras)}")

    # Capture verification images
    print("\nCapturing one frame from each camera...")
    images = capture_one_frame(cameras, OUTPUT_DIR)

    # Optional: BEV stitching
    if args.bev:
        print("\n" + "=" * 50)
        print("Running BEV surround-view stitching...")
        print("=" * 50)
        run_bev_stitching(images)

    # Cleanup cameras
    for name, camera in cameras.items():
        camera.destroy()
    print(f"\n[DONE] All {len(cameras)} cameras verified.")
    if args.bev:
        print(f"  BEV outputs in {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
