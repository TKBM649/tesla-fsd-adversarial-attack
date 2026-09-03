#!/usr/bin/env python3
"""
Tesla FSD 8-Camera BEV Demo for CARLA.

End-to-end pipeline:
  1. Connect to CARLA server
  2. Spawn Tesla Model 3 vehicle
  3. Attach 8 cameras (Tesla HW 3.0 layout)
  4. Capture one frame from each camera
  5. Run BEV surround-view stitching
  6. Save all outputs (individual images, grid view, BEV composite)

Usage:
  # In WSL with carla-env activated:
  source ~/carla-env/bin/activate
  python ~/carla-adversarial/scripts/run_bev_demo.py

  # Or from the chat-3 directory (Windows side):
  wsl.exe -d Ubuntu -- bash -lc "source ~/carla-env/bin/activate && python ~/carla-adversarial/scripts/run_bev_demo.py"
"""

import sys
import os
import time
import numpy as np

# Add script directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import carla
from tesla_camera_layout import TESLA_CAMERAS
from bev_stitching import stitch_bev, stitch_bev_simple, create_grid_view, save_individual_warps


OUTPUT_DIR = os.path.join(os.path.expanduser('~'), 'carla-adversarial', 'output')


def spawn_vehicle(world):
    """Find existing vehicle or spawn a new Tesla Model 3."""
    vehicles = world.get_actors().filter('vehicle.*')
    if vehicles:
        vehicle = vehicles[0]
        print(f"[OK] Using existing vehicle: {vehicle.type_id}")
        return vehicle

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"[OK] Spawned vehicle: {vehicle.type_id}")
    time.sleep(1.0)
    return vehicle


def attach_cameras(vehicle, world):
    """Attach 8 Tesla cameras to the vehicle."""
    blueprint_library = world.get_blueprint_library()
    cameras = {}

    for name, config in TESLA_CAMERAS.items():
        bp = blueprint_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(config['width']))
        bp.set_attribute('image_size_y', str(config['height']))
        bp.set_attribute('fov', str(config['fov']))
        bp.set_attribute('sensor_tick', str(1.0 / config['fps']))

        transform = carla.Transform(
            carla.Location(
                x=config['location'][0],
                y=config['location'][1],
                z=config['location'][2]
            ),
            carla.Rotation(
                pitch=config['rotation'][0],
                yaw=config['rotation'][1],
                roll=config['rotation'][2]
            )
        )

        camera = world.spawn_actor(bp, transform, attach_to=vehicle)
        cameras[name] = camera
        print(f"[OK] {name:16s} pos=({config['location'][0]:5.1f},{config['location'][1]:5.1f},{config['location'][2]:5.1f}) "
              f"yaw={config['rotation'][1]:>6.1f}  fov={config['fov']:.0f}")

    return cameras


def capture_images(cameras, wait_time=0.5):
    """Capture one frame from each camera."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images = {}

    for name, camera in cameras.items():
        result = []

        def callback(image):
            result.append(image)

        camera.listen(callback)
        time.sleep(wait_time)
        camera.stop()

        if result:
            filepath = os.path.join(OUTPUT_DIR, f'{name}.png')
            result[0].save_to_disk(filepath)

            # Convert to numpy array for BEV stitching
            img_data = np.frombuffer(result[0].raw_data, dtype=np.uint8)
            img_array = img_data.reshape((result[0].height, result[0].width, 4))
            img_bgr = img_array[:, :, :3]  # drop alpha channel
            images[name] = img_bgr
            print(f"[CAPTURED] {name:16s} -> {filepath} ({result[0].width}x{result[0].height})")
        else:
            print(f"[WARN] {name:16s} no image captured")
            images[name] = None

    return images


def cleanup(cameras, vehicle=None):
    """Destroy cameras and optionally the vehicle."""
    for name, camera in cameras.items():
        camera.destroy()
    print(f"[OK] Destroyed {len(cameras)} cameras")

    if vehicle and vehicle.type_id == 'vehicle.tesla.model3':
        vehicle.destroy()
        print("[OK] Destroyed vehicle")


def main():
    print("=" * 60)
    print("Tesla FSD 8-Camera BEV Demo")
    print("=" * 60)

    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print(f"\n[OK] Connected to CARLA. Map: {world.get_map().name}")

    # Spawn vehicle
    print("\n--- Step 1: Vehicle ---")
    vehicle = spawn_vehicle(world)

    # Attach cameras
    print("\n--- Step 2: Cameras ---")
    cameras = attach_cameras(vehicle, world)
    print(f"\nTotal cameras: {len(cameras)}")

    # Capture images
    print("\n--- Step 3: Capture ---")
    images = capture_images(cameras)

    n_valid = sum(1 for v in images.values() if v is not None)
    print(f"\nCaptured {n_valid}/{len(cameras)} images")

    if n_valid == 0:
        print("[ERROR] No images captured!")
        cleanup(cameras)
        return

    # Save individual images
    print("\n--- Step 4: Save individual images ---")
    for name, img in images.items():
        if img is not None:
            path = os.path.join(OUTPUT_DIR, f'{name}.png')
            import cv2
            cv2.imwrite(path, img)

    # Grid view
    print("\n--- Step 5: Grid view ---")
    grid_path = os.path.join(OUTPUT_DIR, 'camera_grid.png')
    create_grid_view(images, grid_path)

    # BEV stitching (simple)
    print("\n--- Step 6: BEV stitching (simple overlay) ---")
    bev_simple = stitch_bev_simple(images)
    bev_simple_path = os.path.join(OUTPUT_DIR, 'bev_surround_simple.png')
    import cv2
    cv2.imwrite(bev_simple_path, bev_simple)
    print(f"[SAVED] BEV simple -> {bev_simple_path}")

    # BEV stitching (weighted blend)
    print("\n--- Step 7: BEV stitching (weighted blend) ---")
    bev_blend = stitch_bev(images)
    bev_blend_path = os.path.join(OUTPUT_DIR, 'bev_surround_blend.png')
    cv2.imwrite(bev_blend_path, bev_blend)
    print(f"[SAVED] BEV blend -> {bev_blend_path}")

    # Individual warps
    print("\n--- Step 8: Individual BEV warps ---")
    save_individual_warps(images, OUTPUT_DIR)

    # Cleanup
    print("\n--- Cleanup ---")
    cleanup(cameras)

    print("\n" + "=" * 60)
    print("[DONE] All outputs saved to:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    camera_grid.png          - 8-camera overview grid")
    print(f"    bev_surround_simple.png  - BEV (simple overlay)")
    print(f"    bev_surround_blend.png   - BEV (weighted blend)")
    print(f"    bev_*.png                - individual camera warps")
    print(f"    *.png                    - raw camera images")
    print("=" * 60)


if __name__ == '__main__':
    main()
