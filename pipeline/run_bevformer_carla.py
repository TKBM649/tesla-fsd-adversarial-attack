#!/usr/bin/env python3
"""
BEVFormer Inference on CARLA Images (Offline & Online)

Usage:
  # Offline mode (from saved images, no CARLA needed):
  python run_bevformer_carla.py --offline --img-dir ~/carla-adversarial/output

  # Online mode (connect to CARLA):
  python run_bevformer_carla.py --online --host localhost --port 2000

  # Quick test (adapter self-test only):
  python run_bevformer_carla.py --test
"""

import argparse
import glob
import os
import sys
import time
import numpy as np
import cv2
import torch

# ============================================================================
# Path Setup — BEVFormer project modules
# ============================================================================

BEVFORMER_ROOT = os.path.expanduser('~/carla-adversarial/BEVFormer')
BEVFORMER_PROJECTS = os.path.join(BEVFORMER_ROOT, 'projects')


def setup_bevformer_path():
    """Add BEVFormer directories to Python path for module registration."""
    for p in [BEVFORMER_ROOT, BEVFORMER_PROJECTS]:
        if p not in sys.path:
            sys.path.insert(0, p)


# ============================================================================
# Model Loading
# ============================================================================

def load_bevformer_model(config_path=None, checkpoint_path=None):
    """Load BEVFormer-tiny model.

    Args:
        config_path: path to config .py file (default: BEVFormer-tiny)
        checkpoint_path: path to .pth checkpoint

    Returns:
        model: BEVFormer model in eval mode on CUDA
    """
    # Fix: mmdet3d v0.17.1 double-registration of sparse conv modules
    import mmcv
    _orig_register = mmcv.utils.Registry._register_module
    def _patched_register(self, *args, **kwargs):
        try:
            return _orig_register(self, *args, **kwargs)
        except KeyError:
            kwargs['force'] = True
            return _orig_register(self, *args, **kwargs)
    mmcv.utils.Registry._register_module = _patched_register

    # Fix: numba >= 0.53 removed numba.errors module
    import sys, types
    try:
        import numba
        if not hasattr(numba, 'errors') or 'numba.errors' not in sys.modules:
            _errors_mod = types.ModuleType('numba.errors')
            _errors_mod.NumbaPerformanceWarning = numba.NumbaPerformanceWarning
            sys.modules['numba.errors'] = _errors_mod
            numba.errors = _errors_mod
    except ImportError:
        pass

    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model

    if config_path is None:
        config_path = os.path.join(
            BEVFORMER_ROOT, 'projects', 'configs', 'bevformer',
            'bevformer_tiny.py'
        )
    if checkpoint_path is None:
        checkpoint_path = os.path.join(BEVFORMER_ROOT, 'ckpts',
                                       'bevformer_tiny_epoch_24.pth')

    print(f"Loading config: {config_path}")
    cfg = Config.fromfile(config_path)

    # Set plugin directory
    cfg.plugin = True
    cfg.plugin_dir = 'projects/mmdet3d_plugin/'

    # Import all custom modules (register decorators)
    import projects.mmdet3d_plugin  # noqa: F401

    # Build model
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    # Set eval mode
    model.fp16_enabled = False
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("Model on CUDA GPU")
    else:
        print("WARNING: No CUDA, running on CPU (very slow)")

    # Save config for later use
    model.cfg = cfg
    print(f"Model loaded: {cfg.model.type}")
    print(f"  BEV: {cfg.bev_h_}x{cfg.bev_w_}, "
          f"Classes: {len(cfg.class_names)}, "
          f"Queue: {cfg.queue_length}")

    return model, cfg


# ============================================================================
# Inference
# ============================================================================

def run_inference(model, img_tensor, img_metas, device='cuda'):
    """Run BEVFormer inference on a single frame.

    Args:
        model: BEVFormer model (eval mode)
        img_tensor: (num_cams, C, H, W) tensor
        img_metas: dict with all required fields
        device: 'cuda' or 'cpu'

    Returns:
        result: raw model output (list of dicts with 'pts_bbox')
        bev_embed: BEV feature embedding tensor, shape (H*W, C) or None
    """
    # Move to device
    if device == 'cuda':
        img_tensor = img_tensor.cuda()

    # BEVFormer's extract_img_feat squeezes batch dim when B=1,
    # causing the camera dimension (6) to be treated as batch.
    # Fix: replicate img_metas 6 times to match the squeezed batch size.
    # Also set box_type_3d to the actual class (required by get_bboxes).
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    img_metas['box_type_3d'] = LiDARInstance3DBoxes
    img_metas_batched = [img_metas] * 6

    with torch.no_grad():
        result = model.simple_test(
            img_metas=img_metas_batched,
            img=img_tensor,
            prev_bev=None,
        )

    # simple_test returns (bev_embed, bbox_list) or just bbox_list
    bev_embed = None
    if isinstance(result, tuple) and len(result) == 2:
        bev_embed = result[0]   # BEV features: (H*W, C) = (2500, 256) for tiny
        result = result[1]       # Detection results list

    return result, bev_embed


def decode_results(result):
    """Decode BEVFormer output to readable format.

    Returns:
        dict with:
            'boxes_3d': (N, 9) numpy [x, y, z, w, l, h, yaw, vx, vy]
            'scores_3d': (N,) numpy
            'labels_3d': (N,) numpy
    """
    # simple_test returns (bev_embed, bbox_list)
    if isinstance(result, tuple):
        _, result = result

    # result is a list of dicts (one per batch element)
    if isinstance(result, list) and len(result) > 0:
        result = result[0]

    if isinstance(result, dict) and 'pts_bbox' in result:
        bbox_dict = result['pts_bbox']
    elif isinstance(result, dict):
        bbox_dict = result
    else:
        # Unexpected format, return empty
        return {'boxes_3d': np.zeros((0, 9)), 'scores_3d': np.zeros(0), 'labels_3d': np.zeros(0, dtype=int)}

    boxes_3d = bbox_dict.get('boxes_3d', None)
    scores_3d = bbox_dict.get('scores_3d', None)
    labels_3d = bbox_dict.get('labels_3d', None)

    # Convert to numpy
    if boxes_3d is not None:
        if hasattr(boxes_3d, 'tensor'):
            boxes_np = boxes_3d.tensor.cpu().numpy()
        elif isinstance(boxes_3d, torch.Tensor):
            boxes_np = boxes_3d.cpu().numpy()
        else:
            boxes_np = np.array(boxes_3d)
    else:
        boxes_np = np.zeros((0, 9))

    if scores_3d is not None:
        scores_np = scores_3d.cpu().numpy() if hasattr(scores_3d, 'cpu') else np.array(scores_3d)
    else:
        scores_np = np.zeros(0)

    if labels_3d is not None:
        labels_np = labels_3d.cpu().numpy() if hasattr(labels_3d, 'cpu') else np.array(labels_3d)
    else:
        labels_np = np.zeros(0, dtype=int)

    return {
        'boxes_3d': boxes_np,
        'scores_3d': scores_np,
        'labels_3d': labels_np,
    }


def compute_bev_similarity(bev_a, bev_b):
    """Compute cosine similarity between two BEV embedding vectors.

    Args:
        bev_a, bev_b: 1D or 2D tensors (will be flattened)

    Returns:
        float: cosine similarity in [-1, 1]
    """
    if bev_a is None or bev_b is None:
        return 0.0

    # Convert to numpy (handle both torch.Tensor and numpy.ndarray)
    def _to_numpy(x):
        if hasattr(x, 'cpu'):
            x = x.cpu()
        if hasattr(x, 'detach'):
            x = x.detach()
        if hasattr(x, 'numpy'):
            return np.array(x.numpy()).flatten().astype(np.float64)
        return np.array(x).flatten().astype(np.float64)

    a = _to_numpy(bev_a)
    b = _to_numpy(bev_b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def extract_bev_features(model, img_tensor, img_metas, device='cuda'):
    """Extract normalized BEV feature vector from BEVFormer.

    Convenience wrapper around run_inference that returns only the
    L2-normalized BEV embedding for downstream attack/defense use.

    Returns:
        numpy array: (D,) normalized BEV feature vector, or None
    """
    _, bev_embed = run_inference(model, img_tensor, img_metas, device)
    if bev_embed is None:
        return None

    if hasattr(bev_embed, 'cpu'):
        bev_np = bev_embed.flatten().cpu().numpy()
    else:
        bev_np = np.array(bev_embed).flatten()

    norm = np.linalg.norm(bev_np)
    if norm < 1e-8:
        return bev_np
    return bev_np / norm


# ============================================================================
# Offline Mode — Load images from disk
# ============================================================================

# Camera name → expected filename pattern
OFFLINE_FILE_PATTERNS = {
    'front_wide':       'front_wide*.png',
    'front_main':       'front_main*.png',
    'front_narrow':     'front_narrow*.png',
    'side_front_left':  'side_front_left*.png',
    'side_front_right': 'side_front_right*.png',
    'side_rear_left':   'side_rear_left*.png',
    'side_rear_right':  'side_rear_right*.png',
    'rear':             'rear*.png',
}


def load_offline_images(img_dir):
    """Load 8 camera images from disk.

    Args:
        img_dir: directory containing camera images (e.g., ~/carla-adversarial/output)

    Returns:
        dict: Tesla camera name → BGR image (numpy array)
    """
    images = {}
    for cam_name, pattern in OFFLINE_FILE_PATTERNS.items():
        matches = glob.glob(os.path.join(img_dir, pattern))
        if matches:
            img = cv2.imread(matches[0])
            if img is not None:
                images[cam_name] = img
                print(f"  Loaded: {cam_name:18s} ← {os.path.basename(matches[0])}  "
                      f"{img.shape}")
            else:
                print(f"  WARNING: Cannot read {matches[0]}")
        else:
            print(f"  MISSING: {cam_name:18s} (pattern: {pattern})")

    # Check minimum required cameras (6 for nuScenes mapping)
    from carla_bev_adapter import TESLA_TO_NUSCENES
    required = set(TESLA_TO_NUSCENES.values())
    available = set(images.keys())
    missing = required - available
    if missing:
        print(f"\n  ERROR: Missing required cameras: {missing}")
        return None

    print(f"\n  Loaded {len(images)} images, "
          f"{len(required)} required cameras available.")
    return images


def run_offline(args):
    """Run BEVFormer inference on saved images."""
    from carla_bev_adapter import (
        prepare_img_tensor, build_img_metas, draw_3d_detections_on_images,
        NUSCENES_CAM_ORDER, CLASS_NAMES, TESLA_TO_NUSCENES,
    )

    # Import Tesla camera layout
    sys.path.insert(0, os.path.expanduser('~/carla-adversarial/scripts'))
    from tesla_camera_layout import TESLA_CAMERAS

    print("=" * 60)
    print("BEVFormer Offline Inference")
    print("=" * 60)

    # Load images
    print(f"\nLoading images from: {args.img_dir}")
    images = load_offline_images(args.img_dir)
    if images is None:
        print("Failed to load required images.")
        return

    # Load model
    print("\nLoading BEVFormer model...")
    setup_bevformer_path()
    model, cfg = load_bevformer_model(args.config, args.checkpoint)

    # Build mock vehicle transform (stationary)
    class MockLocation:
        x, y, z = 0.0, 0.0, 0.0

    class MockRotation:
        pitch, yaw, roll = 0.0, 0.0, 0.0

    class MockTransform:
        location = MockLocation()
        rotation = MockRotation()

    # Prepare input
    print("\nPreparing model input...")
    img_tensor = prepare_img_tensor(images)
    img_metas = build_img_metas(images, TESLA_CAMERAS, MockTransform(), velocity=0.0)

    print(f"  img_tensor: {img_tensor.shape}")
    print(f"  lidar2img:  {len(img_metas['lidar2img'])} matrices")

    # Run inference
    print("\nRunning inference...")
    t0 = time.time()
    result, bev_embed = run_inference(model, img_tensor, img_metas)
    elapsed = time.time() - t0
    print(f"  Inference time: {elapsed:.3f}s")

    # Decode results
    decoded = decode_results(result)
    n_detections = len(decoded['scores_3d'])
    print(f"  Detections: {n_detections}")

    if n_detections > 0:
        # Print top detections
        top_idx = np.argsort(-decoded['scores_3d'])[:10]
        print("\n  Top detections:")
        for i in top_idx:
            box = decoded['boxes_3d'][i]
            score = decoded['scores_3d'][i]
            label = int(decoded['labels_3d'][i])
            cls = CLASS_NAMES[label] if label < len(CLASS_NAMES) else '?'
            print(f"    [{i}] {cls:18s} score={score:.3f}  "
                  f"pos=({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f})  "
                  f"size=({box[3]:.1f}, {box[4]:.1f}, {box[5]:.1f})")

    # Visualize
    output_dir = args.output_dir or args.img_dir
    print(f"\nSaving visualization to: {output_dir}")

    # Save individual camera images with detections
    cam_images = []
    for cam_name in NUSCENES_CAM_ORDER:
        tesla_name = TESLA_TO_NUSCENES[cam_name] if cam_name in TESLA_TO_NUSCENES else None
        if tesla_name and tesla_name in images:
            cam_images.append(images[tesla_name])
        else:
            cam_images.append(np.zeros((450, 800, 3), dtype=np.uint8))

    # draw_3d_detections_on_images expects list of per-camera results;
    # replicate decoded for all 6 cameras
    decoded_per_cam = [decoded] * 6
    vis_images = draw_3d_detections_on_images(
        cam_images, decoded_per_cam, img_metas, score_thr=0.1)

    # Save each camera view
    for i, cam_name in enumerate(NUSCENES_CAM_ORDER):
        out_path = os.path.join(output_dir, f'bevformer_{cam_name}.png')
        cv2.imwrite(out_path, vis_images[i])
        print(f"  Saved: {out_path}")

    # Save combined grid
    grid = _make_grid(vis_images, cols=3)
    grid_path = os.path.join(output_dir, 'bevformer_detections_grid.png')
    cv2.imwrite(grid_path, grid)
    print(f"  Grid:  {grid_path}")

    print("\nDone!")


# ============================================================================
# Online Mode — Connect to CARLA
# ============================================================================

def run_online(args):
    """Run BEVFormer inference with live CARLA connection."""
    import carla
    from carla_bev_adapter import (
        prepare_img_tensor, build_img_metas, draw_3d_detections_on_images,
        TESLA_TO_NUSCENES, NUSCENES_CAM_ORDER, CLASS_NAMES,
    )

    sys.path.insert(0, os.path.expanduser('~/carla-adversarial/scripts'))
    from tesla_camera_layout import TESLA_CAMERAS
    from setup_tesla_cameras import spawn_tesla_cameras

    print("=" * 60)
    print("BEVFormer Online Inference (CARLA)")
    print("=" * 60)

    # Load model first
    print("\nLoading BEVFormer model...")
    setup_bevformer_path()
    model, cfg = load_bevformer_model(args.config, args.checkpoint)

    # Connect to CARLA
    print(f"\nConnecting to CARLA at {args.host}:{args.port}...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    print(f"  Connected. Map: {world.get_map().name}")

    # Find or spawn vehicle (async mode for safe actor spawning)
    vehicles = world.get_actors().filter('vehicle.*')
    if vehicles:
        vehicle = vehicles[0]
        print(f"  Using existing vehicle: {vehicle.type_id}")
    else:
        bp = world.get_blueprint_library().filter('model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.spawn_actor(bp, spawn_points[0])
        print(f"  Vehicle spawned at {spawn_points[0].location}")

    vehicle.set_autopilot(True)

    # Setup cameras (still in async mode)
    print("  Attaching 8 cameras...")
    cameras = spawn_tesla_cameras(vehicle, world)
    print(f"  {len(cameras)} cameras attached")

    # NOW enable synchronous mode for deterministic ticking
    settings = world.get_settings()
    if not settings.synchronous_mode:
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        print("  Enabled synchronous mode (dt=0.05s)")
    else:
        print("  Synchronous mode already enabled")

    # Wait for sensors to initialize
    print("  Waiting for sensors to warm up...")
    time.sleep(2.0)
    # Pre-tick a few frames to let all cameras start streaming
    for _ in range(5):
        world.tick()
        time.sleep(0.1)

    # Frame buffer — shared across camera callbacks
    frame_buffer = {}
    import threading
    buffer_lock = threading.Lock()

    def make_callback(name):
        def callback(image):
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((image.height, image.width, 4))
            arr = arr[:, :, :3]  # Drop alpha
            with buffer_lock:
                frame_buffer[name] = arr.copy()
        return callback

    for name, cam in cameras.items():
        cam.listen(make_callback(name))

    output_dir = args.output_dir or os.path.expanduser('~/carla-adversarial/output')
    os.makedirs(output_dir, exist_ok=True)

    # Inference loop
    num_frames = args.num_frames
    print(f"\nRunning inference loop ({num_frames} frames)...")
    frame_count = 0
    total_time = 0.0

    try:
        for frame_idx in range(num_frames):
            # Tick CARLA and wait for callbacks
            try:
                world.tick()
            except Exception as e:
                print(f"  Frame {frame_idx}: tick failed: {e}")
                continue
            time.sleep(0.2)

            # Check all images received (buffer keeps latest from each camera)
            with buffer_lock:
                n_received = len(frame_buffer)
                if n_received < 8:
                    print(f"  Frame {frame_idx}: only {n_received}/8 images, skipping")
                    continue
                # Snapshot the buffer
                images_snapshot = {k: v.copy() for k, v in frame_buffer.items()}

            # Get vehicle state
            transform = vehicle.get_transform()
            velocity_vec = vehicle.get_velocity()
            speed = np.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)

            # Prepare input
            img_tensor = prepare_img_tensor(images_snapshot)
            img_metas = build_img_metas(images_snapshot, TESLA_CAMERAS,
                                        transform, velocity=speed)

            # Inference
            t0 = time.time()
            result, bev_embed = run_inference(model, img_tensor, img_metas)
            elapsed = time.time() - t0
            total_time += elapsed
            frame_count += 1

            # Decode
            decoded = decode_results(result)
            n_det = len(decoded['scores_3d'])

            print(f"  Frame {frame_idx}: {n_det} detections, "
                  f"{elapsed:.3f}s, speed={speed:.1f}m/s")

            # Save last frame visualization
            if frame_idx == num_frames - 1:
                cam_images = []
                for cam_name in NUSCENES_CAM_ORDER:
                    tesla_name = TESLA_TO_NUSCENES.get(cam_name, '')
                    if tesla_name in images_snapshot:
                        cam_images.append(images_snapshot[tesla_name])
                    else:
                        cam_images.append(np.zeros((450, 800, 3), dtype=np.uint8))

                # draw expects list of per-camera decoded dicts
                decoded_per_cam = [decoded] * 6
                vis_images = draw_3d_detections_on_images(
                    cam_images, decoded_per_cam, img_metas, score_thr=0.1
                )

                for i, cn in enumerate(NUSCENES_CAM_ORDER):
                    out_path = os.path.join(output_dir, f'bevformer_online_{cn}.png')
                    cv2.imwrite(out_path, vis_images[i])

                grid = _make_grid(vis_images, cols=3)
                grid_path = os.path.join(output_dir, 'bevformer_online_grid.png')
                cv2.imwrite(grid_path, grid)
                print(f"\n  Saved visualization to {output_dir}")

    finally:
        # Cleanup
        print("\nCleaning up...")
        for cam in cameras.values():
            cam.stop()
            cam.destroy()
        vehicle.set_autopilot(False)
        vehicle.destroy()

        # Restore async mode
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

    avg = total_time / max(frame_count, 1)
    print(f"\nProcessed {frame_count}/{num_frames} frames. "
          f"Avg inference: {avg:.3f}s/frame. Done!")


# ============================================================================
# Utility
# ============================================================================

def _make_grid(images, cols=3, cell_h=450, cell_w=800):
    """Arrange images in a grid."""
    rows = int(np.ceil(len(images) / cols))
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        resized = cv2.resize(img, (cell_w, cell_h))
        grid[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized
    return grid


# ============================================================================
# Adapter Self-Test
# ============================================================================

def run_test():
    """Run adapter self-test."""
    sys.path.insert(0, os.path.expanduser('~/carla-adversarial/scripts'))
    from carla_bev_adapter import test_adapter
    test_adapter()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='BEVFormer Inference on CARLA Images')

    # Mode
    parser.add_argument('--offline', action='store_true',
                        help='Offline mode: load images from disk')
    parser.add_argument('--online', action='store_true',
                        help='Online mode: connect to CARLA')
    parser.add_argument('--test', action='store_true',
                        help='Run adapter self-test only')

    # Paths
    parser.add_argument('--img-dir', type=str,
                        default=os.path.expanduser('~/carla-adversarial/output'),
                        help='Directory with camera images (offline mode)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for visualizations')
    parser.add_argument('--config', type=str, default=None,
                        help='BEVFormer config file path')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='BEVFormer checkpoint file path')

    # CARLA connection
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--num-frames', type=int, default=50,
                        help='Number of frames for online mode')

    args = parser.parse_args()

    if args.test:
        run_test()
    elif args.offline:
        run_offline(args)
    elif args.online:
        run_online(args)
    else:
        parser.print_help()
        print("\nPlease specify --offline, --online, or --test")


if __name__ == '__main__':
    main()
