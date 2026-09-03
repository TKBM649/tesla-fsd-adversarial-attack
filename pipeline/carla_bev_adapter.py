"""
CARLA → BEVFormer Adapter Module

Maps Tesla 8-camera CARLA layout to BEVFormer's expected nuScenes 6-camera format.
Handles coordinate transforms, image preprocessing, and metadata construction.

Coordinate systems:
  - CARLA: left-handed (x=forward, y=left, z=up)
  - nuScenes: right-handed (x=forward, y=left, z=up)
  - Ego vehicle frame is identical; only the handedness convention differs.
"""

import numpy as np
import cv2
import math
import torch
from scipy.spatial.transform import Rotation as R

# ============================================================================
# Constants
# ============================================================================

# Tesla 8 cameras → nuScenes 6 cameras mapping
# nuScenes order: FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT
TESLA_TO_NUSCENES = {
    'CAM_FRONT':       'front_main',
    'CAM_FRONT_RIGHT': 'front_wide',
    'CAM_FRONT_LEFT':  'side_front_left',
    'CAM_BACK':        'rear',
    'CAM_BACK_LEFT':   'side_rear_left',
    'CAM_BACK_RIGHT':  'side_front_right',
}
NUSCENES_CAM_ORDER = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]

# BEVFormer-tiny image pipeline constants
# Original: 1600x900 * scale 0.5 = 800x450, pad to divisor 32
TARGET_IMG_W = 800
TARGET_IMG_H = 450
PAD_SIZE_DIVISOR = 32

# ImageNet normalization (BEVFormer standard)
IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# BEVFormer-tiny detection classes (nuScenes 10-class)
CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

# Visualization colors (BGR) for each class
CLASS_COLORS = [
    (255, 100, 50),   # car - orange
    (0, 200, 200),    # truck - yellow
    (0, 165, 255),    # construction - dark orange
    (255, 200, 0),    # bus - light blue
    (0, 255, 255),    # trailer - cyan
    (128, 0, 128),    # barrier - purple
    (0, 255, 0),      # motorcycle - green
    (255, 0, 255),    # bicycle - magenta
    (0, 0, 255),      # pedestrian - red
    (0, 128, 0),      # traffic_cone - dark green
]


# ============================================================================
# Coordinate Transform Utilities
# ============================================================================

def euler_to_rotation(pitch, yaw, roll):
    """Convert CARLA Euler angles (degrees) to 3x3 rotation matrix.

    CARLA convention: applied in ZYX order
      (yaw around Z first, then pitch around Y, then roll around X).
    """
    r = R.from_euler('ZYX', [yaw, pitch, roll], degrees=True)
    return r.as_matrix()


def carla_to_nuscenes_quaternion(pitch, yaw, roll):
    """Convert CARLA Euler angles (degrees) to nuScenes quaternion (xyzw).
    """
    r = R.from_euler('ZYX', [yaw, pitch, roll], degrees=True)
    quat_xyzw = r.as_quat()  # scipy returns [x, y, z, w]
    return quat_xyzw


def compute_intrinsic_matrix(fov_deg, width, height):
    """Compute 3x3 camera intrinsic matrix K from FOV and image size.

    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

    where fx = fy = (width/2) / tan(fov/2)
    """
    fov_rad = math.radians(fov_deg)
    fx = fy = (width / 2.0) / math.tan(fov_rad / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    return np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1.0],
    ], dtype=np.float64)


def build_lidar2cam(location, pitch, yaw, roll):
    """Build 4x4 lidar (ego vehicle) → camera transformation matrix.

    Both CARLA and nuScenes use the same ego frame:
      x=forward, y=left, z=up

    For a camera at position `t` with rotation `R` in the ego frame:
      point_cam = R^T @ (point_lidar - t)

    So:
      lidar2cam[:3,:3] = R^T
      lidar2cam[:3, 3] = -R^T @ t
    """
    rot = euler_to_rotation(pitch, yaw, roll)
    R_inv = rot.T
    t = np.array(location, dtype=np.float64)

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R_inv
    M[:3, 3] = -R_inv @ t
    return M


def build_lidar2img(location, pitch, yaw, roll, fov_deg,
                    img_width, img_height):
    """Build 4x4 lidar → image projection matrix.

    lidar2img = K_padded @ lidar2cam

    where K_padded is the 4x4 padded intrinsic matrix:
      [[fx,  0, cx, 0],
       [ 0, fy, cy, 0],
       [ 0,  0,  1, 0],
       [ 0,  0,  0, 1]]

    This matrix projects 3D points in the ego frame to 2D pixel coordinates:
      [u, v, d, w]^T = lidar2img @ [x, y, z, 1]^T
      pixel_u = u / d,  pixel_v = v / d
    """
    K = compute_intrinsic_matrix(fov_deg, img_width, img_height)
    lidar2cam = build_lidar2cam(location, pitch, yaw, roll)

    # Pad K to 4x4 for matrix multiplication
    K_4x4 = np.eye(4, dtype=np.float64)
    K_4x4[:3, :3] = K

    lidar2img = K_4x4 @ lidar2cam
    return lidar2img, K, lidar2cam


# ============================================================================
# Image Preprocessing
# ============================================================================

def preprocess_image(img):
    """Preprocess a single CARLA image for BEVFormer input.

    Pipeline (matching BEVFormer test_pipeline):
      1. BGR → RGB
      2. Resize to (TARGET_IMG_W, TARGET_IMG_H)
      3. Normalize with ImageNet mean/std
      4. Pad to multiple of PAD_SIZE_DIVISOR
      5. Convert to CHW tensor

    Args:
        img: numpy array (H, W, 3) in BGR format (from CARLA)

    Returns:
        tensor: (C, H_padded, W_padded) float32 tensor
        tuple: (H_padded, W_padded, 3) padded shape
    """
    # 1. BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Resize to target
    img_resized = cv2.resize(
        img_rgb, (TARGET_IMG_W, TARGET_IMG_H),
        interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)

    # 3. Normalize
    img_norm = mmcv_normalize(img_resized, IMG_MEAN, IMG_STD, to_rgb=False)

    # 4. Pad to divisor
    H, W = img_norm.shape[:2]
    pad_h = int(math.ceil(H / PAD_SIZE_DIVISOR)) * PAD_SIZE_DIVISOR
    pad_w = int(math.ceil(W / PAD_SIZE_DIVISOR)) * PAD_SIZE_DIVISOR
    img_padded = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
    img_padded[:H, :W] = img_norm

    # 5. HWC → CHW tensor
    tensor = torch.from_numpy(img_padded).permute(2, 0, 1).contiguous()
    return tensor, (pad_h, pad_w, 3)


def mmcv_normalize(img, mean, std, to_rgb=False):
    """Image normalization (replicates mmcv.imnormalize)."""
    img = img.copy().astype(np.float32)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    img = (img - mean) / std
    return img


# ============================================================================
# CAN Bus & Metadata Construction
# ============================================================================

def build_can_bus(transform, velocity=0.0):
    """Build 18-dim can_bus vector for BEVFormer.

    Layout:
      [0:3]   position (x, y, z) in global frame
      [3:7]   rotation quaternion (x, y, z, w)
      [7:10]  velocity (vx, vy, vz)
      [10:13] acceleration (0, 0, 0)
      [13:16] rotation rate (0, 0, 0)
      [16]    patch_angle_normalized = yaw / 180 * pi
      [17]    patch_angle_degrees = yaw
    """
    can_bus = np.zeros(18, dtype=np.float32)

    # Position
    loc = transform.location
    can_bus[0] = loc.x
    can_bus[1] = loc.y
    can_bus[2] = loc.z

    # Rotation (CARLA → nuScenes quaternion)
    rot = transform.rotation
    quat = carla_to_nuscenes_quaternion(rot.pitch, rot.yaw, rot.roll)
    can_bus[3:7] = quat

    # Velocity (simplified: forward speed only)
    can_bus[7] = float(velocity)

    # Yaw angle
    yaw_deg = rot.yaw
    if yaw_deg < 0:
        yaw_deg += 360
    can_bus[16] = yaw_deg / 180.0 * math.pi
    can_bus[17] = yaw_deg

    return can_bus


def build_img_metas(carla_images, carla_camera_configs, carla_transform,
                    velocity=0.0):
    """Build img_metas dict for BEVFormer from CARLA data.

    Args:
        carla_images: dict mapping Tesla camera name → BGR image (numpy array)
        carla_camera_configs: dict mapping Tesla camera name → config dict
            (with keys: location, rotation, fov, width, height)
        carla_transform: CARLA Transform object (vehicle global pose)
        velocity: scalar forward speed (m/s)

    Returns:
        dict: img_metas with all required fields
    """
    lidar2img_list = []
    lidar2cam_list = []
    cam_intrinsic_list = []
    filenames = []

    for cam_name in NUSCENES_CAM_ORDER:
        tesla_name = TESLA_TO_NUSCENES[cam_name]
        cfg = carla_camera_configs[tesla_name]

        loc = cfg['location']
        rot = cfg['rotation']
        fov = cfg['fov']
        W, H = cfg['width'], cfg['height']

        # Match the yaw correction applied in spawn_tesla_cameras:
        # For side/rear cameras (|yaw| >= 45°), CARLA inverts the yaw
        # direction when attaching to a vehicle, so we negate it here
        # to keep the lidar2cam matrix consistent with the actual camera orientation.
        raw_yaw = rot[1]
        effective_yaw = -raw_yaw if abs(raw_yaw) >= 45 else raw_yaw

        # Compute matrices at ORIGINAL resolution
        l2i, K, l2c = build_lidar2img(loc, rot[0], effective_yaw, rot[2], fov, W, H)

        # Scale to processed resolution (BEVFormer expects this)
        scale_x = TARGET_IMG_W / W
        scale_y = TARGET_IMG_H / H
        scale_mat = np.eye(4, dtype=np.float64)
        scale_mat[0, 0] = scale_x
        scale_mat[1, 1] = scale_y
        l2i_scaled = scale_mat @ l2i

        lidar2img_list.append(l2i_scaled.astype(np.float32))
        lidar2cam_list.append(l2c.astype(np.float32))
        cam_intrinsic_list.append(K.astype(np.float32))
        filenames.append(f'carla_{tesla_name}.png')

    # Compute padded image shape
    pad_h = int(math.ceil(TARGET_IMG_H / PAD_SIZE_DIVISOR)) * PAD_SIZE_DIVISOR
    pad_w = int(math.ceil(TARGET_IMG_W / PAD_SIZE_DIVISOR)) * PAD_SIZE_DIVISOR
    img_shape = (pad_h, pad_w, 3)

    can_bus = build_can_bus(carla_transform, velocity)

    img_metas = {
        'lidar2img': lidar2img_list,
        'lidar2cam': lidar2cam_list,
        'cam_intrinsic': cam_intrinsic_list,
        'filename': filenames,
        'img_shape': [img_shape] * 6,
        'ori_shape': [img_shape] * 6,
        'pad_shape': [img_shape] * 6,
        'img_norm_cfg': dict(
            mean=IMG_MEAN, std=IMG_STD, to_rgb=True
        ),
        'scene_token': 'carla_scene',
        'prev_bev_exists': False,
        'can_bus': can_bus,
        'sample_idx': 'carla_frame',
        'box_type_3d': 'lidar',
        'box_mode_3d': 'lidar',
    }

    return img_metas


def prepare_img_tensor(carla_images):
    """Convert CARLA images to BEVFormer img tensor.

    Args:
        carla_images: dict mapping Tesla camera name → BGR image

    Returns:
        tensor: shape (num_cams=6, C=3, H, W)
    """
    tensors = []
    for cam_name in NUSCENES_CAM_ORDER:
        tesla_name = TESLA_TO_NUSCENES[cam_name]
        img = carla_images[tesla_name]
        t, _ = preprocess_image(img)
        tensors.append(t)
    return torch.stack(tensors, dim=0)  # (6, 3, H, W)


# ============================================================================
# Visualization
# ============================================================================

def draw_3d_detections_on_images(images, bbox_results, img_metas,
                                  score_thr=0.3):
    """Draw 3D detection boxes projected onto camera images.

    Args:
        images: list of 6 BGR images (numpy arrays)
        bbox_results: list of 6 dicts, each with keys:
            'boxes_3d': (N, 9) tensor [x, y, z, w, l, h, yaw, vx, vy]
            'scores_3d': (N,) tensor
            'labels_3d': (N,) tensor
        img_metas: dict with lidar2img matrices
        score_thr: minimum score threshold for visualization

    Returns:
        list: 6 annotated images
    """
    vis_images = [img.copy() for img in images]

    for cam_idx in range(6):
        result = bbox_results[cam_idx]
        if isinstance(result, dict) and 'pts_bbox' in result:
            result = result['pts_bbox']

        boxes = result.get('boxes_3d', None)
        scores = result.get('scores_3d', None)
        labels = result.get('labels_3d', None)

        if boxes is None or len(boxes) == 0:
            continue

        lidar2img = np.array(img_metas['lidar2img'][cam_idx])

        # Convert to numpy if tensor
        if hasattr(boxes, 'tensor'):
            boxes_np = boxes.tensor.cpu().numpy()
        elif isinstance(boxes, torch.Tensor):
            boxes_np = boxes.cpu().numpy()
        else:
            boxes_np = np.array(boxes)

        if hasattr(scores, 'cpu'):
            scores_np = scores.cpu().numpy()
        else:
            scores_np = np.array(scores)

        if hasattr(labels, 'cpu'):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = np.array(labels)

        for i in range(len(boxes_np)):
            if scores_np[i] < score_thr:
                continue

            box = boxes_np[i]
            label = int(labels_np[i])
            color = CLASS_COLORS[label % len(CLASS_COLORS)]

            # Box center and dimensions
            cx, cy, cz = box[0], box[1], box[2]
            w, l, h = box[3], box[4], box[5]
            yaw = box[6]

            # Generate 8 corners of the 3D box
            corners = _box_corners_3d(cx, cy, cz, w, l, h, yaw)

            # Project to image
            corners_2d = _project_points(corners, lidar2img)
            if corners_2d is None:
                continue

            # Check for valid (finite) coordinates
            if not np.all(np.isfinite(corners_2d)):
                continue

            H_img, W_img = vis_images[cam_idx].shape[:2]

            # Draw edges
            edges = [(0,1), (1,2), (2,3), (3,0),
                     (4,5), (5,6), (6,7), (7,4),
                     (0,4), (1,5), (2,6), (3,7)]
            for e in edges:
                x1, y1 = int(corners_2d[e[0], 0]), int(corners_2d[e[0], 1])
                x2, y2 = int(corners_2d[e[1], 0]), int(corners_2d[e[1], 1])
                # Clip to image bounds
                if (-1000 < x1 < W_img+1000 and -1000 < y1 < H_img+1000 and
                        -1000 < x2 < W_img+1000 and -1000 < y2 < H_img+1000):
                    cv2.line(vis_images[cam_idx], (x1, y1), (x2, y2), color, 1)

            # Score label
            score = scores_np[i]
            cls_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else '?'
            text = f'{cls_name} {score:.2f}'
            px, py = int(corners_2d[0, 0]), int(corners_2d[0, 1])
            if 0 <= px < W_img and 0 <= py < H_img:
                cv2.putText(vis_images[cam_idx], text, (px, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    return vis_images


def _box_corners_3d(cx, cy, cz, w, l, h, yaw):
    """Generate 8 corners of a 3D bounding box.

    Args:
        cx, cy, cz: center position
        w, l, h: width (y-axis), length (x-axis), height (z-axis)
        yaw: rotation around z-axis

    Returns:
        (8, 3) numpy array of corner positions
    """
    # Half dimensions
    dx = l / 2
    dy = w / 2
    dz = h / 2

    # 8 corners in local frame
    corners = np.array([
        [-dx, -dy, -dz], [-dx, -dy, +dz],
        [-dx, +dy, -dz], [-dx, +dy, +dz],
        [+dx, -dy, -dz], [+dx, -dy, +dz],
        [+dx, +dy, -dz], [+dx, +dy, +dz],
    ])

    # Rotation around z-axis
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    R_z = np.array([
        [cos_y, -sin_y, 0],
        [sin_y,  cos_y, 0],
        [0,      0,     1],
    ])

    # Transform to world frame
    corners = (R_z @ corners.T).T + np.array([cx, cy, cz])
    return corners


def _project_points(points_3d, lidar2img):
    """Project 3D points to 2D image coordinates.

    Args:
        points_3d: (N, 3) array
        lidar2img: (4, 4) projection matrix

    Returns:
        (N, 2) array of pixel coordinates, or None if all behind camera
    """
    N = len(points_3d)
    pts_h = np.hstack([points_3d, np.ones((N, 1))])  # (N, 4)
    pts_img = (lidar2img @ pts_h.T).T  # (N, 4)

    # Depth check (points must be in front of camera)
    depth = pts_img[:, 2]
    if np.all(depth <= 0.1):
        return None

    # Perspective division
    uv = pts_img[:, :2] / np.maximum(depth[:, None], 1e-5)
    return uv


# ============================================================================
# Quick Self-Test
# ============================================================================

def test_adapter():
    """Run adapter self-test with synthetic data."""
    from tesla_camera_layout import TESLA_CAMERAS

    print("=" * 60)
    print("CARLA → BEVFormer Adapter Self-Test")
    print("=" * 60)

    # Test 1: Camera mapping
    print("\n[1] Camera Mapping (Tesla 8 → nuScenes 6):")
    for nusc_name, tesla_name in TESLA_TO_NUSCENES.items():
        cam = TESLA_CAMERAS[tesla_name]
        print(f"  {nusc_name:18s} ← {tesla_name:18s} "
              f"FOV={cam['fov']:6.1f}°  pos={cam['location']}")

    # Test 2: Matrix dimensions
    print("\n[2] Matrix Dimensions:")
    for cam_name in NUSCENES_CAM_ORDER:
        tesla_name = TESLA_TO_NUSCENES[cam_name]
        cfg = TESLA_CAMERAS[tesla_name]
        loc = cfg['location']
        rot = cfg['rotation']
        fov = cfg['fov']
        W, H = cfg['width'], cfg['height']

        l2i, K, l2c = build_lidar2img(
            loc, rot[0], rot[1], rot[2], fov, W, H
        )
        print(f"  {cam_name:18s}: K={K.shape}  "
              f"lidar2cam={l2c.shape}  lidar2img={l2i.shape}")

    # Test 3: Image preprocessing
    print("\n[3] Image Preprocessing:")
    dummy_img = np.random.randint(0, 255, (960, 1280, 3), dtype=np.uint8)
    tensor, shape = preprocess_image(dummy_img)
    print(f"  Input:  {dummy_img.shape}  (BGR, original)")
    print(f"  Output: {tuple(tensor.shape)}  (CHW, normalized+padded)")
    print(f"  Shape:  {shape}")

    # Test 4: Full pipeline
    print("\n[4] Full Pipeline (mock CARLA data):")
    mock_images = {}
    for name in TESLA_CAMERAS:
        mock_images[name] = np.random.randint(
            0, 255, (960, 1280, 3), dtype=np.uint8
        )

    class MockLocation:
        def __init__(self):
            self.x, self.y, self.z = 0.0, 0.0, 0.0

    class MockRotation:
        def __init__(self):
            self.pitch, self.yaw, self.roll = 0.0, 0.0, 0.0

    class MockTransform:
        def __init__(self):
            self.location = MockLocation()
            self.rotation = MockRotation()

    img_tensor = prepare_img_tensor(mock_images)
    print(f"  img tensor: {img_tensor.shape}  "
          f"dtype={img_tensor.dtype}  "
          f"range=[{img_tensor.min():.2f}, {img_tensor.max():.2f}]")

    img_metas = build_img_metas(
        mock_images, TESLA_CAMERAS, MockTransform(), velocity=10.0
    )
    print(f"  lidar2img:    {len(img_metas['lidar2img'])} × "
          f"{img_metas['lidar2img'][0].shape}")
    print(f"  lidar2cam:    {len(img_metas['lidar2cam'])} × "
          f"{img_metas['lidar2cam'][0].shape}")
    print(f"  cam_intrinsic: {len(img_metas['cam_intrinsic'])} × "
          f"{img_metas['cam_intrinsic'][0].shape}")
    print(f"  can_bus:      {img_metas['can_bus'].shape}  "
          f"pos=({img_metas['can_bus'][0]:.1f}, "
          f"{img_metas['can_bus'][1]:.1f}, "
          f"{img_metas['can_bus'][2]:.1f})")
    print(f"  scene_token:  {img_metas['scene_token']}")
    print(f"  img_shape:    {img_metas['img_shape'][0]}")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
    return True


if __name__ == '__main__':
    test_adapter()
