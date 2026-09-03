#!/usr/bin/env python3
"""
8-Camera BEV (Bird's Eye View) Surround-View Stitching for Tesla FSD in CARLA.

Approach (ref: CSDN tgj891 4-camera BEV series):
  1. For each camera, compute homography H from image -> ground plane (z=0)
  2. Warp each camera image to a common BEV canvas
  3. Composite all 8 warped images with alpha blending

Uses known camera intrinsics (from FOV) and extrinsics (from TESLA_CAMERAS config)
instead of chessboard calibration, since CARLA provides exact poses.

Coordinate systems:
  - CARLA (world): x=forward, y=left, z=up (left-handed)
  - OpenCV (camera): x=right, y=down, z=forward (right-handed)
  - BEV output: x=right, y=up (image coordinates)
"""

import sys
import os
import math
import numpy as np
import cv2

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CONFIG_DIR)

from tesla_camera_layout import TESLA_CAMERAS, get_intrinsic_matrix, VEHICLE_HALF_WIDTH


# ============================================================================
# BEV Canvas Configuration
# ============================================================================
BEV_PIXELS_PER_METER = 40       # resolution: 40 pixels per meter
BEV_RANGE_X = (-6, 8)           # forward range: -6m (rear) to +8m (front)
BEV_RANGE_Y = (-6, 6)           # lateral range: -6m (right) to +6m (left)

BEV_WIDTH = int((BEV_RANGE_X[1] - BEV_RANGE_X[0]) * BEV_PIXELS_PER_METER)
BEV_HEIGHT = int((BEV_RANGE_Y[1] - BEV_RANGE_Y[0]) * BEV_PIXELS_PER_METER)


def euler_to_rotation_matrix(pitch_deg, yaw_deg, roll_deg):
    """Convert Euler angles (degrees) to 3x3 rotation matrix.

    CARLA convention: pitch (around Y), yaw (around Z), roll (around X)
    Returns rotation matrix R such that world_pt = R * local_pt
    """
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    r = math.radians(roll_deg)

    # Rotation around X (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(r), -math.sin(r)],
        [0, math.sin(r), math.cos(r)]
    ])
    # Rotation around Y (pitch)
    Ry = np.array([
        [math.cos(p), 0, math.sin(p)],
        [0, 1, 0],
        [-math.sin(p), 0, math.cos(p)]
    ])
    # Rotation around Z (yaw)
    Rz = np.array([
        [math.cos(y), -math.sin(y), 0],
        [math.sin(y), math.cos(y), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx


def carla_to_opencv_transform(location, rotation):
    """Convert CARLA camera pose to OpenCV extrinsic (R, t).

    CARLA world: x=fwd, y=left, z=up (left-handed)
    OpenCV camera: x=right, y=down, z=fwd (right-handed)

    The conversion flips y-axis: x_cv=x_carla, y_cv=-y_carla, z_cv=z_carla
    """
    pitch, yaw, roll = rotation
    tx, ty, tz = location

    # Rotation: CARLA -> OpenCV
    R_carla = euler_to_rotation_matrix(pitch, yaw, roll)

    # Coordinate flip matrix: CARLA -> OpenCV
    T_flip = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    # OpenCV rotation: R_cv = T_flip @ R_carla @ T_flip
    R_cv = T_flip @ R_carla @ T_flip

    # Translation in OpenCV frame
    t_cv = T_flip @ np.array([tx, ty, tz])

    # Extrinsic: world_to_camera = R_cv^T, t = -R_cv^T @ t_cv
    R_ext = R_cv.T
    t_ext = -R_cv.T @ t_cv

    return R_ext, t_ext


def compute_ground_homography(camera_name):
    """Compute 3x3 homography H that maps image pixel -> ground plane (BEV coords).

    For ground plane z=0 in world frame:
      p_image ~ K @ [R | t] @ P_ground
    where P_ground = [X, Y, 0, 1]^T

    The homography is the first two columns of [R|t] plus the translation column.
    """
    cam = TESLA_CAMERAS[camera_name]
    K = np.array(get_intrinsic_matrix(camera_name), dtype=np.float64)
    R_ext, t_ext = carla_to_opencv_transform(cam['location'], cam['rotation'])

    # Projection matrix P = K @ [R | t]  (3x4)
    P = K @ np.hstack([R_ext, t_ext.reshape(3, 1)])

    # For ground plane (z_world=0), the OpenCV Y coordinate after flip is -y_carla
    # In OpenCV frame, ground plane is at y_cv = -z_carla (but z_carla=0 for ground)
    # Actually, ground is at z_carla=0, which maps to z_cv=0 after our flip...
    # Wait: CARLA z=up, ground is at z=0. After flip: z_cv = z_carla = 0.
    # So in OpenCV frame, ground points have z_cv=0.
    # The homography uses columns 0 and 1 of P (corresponding to x_cv and y_cv)
    # But we need to map to world x,y coordinates for BEV.

    # Actually, let's compute it differently:
    # World point on ground: P_w = [X, Y, 0, 1] in CARLA frame
    # Convert to OpenCV: P_cv = [X, -Y, 0, 1] (flip y)
    # Image point: p = K @ [R_ext | t_ext] @ P_cv
    # Homography: H maps [X, -Y, 1] -> p
    # So H = K @ [R_ext[:, 0], R_ext[:, 1], t_ext] (columns 0,1 of R_ext + t_ext)

    # But we want the INVERSE: image -> ground
    # H_img_to_ground = inv(K @ [r1, r2, t])
    # Then ground point = H_img_to_ground @ pixel

    # Build 3x3 matrix: columns are R_ext col0, R_ext col1, t_ext
    # This maps [x_cv, y_cv, 1] (ground in OpenCV frame) to image pixel
    H_ground_to_img = P[:, [0, 1, 3]]  # columns 0, 1, 3 of 3x4 matrix

    # Inverse: image -> ground (OpenCV frame)
    H_img_to_ground_cv = np.linalg.inv(H_ground_to_img)

    return H_img_to_ground_cv


def world_to_bev_pixel(x_world, y_world):
    """Convert world coordinates (CARLA: x=fwd, y=left) to BEV pixel coordinates.

    BEV image: x_pixel = right, y_pixel = down
    We map: world_x -> BEV y (inverted: forward = up in BEV)
            world_y -> BEV x (inverted: left = left in BEV)
    """
    bev_x = int((y_world - BEV_RANGE_Y[0]) * BEV_PIXELS_PER_METER)
    bev_y = int((BEV_RANGE_X[1] - x_world) * BEV_PIXELS_PER_METER)
    return bev_x, bev_y


def create_bev_canvas():
    """Create an empty BEV canvas (black background)."""
    canvas = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
    return canvas


def draw_vehicle_on_bev(canvas):
    """Draw a simple vehicle rectangle on the BEV canvas."""
    # Tesla Model 3: 4.69m x 1.85m
    half_w = VEHICLE_HALF_WIDTH
    half_l = 2.345

    # Vehicle corners in world coords (center at origin)
    corners = [
        (-half_l, half_w),   # front-left
        (-half_l, -half_w),  # front-right
        (half_l, -half_w),   # rear-right
        (half_l, half_w),    # rear-left
    ]

    # Convert to BEV pixels
    pts = []
    for x, y in corners:
        px, py = world_to_bev_pixel(x, y)
        pts.append([px, py])
    pts = np.array(pts, dtype=np.int32)

    cv2.polylines(canvas, [pts], True, (0, 255, 255), 2)  # yellow outline
    return canvas


def warp_image_to_bev(image, camera_name):
    """Warp a single camera image to BEV perspective using homography."""
    H_img_to_ground_cv = compute_ground_homography(camera_name)

    # We need to map BEV pixels -> image pixels
    # BEV pixel (u, v) -> world ground point -> OpenCV ground point -> image pixel

    # Build BEV pixel -> world ground mapping
    # BEV pixel (px, py) -> world (x, y)
    # x_world = BEV_RANGE_X[1] - py / PIXELS_PER_METER
    # y_world = BEV_RANGE_Y[0] + px / PIXELS_PER_METER

    # Then world -> OpenCV ground: x_cv = x_world, y_cv = -y_world
    # Then OpenCV ground -> image: p = H_ground_to_img @ [x_cv, y_cv, 1]

    # Combined: BEV pixel -> OpenCV ground
    # x_cv = BEV_RANGE_X[1] - py / PPM
    # y_cv = -(BEV_RANGE_Y[0] + px / PPM)

    # This is an affine transform from BEV pixel to OpenCV ground:
    # [x_cv]   [0,     -1/PPM] [px]   [BEV_RANGE_X[1]    ]
    # [y_cv] = [-1/PPM, 0    ] [py] + [-BEV_RANGE_Y[0]    ]
    # [ 1  ]   [0,      0    ] [ 1]   [1                  ]

    ppm = BEV_PIXELS_PER_METER
    M_bev_to_cv_ground = np.array([
        [0, -1.0 / ppm, BEV_RANGE_X[1]],
        [-1.0 / ppm, 0, -BEV_RANGE_Y[0]],
        [0, 0, 1]
    ], dtype=np.float64)

    # H_ground_cv_to_img = inv(H_img_to_ground_cv)
    H_ground_cv_to_img = np.linalg.inv(H_img_to_ground_cv)

    # Full mapping: BEV pixel -> image pixel
    H_bev_to_img = H_ground_cv_to_img @ M_bev_to_cv_ground

    # Warp: for each BEV pixel, find corresponding image pixel
    warped = cv2.warpPerspective(
        image, H_bev_to_img, (BEV_WIDTH, BEV_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return warped


def create_weight_mask(warped_image):
    """Create a weight mask for blending based on distance from image center.

    Pixels closer to the center of the warped image get higher weight,
    which helps reduce seams at the boundaries.
    """
    h, w = warped_image.shape[:2]
    # Create distance-from-center map
    cy, cx = h // 2, w // 2
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
    max_dist = math.sqrt(cx ** 2 + cy ** 2)
    # Normalize to [0, 1], closer to center = higher weight
    weight = 1.0 - (dist / max_dist) * 0.5
    # Zero out black regions
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    mask = gray > 5  # non-black pixels
    weight = weight * mask.astype(np.float32)
    return weight


def stitch_bev(images_dict):
    """Stitch 8 camera images into a BEV surround view.

    Args:
        images_dict: dict of {camera_name: BGR image (numpy array)}

    Returns:
        bev_image: stitched BEV image (numpy array)
    """
    canvas = create_bev_canvas()
    weight_sum = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.float32)
    color_sum = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.float32)

    for name, image in images_dict.items():
        if image is None:
            print(f"[WARN] {name}: no image, skipping")
            continue

        # Warp to BEV
        warped = warp_image_to_bev(image, name)

        # Compute weight mask
        weight = create_weight_mask(warped)

        # Accumulate weighted colors
        weight_3ch = np.stack([weight] * 3, axis=-1)
        color_sum += warped.astype(np.float32) * weight_3ch
        weight_sum += weight_3ch

    # Normalize by weight
    mask = weight_sum[:, :, 0] > 0
    for c in range(3):
        channel = color_sum[:, :, c]
        channel[mask] /= weight_sum[:, :, c][mask]
        color_sum[:, :, c] = channel

    canvas = np.clip(color_sum, 0, 255).astype(np.uint8)

    # Draw vehicle outline
    canvas = draw_vehicle_on_bev(canvas)

    return canvas


def stitch_bev_simple(images_dict):
    """Simpler BEV stitching: just overlay without blending.

    Later cameras overwrite earlier ones. Good for quick visualization.
    """
    canvas = create_bev_canvas()

    # Priority order: front cameras first, then sides, then rear
    priority = [
        'rear',
        'side_rear_left', 'side_rear_right',
        'side_front_left', 'side_front_right',
        'front_wide', 'front_main', 'front_narrow',
    ]

    for name in priority:
        if name not in images_dict or images_dict[name] is None:
            continue
        warped = warp_image_to_bev(images_dict[name], name)
        # Only overwrite non-black pixels
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mask = gray > 5
        canvas[mask] = warped[mask]

    canvas = draw_vehicle_on_bev(canvas)
    return canvas


def save_individual_warps(images_dict, output_dir):
    """Save each camera's warped BEV image individually for debugging."""
    os.makedirs(output_dir, exist_ok=True)
    for name, image in images_dict.items():
        if image is None:
            continue
        warped = warp_image_to_bev(image, name)
        path = os.path.join(output_dir, f'bev_{name}.png')
        cv2.imwrite(path, warped)
        print(f"[SAVED] {name} warp -> {path}")


def create_grid_view(images_dict, output_path):
    """Create a grid view of all 8 camera images for overview."""
    names = list(images_dict.keys())
    n = len(names)
    cols = 4
    rows = (n + cols - 1) // cols

    # Get image size from first valid image
    h, w = None, None
    for name in names:
        if images_dict[name] is not None:
            h, w = images_dict[name].shape[:2]
            break
    if h is None:
        return

    # Resize for grid
    cell_h, cell_w = h // 2, w // 2
    grid = np.zeros((rows * (cell_h + 30), cols * cell_w, 3), dtype=np.uint8)

    for i, name in enumerate(names):
        if images_dict[name] is None:
            continue
        r, c = divmod(i, cols)
        img = cv2.resize(images_dict[name], (cell_w, cell_h))
        y0 = r * (cell_h + 30)
        x0 = c * cell_w
        grid[y0:y0 + cell_h, x0:x0 + cell_w] = img
        # Add label
        cv2.putText(grid, name, (x0 + 5, y0 + cell_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(output_path, grid)
    print(f"[SAVED] Grid view -> {output_path}")


if __name__ == '__main__':
    # Test with saved images from setup_tesla_cameras.py
    output_dir = os.path.join(os.path.expanduser('~'), 'carla-adversarial', 'output')

    images = {}
    for name in TESLA_CAMERAS:
        path = os.path.join(output_dir, f'{name}.png')
        if os.path.exists(path):
            img = cv2.imread(path)
            images[name] = img
            print(f"[LOADED] {name}: {img.shape}")
        else:
            print(f"[MISS] {name}: {path} not found")
            images[name] = None

    if len([v for v in images.values() if v is not None]) == 0:
        print("No images found. Run setup_tesla_cameras.py first.")
        sys.exit(1)

    # Save individual warps
    print("\n--- Saving individual BEV warps ---")
    save_individual_warps(images, output_dir)

    # Stitch BEV (simple overlay)
    print("\n--- Stitching BEV (simple overlay) ---")
    bev_simple = stitch_bev_simple(images)
    path_simple = os.path.join(output_dir, 'bev_surround_simple.png')
    cv2.imwrite(path_simple, bev_simple)
    print(f"[SAVED] BEV simple -> {path_simple}")

    # Stitch BEV (weighted blend)
    print("\n--- Stitching BEV (weighted blend) ---")
    bev_blend = stitch_bev(images)
    path_blend = os.path.join(output_dir, 'bev_surround_blend.png')
    cv2.imwrite(path_blend, bev_blend)
    print(f"[SAVED] BEV blend -> {path_blend}")

    # Grid view
    print("\n--- Creating grid view ---")
    path_grid = os.path.join(output_dir, 'camera_grid.png')
    create_grid_view(images, path_grid)

    print("\n[DONE] All BEV outputs saved.")
