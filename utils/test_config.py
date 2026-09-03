#!/usr/bin/env python3
"""Quick test to verify camera config and BEV module import."""
import sys
sys.path.insert(0, '.')

from tesla_camera_layout import TESLA_CAMERAS, get_intrinsic_matrix, get_camera_extrinsics

print(f"Total cameras: {len(TESLA_CAMERAS)}")
print()

for name in TESLA_CAMERAS:
    loc, rot = get_camera_extrinsics(name)
    K = get_intrinsic_matrix(name)
    cam = TESLA_CAMERAS[name]
    print(f"{name:16s}  pos=({loc[0]:5.1f},{loc[1]:5.1f},{loc[2]:5.1f})  "
          f"yaw={rot[1]:>6.1f}  fov={cam['fov']:.0f}  "
          f"fx={K[0][0]:.1f}  fy={K[1][1]:.1f}")

print()
print("Testing BEV module import...")
try:
    from bev_stitching import BEV_WIDTH, BEV_HEIGHT, BEV_PIXELS_PER_METER
    print(f"BEV canvas: {BEV_WIDTH}x{BEV_HEIGHT} px  ({BEV_PIXELS_PER_METER} px/m)")
    print("[OK] All modules imported successfully.")
except Exception as e:
    print(f"[ERROR] {e}")
