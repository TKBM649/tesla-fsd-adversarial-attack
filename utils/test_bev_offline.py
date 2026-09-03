#!/usr/bin/env python3
"""Test BEV stitching with saved images (no CARLA connection needed)."""
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, '.')
from tesla_camera_layout import TESLA_CAMERAS
from bev_stitching import stitch_bev_simple, stitch_bev, create_grid_view

OUTPUT_DIR = os.path.join(os.path.expanduser('~'), 'carla-adversarial', 'output')

print("Loading saved images...")
images = {}
for name in TESLA_CAMERAS:
    path = os.path.join(OUTPUT_DIR, f'{name}.png')
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            images[name] = img
            print(f"[LOADED] {name}: {img.shape}")
        else:
            print(f"[ERROR] Failed to read {path}")
            images[name] = None
    else:
        print(f"[MISS] {path}")
        images[name] = None

n_valid = sum(1 for v in images.values() if v is not None)
print(f"\nLoaded {n_valid}/{len(TESLA_CAMERAS)} images")

if n_valid == 0:
    print("No images to process!")
    sys.exit(1)

# Test grid view
print("\n--- Creating grid view ---")
try:
    grid_path = os.path.join(OUTPUT_DIR, 'camera_grid.png')
    create_grid_view(images, grid_path)
    print(f"[OK] Grid saved: {grid_path}")
except Exception as e:
    print(f"[ERROR] Grid view failed: {e}")
    import traceback
    traceback.print_exc()

# Test BEV simple
print("\n--- BEV stitching (simple overlay) ---")
try:
    bev_simple = stitch_bev_simple(images)
    path = os.path.join(OUTPUT_DIR, 'bev_surround_simple.png')
    cv2.imwrite(path, bev_simple)
    print(f"[OK] BEV simple saved: {path}")
except Exception as e:
    print(f"[ERROR] BEV simple failed: {e}")
    import traceback
    traceback.print_exc()

# Test BEV blend
print("\n--- BEV stitching (weighted blend) ---")
try:
    bev_blend = stitch_bev(images)
    path = os.path.join(OUTPUT_DIR, 'bev_surround_blend.png')
    cv2.imwrite(path, bev_blend)
    print(f"[OK] BEV blend saved: {path}")
except Exception as e:
    print(f"[ERROR] BEV blend failed: {e}")
    import traceback
    traceback.print_exc()

print("\n[DONE] Test complete.")
