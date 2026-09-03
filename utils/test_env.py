#!/usr/bin/env python3
"""Test mmdet3d import to diagnose SparseConv2d registration conflict."""
import sys
sys.path.insert(0, "/home/cwq/carla-adversarial/scripts")

print("[1] Testing basic imports...")
try:
    from run_bevformer_carla import run_inference, decode_results
    print("  [PASS] run_bevformer_carla imports OK")
except Exception as e:
    print(f"  [FAIL] {e}")

print("\n[2] Testing model load + inference...")
try:
    import torch
    from run_bevformer_carla import load_bevformer_model
    model = load_bevformer_model()
    print(f"  [PASS] Model loaded: {type(model).__name__}")
    
    # Test inference with dummy data
    import numpy as np
    dummy_imgs = {
        'front_wide': np.zeros((900, 1600, 3), dtype=np.uint8),
        'front_main': np.zeros((900, 1600, 3), dtype=np.uint8),
        'front_narrow': np.zeros((900, 1600, 3), dtype=np.uint8),
        'side_front_left': np.zeros((900, 1600, 3), dtype=np.uint8),
        'side_front_right': np.zeros((900, 1600, 3), dtype=np.uint8),
        'side_rear_left': np.zeros((900, 1600, 3), dtype=np.uint8),
        'side_rear_right': np.zeros((900, 1600, 3), dtype=np.uint8),
        'rear': np.zeros((900, 1600, 3), dtype=np.uint8),
    }
    from carla_bev_adapter import prepare_img_tensor, build_img_metas
    from tesla_camera_layout import TESLA_CAMERAS
    import carla
    
    img_tensor = prepare_img_tensor(dummy_imgs)
    # Need a dummy transform
    dummy_transform = carla.Transform()
    img_metas = build_img_metas(dummy_imgs, TESLA_CAMERAS, dummy_transform, velocity=0.0)
    
    print("[3] Running inference...")
    result, bev_embed = run_inference(model, img_tensor, img_metas)
    print(f"  [PASS] Inference OK: result type={type(result)}")
    
    decoded = decode_results(result)
    print(f"  [PASS] Decode OK: keys={list(decoded.keys())}")
    
except Exception as e:
    import traceback
    print(f"  [FAIL] {e}")
    traceback.print_exc()

print("\n[DONE]")
