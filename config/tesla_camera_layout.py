"""
Tesla FSD Hardware 3.0 8-Camera Surround-View Layout
Based on official Tesla specifications (tesla.cn/autopilot)

Camera positions relative to vehicle center (x=forward, y=left, z=up), unit: meters
FOV and detection range from Tesla official documentation.

HW 3.0 8 cameras:
  - 3x front (windshield): wide 120deg/60m, main ~50deg/150m, narrow ~35deg/250m
  - 2x side-front (B-pillar): 90deg/80m
  - 2x side-rear (front fender): ~90deg/100m
  - 1x rear (license plate): 140deg/50m
"""

TESLA_CAMERAS = {
    # === Front triplet (mounted behind windshield, high position) ===
    "front_wide": {
        "location": [1.8, -0.15, 1.5],    # slightly right of center (passenger side)
        "rotation": [-3.0, 5.0, 0.0],     # slight yaw right
        "fov": 120.0,                       # fish-eye lens
        "width": 1280,
        "height": 960,
        "fps": 36,
        "detection_range": 60,
    },
    "front_main": {
        "location": [1.8, 0.0, 1.5],      # center of windshield
        "rotation": [-3.0, 0.0, 0.0],
        "fov": 50.0,
        "width": 1280,
        "height": 960,
        "fps": 36,
        "detection_range": 150,
    },
    "front_narrow": {
        "location": [1.8, 0.15, 1.5],     # slightly left of center (driver side)
        "rotation": [-3.0, -5.0, 0.0],    # slight yaw left
        "fov": 35.0,
        "width": 1280,
        "height": 960,
        "fps": 36,
        "detection_range": 250,
    },

    # === Side-front cameras (B-pillar, looking sideways-forward) ===
    "side_front_left": {
        "location": [-0.3, 2.0, 1.5],    # left side, far outside vehicle
        "rotation": [-5.0, -90.0, 0.0],   # pure left view, slight down pitch
        "fov": 40.0,
        "width": 1280,
        "height": 960,
        "fps": 36,
        "detection_range": 80,
    },
    "side_front_right": {
        "location": [-0.3, -2.0, 1.5],   # right side, far outside vehicle
        "rotation": [-5.0, 90.0, 0.0],    # pure right view, slight down pitch
        "fov": 40.0,
        "width": 1280,
        "height": 960,
        "fps": 36,
        "detection_range": 80,
    },

    # === Side-rear cameras (front fender, looking sideways-backward) ===
    "side_rear_left": {
        "location": [0.5, 2.0, 1.5],     # left side rear, far outside vehicle
        "rotation": [-5.0, -90.0, 0.0],   # pure left view, slight down pitch
        "fov": 40.0,
        "width": 1280,
        "height": 960,
        "fps": 36,
        "detection_range": 100,
    },
    "side_rear_right": {
        "location": [0.5, -2.0, 1.5],    # right side rear, far outside vehicle
        "rotation": [-5.0, 90.0, 0.0],    # pure right view, slight down pitch
        "fov": 40.0,
        "width": 1280,
        "height": 960,
        "fps": 36,
        "detection_range": 100,
    },

    # === Rear camera (above license plate) ===
    "rear": {
        "location": [-2.2, 0.0, 1.1],     # rear of vehicle, low position
        "rotation": [0.0, 180.0, 0.0],    # looking backward
        "fov": 140.0,
        "width": 1280,
        "height": 960,
        "fps": 36,
        "detection_range": 50,
    },
}
