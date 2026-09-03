import sys
sys.path.insert(0, '/home/cwq/carla-adversarial/config')
from tesla_camera_layout import TESLA_CAMERAS
print(len(TESLA_CAMERAS), 'cameras loaded')
for k, v in TESLA_CAMERAS.items():
    print(f'  {k}: loc={v["location"]}, yaw={v["rotation"][1]}')
