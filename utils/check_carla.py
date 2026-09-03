#!/usr/bin/env python3
import carla
try:
    c = carla.Client('localhost', 2000)
    c.set_timeout(10)
    w = c.get_world()
    print(f"CARLA OK: {w.get_map().name}")
except Exception as e:
    print(f"CARLA FAIL: {e}")
