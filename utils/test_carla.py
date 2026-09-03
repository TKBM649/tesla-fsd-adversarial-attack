#!/usr/bin/env python3
"""Quick CARLA connection test."""
import carla
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    print(f"[OK] Connected! Map: {world.get_map().name}")
    vehicles = world.get_actors().filter('vehicle.*')
    print(f"[OK] Vehicles in world: {len(vehicles)}")
except Exception as e:
    print(f"[ERROR] {e}")
