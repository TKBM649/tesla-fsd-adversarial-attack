#!/usr/bin/env python3
"""Step-by-step CARLA pipeline diagnosis — pinpoint which call times out."""
import sys
import time

import carla

sys.path.insert(0, '/home/cwq/carla-adversarial/scripts')

def main():
    t0 = time.time()
    c = carla.Client('localhost', 2000)
    c.set_timeout(30.0)
    w = c.get_world()
    print(f"[{time.time()-t0:5.1f}s] 1. world OK: {w.get_map().name}", flush=True)

    bp = w.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    sp = w.get_map().get_spawn_points()[23]
    v = w.try_spawn_actor(bp, sp)
    if v is None:
        print("2. vehicle spawn FAILED (point blocked)", flush=True)
        return
    print(f"[{time.time()-t0:5.1f}s] 2. vehicle spawned: {v.id}", flush=True)

    from setup_tesla_cameras import spawn_tesla_cameras
    cams = spawn_tesla_cameras(v, w)
    print(f"[{time.time()-t0:5.1f}s] 3. cameras attached: {len(cams)}", flush=True)

    s = w.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 0.05
    w.apply_settings(s)
    print(f"[{time.time()-t0:5.1f}s] 4. sync mode ON", flush=True)

    for i in range(5):
        t1 = time.time()
        w.tick()
        print(f"[{time.time()-t0:5.1f}s] 5.{i} tick OK ({time.time()-t1:.2f}s/tick)",
              flush=True)

    try:
        v.set_autopilot(True, 8000)
        print(f"[{time.time()-t0:5.1f}s] 6. autopilot ON (tm 8000)", flush=True)
    except Exception as e:
        print(f"[{time.time()-t0:5.1f}s] 6. autopilot FAILED: {e}", flush=True)

    for i in range(5):
        t1 = time.time()
        w.tick()
        print(f"[{time.time()-t0:5.1f}s] 7.{i} tick with autopilot OK "
              f"({time.time()-t1:.2f}s/tick)", flush=True)

    print(f"[{time.time()-t0:5.1f}s] 8. cleanup", flush=True)
    v.set_autopilot(False, 8000)
    for cam in cams.values():
        cam.stop()
        cam.destroy()
    v.destroy()
    s = w.get_settings()
    s.synchronous_mode = False
    w.apply_settings(s)
    print(f"[{time.time()-t0:5.1f}s] ALL STEPS OK", flush=True)


if __name__ == '__main__':
    main()
