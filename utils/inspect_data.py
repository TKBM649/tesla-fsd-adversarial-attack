import json

# Load one baseline route
with open('/home/cwq/carla-adversarial/results/baseline_route_0.json') as f:
    bl = json.load(f)
print("BASELINE keys:", list(bl.keys()))
print("frame[0]:", json.dumps(bl['frames'][0], indent=2)[:600])
print("total frames:", len(bl['frames']))
print()

# Load one attack route
with open('/home/cwq/carla-adversarial/results/attack_sign_patch_front/attack_sign_patch_front_route_0.json') as f:
    at = json.load(f)
print("ATTACK keys:", list(at.keys()))
print("frame[0]:", json.dumps(at['frames'][0], indent=2)[:600])
print("frame[80]:", json.dumps(at['frames'][80], indent=2)[:600])
print("total frames:", len(at['frames']))
