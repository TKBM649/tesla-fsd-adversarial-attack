import os
d = '/home/cwq/carla-adversarial/output'
for f in sorted(os.listdir(d)):
    if f.endswith('.png'):
        path = os.path.join(d, f)
        size = os.path.getsize(path)
        print(f'{f}: {size/1024:.0f} KB')
