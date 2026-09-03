import carla
c = carla.Client("localhost", 2000)
c.set_timeout(5)
w = c.get_world()
print("CARLA OK, map:", w.get_map().name)
