#!/bin/bash
# Reduce camera resolution from 1280x960 to 640x480
sed -i 's/"width": 1280/"width": 640/g' ~/carla-adversarial/scripts/tesla_camera_layout.py
sed -i 's/"height": 960/"height": 480/g' ~/carla-adversarial/scripts/tesla_camera_layout.py
echo "Resolution updated:"
grep -n 'width\|height' ~/carla-adversarial/scripts/tesla_camera_layout.py | grep -E '640|480'
