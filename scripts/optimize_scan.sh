#!/bin/bash
# 1. Reduce num_routes from 5 to 3 in h1_quick mode
sed -i "s/'num_routes': 5,/'num_routes': 3,/" ~/carla-adversarial/scripts/collapse_configs.py
echo "Config updated:"
grep -A2 "h1_quick" ~/carla-adversarial/scripts/collapse_configs.py | grep num_routes

# 2. Clean old results
rm -rf ~/carla-adversarial/results/collapse_scan/h1_quick
echo "Old results cleaned"

# 3. Verify
echo "New h1_quick mode:"
grep -A8 "'h1_quick'" ~/carla-adversarial/scripts/collapse_configs.py
