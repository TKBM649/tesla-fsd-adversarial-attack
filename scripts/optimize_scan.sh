#!/bin/bash
# Optimize h1_quick scan: reduce num_routes 5 -> 3 for faster iteration.
# Idempotent: safe to re-run; becomes a no-op once num_routes is already 3.
CONFIG="$HOME/carla-adversarial/scripts/collapse_configs.py"
SCAN_DIR="$HOME/carla-adversarial/results/collapse_scan/h1_quick"

# 1. Reduce num_routes from 5 to 3 in h1_quick mode (only if not already 3)
current=$(grep -A8 "'h1_quick'" "$CONFIG" | grep -oE "'num_routes': [0-9]+" | grep -oE '[0-9]+' | head -1)
echo "Current h1_quick num_routes: ${current:-unknown}"
if [ "$current" = "3" ]; then
    echo "Already optimized (num_routes=3); skipping sed."
else
    sed -i "s/'num_routes': 5,/'num_routes': 3,/" "$CONFIG"
    echo "Config updated:"
    grep -A2 "h1_quick" "$CONFIG" | grep num_routes
fi

# 2. Clean old results (only if the directory exists)
if [ -d "$SCAN_DIR" ]; then
    rm -rf "$SCAN_DIR"
    echo "Old results cleaned: $SCAN_DIR"
else
    echo "No old results to clean."
fi

# 3. Verify
echo "New h1_quick mode:"
grep -A8 "'h1_quick'" "$CONFIG"
