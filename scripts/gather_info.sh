#!/bin/bash
echo '=== CAMERA RESOLUTION ==='
grep '"width"' ~/carla-adversarial/scripts/tesla_camera_layout.py | head -2

echo '=== H1_QUICK CONFIG ==='
grep -A8 "'h1_quick'" ~/carla-adversarial/scripts/collapse_configs.py

echo '=== COLLAPSE REPORT KEY FINDINGS ==='
python3 << 'PYEOF'
import json
d = json.load(open("/home/cwq/carla-adversarial/results/collapse_scan/h1_quick/analysis/collapse_report.json"))
print(json.dumps(d["hypothesis_tests"]["H1_front_critical"], indent=2))
PYEOF

echo '=== PHASE2 REPORT EXISTS ==='
ls -la ~/carla-adversarial/results/PHASE2_REPORT.md 2>/dev/null || echo "No PHASE2_REPORT.md"

echo '=== ARCHIVE MANIFEST ==='
ls -la ~/carla-adversarial/results/ARCHIVE_MANIFEST.json 2>/dev/null || echo "No ARCHIVE_MANIFEST.json"
