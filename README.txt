================================================================================
  Tesla FSD Adversarial Attack Experiment - Organized Project Files
================================================================================

Project: CARLA Simulator + BEVFormer Perception Model Adversarial Attack
Platform: Windows (CARLA Server) + WSL2 Ubuntu 22.04 (Python Client / ML)
GPU: NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)

================================================================================
  Directory Structure
================================================================================

config/
  tesla_camera_layout.py    - Tesla 8-camera layout parameters

core/
  attack_configs.py         - Attack scenario definitions (6 presets + dataclass)
  attack_injector.py        - Image-space adversarial patch injection engine
  collapse_configs.py       - Collapse point scan experiment matrix config

pipeline/
  setup_tesla_cameras.py    - Phase 1: Spawn 8 Tesla cameras in CARLA
  bev_stitching.py          - Phase 1: BEV surround-view stitching
  carla_bev_adapter.py      - Phase 1/2: CARLA-BEVFormer bridge adapter
  run_bevformer_carla.py    - Phase 1/2: Online inference main program
  run_bev_demo.py           - Phase 1: BEV demo visualization
  collect_baseline.py       - Phase 2: Baseline data collection
  collect_attack.py         - Phase 2: Attack data collection
  run_collapse_experiment.py - Phase 3B: Collapse point scan CLI entry

analysis/
  analyze_baseline.py       - Baseline statistics analysis
  analyze_collapse_points.py - Collapse point detection + figures
  analysis_baseline_vs_attack.py - Baseline vs attack comparison
  finalize_baseline_stats.py - Finalize baseline statistics
  collapse_point_scanner.py  - Collapse point scanning engine

utils/                      - Test and utility scripts (13 files)
scripts/                    - Shell helper scripts (3 files)
results/                    - Experiment data (baseline, attack, scan)
output/                     - BEV visualization PNGs (23 files)

================================================================================
  Naming Convention: snake_case (Python standard)
  Total: 40 source code files + 3 shell scripts + 73 data/image files
================================================================================
