"""
collapse_configs.py -- Phase 3B collapse point scanning experiment matrix

Defines the experimental conditions for systematically scanning the collapse
point of BEVFormer's detection head under varying attack intensities, camera
combinations, and duty cycles.

Dependencies: attack_configs (no CARLA required)
"""

from attack_configs import AttackConfig, ALL_CAMERAS, BEV_CONSUMED_CAMERAS


# ============================================================================
# 1. Attack intensity gradient (independent variable 1)
#    patch_frac = fraction of image area covered by adversarial patch
# ============================================================================

INTENSITY_LEVELS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0]
# Note: AttackConfig validates patch_frac in [0.05, 0.5].
# 0.05 = minimum baseline (smallest possible patch).
# Values > 0.5 are clamped to 0.5 by the scanner.

# Labels for file naming
INTENSITY_LABELS = {v: f'{v:.2f}' for v in INTENSITY_LEVELS}


# ============================================================================
# 2. Camera groups (independent variable 2)
#    Single / Dual / Triple camera combinations
# ============================================================================

CAMERA_GROUPS = {
    # --- Single cameras (8 groups) --- test H1: which camera is most critical
    'single_front_main':      ('front_main',),
    'single_front_wide':      ('front_wide',),
    'single_side_fl':         ('side_front_left',),
    'single_side_fr':         ('side_front_right',),
    'single_side_rl':         ('side_rear_left',),
    'single_rear':            ('rear',),
    'single_front_narrow':    ('front_narrow',),       # negative control
    'single_side_rr':         ('side_rear_right',),    # negative control

    # --- Dual cameras (5 groups) --- test H2: synergy effect
    'dual_front':             ('front_main', 'front_wide'),
    'dual_left':              ('side_front_left', 'side_rear_left'),
    'dual_right':             ('side_front_right', 'side_rear_right'),
    'dual_front_left':        ('front_main', 'side_front_left'),
    'dual_front_right':       ('front_main', 'side_front_right'),

    # --- Triple cameras (2 groups) --- extreme scenarios
    'triple_front':           ('front_main', 'front_wide', 'side_front_left'),
    'triple_all_front':       ('front_main', 'side_front_left', 'side_front_right'),
}
# Total: 15 groups

# Negative control cameras (not in BEV_CONSUMED_CAMERAS)
NEGATIVE_CONTROL_CAMS = {'front_narrow', 'side_rear_right'}

# BEV-consumed cameras only (for filtering)
BEV_CAMERA_GROUPS = {
    k: v for k, v in CAMERA_GROUPS.items()
    if all(c in BEV_CONSUMED_CAMERAS for c in v)
}
# 6 single + 5 dual + 2 triple = 13 BEV-active groups


# ============================================================================
# 3. Duty cycle gradient (independent variable 3, for H3)
# ============================================================================

DUTY_CYCLES = [1.0, 0.7, 0.5, 0.3, 0.1]


# ============================================================================
# 4. Fixed parameters (consistent with Phase 2)
# ============================================================================

ATTACK_TYPE = 'patch_sticker'
ONSET_FRAME = 60
DURATION_FRAMES = 100
PATTERN_SEED = 42
PERIOD_FRAMES = 100
NUM_ROUTES_DEFAULT = 10
FRAMES_PER_ROUTE_DEFAULT = 200


# ============================================================================
# 5. Experiment modes
# ============================================================================

EXPERIMENT_MODES = {
    'h1_quick': {
        'description': 'H1 fast validation: 8 single cameras x 3 intensities x 5 routes',
        'camera_groups': {k: v for k, v in CAMERA_GROUPS.items()
                          if k.startswith('single_')},
        'intensities': [0.05, 0.5, 1.0],
        'duty_cycles': [1.0],
        'num_routes': 3,
    },
    'h1_full': {
        'description': 'H1 full scan: 8 single cameras x 8 intensities x 10 routes',
        'camera_groups': {k: v for k, v in CAMERA_GROUPS.items()
                          if k.startswith('single_')},
        'intensities': INTENSITY_LEVELS,
        'duty_cycles': [1.0],
        'num_routes': 10,
    },
    'h2_full': {
        'description': 'H2 full scan: all 15 groups x 8 intensities x 10 routes',
        'camera_groups': CAMERA_GROUPS,
        'intensities': INTENSITY_LEVELS,
        'duty_cycles': [1.0],
        'num_routes': 10,
    },
    'h3_full': {
        'description': 'H3 duty cycle: 3 representative cameras x 4 intensities x 5 duty cycles',
        'camera_groups': {
            'single_front_main': ('front_main',),
            'single_front_wide': ('front_wide',),
            'single_side_fl':    ('side_front_left',),
        },
        'intensities': [0.05, 0.2, 0.5, 1.0],  # 0.5 clamped from 0.6
        'duty_cycles': DUTY_CYCLES,
        'num_routes': 10,
    },
    'full': {
        'description': 'Full matrix: all 15 groups x 8 intensities x 5 duty cycles',
        'camera_groups': CAMERA_GROUPS,
        'intensities': INTENSITY_LEVELS,
        'duty_cycles': DUTY_CYCLES,
        'num_routes': 10,
    },
}


# ============================================================================
# 6. Config generation
# ============================================================================

def make_attack_config(group_id, intensity, duty_cycle=1.0):
    """
    Dynamically create an AttackConfig for a collapse scanning condition.

    Args:
        group_id: key from CAMERA_GROUPS
        intensity: patch_frac value (0.0 = no attack, 1.0 = full coverage)
        duty_cycle: fraction of time the attack is active

    Returns:
        AttackConfig instance
    """
    cameras = CAMERA_GROUPS[group_id]
    attack_id = f'collapse_{group_id}_i{intensity:.2f}_d{duty_cycle:.1f}'

    # Clamp to valid patch_frac range [0.05, 0.5]
    patch_frac = max(0.05, min(intensity, 0.5))

    return AttackConfig(
        attack_id=attack_id,
        attack_type=ATTACK_TYPE,
        target_cameras=cameras,
        patch_frac=patch_frac,
        onset_frame=ONSET_FRAME,
        duration_frames=DURATION_FRAMES,
        duty_cycle=duty_cycle,
        period_frames=PERIOD_FRAMES,
        pattern_seed=PATTERN_SEED,
        notes=f'collapse scan: {group_id}, intensity={intensity}, duty={duty_cycle}',
    )


def generate_experiment_matrix(mode=None, camera_groups=None,
                               intensities=None, duty_cycles=None):
    """
    Generate the full list of (group_id, intensity, duty_cycle) conditions.

    Args:
        mode: experiment mode name (from EXPERIMENT_MODES). If provided,
              overrides the other parameters.
        camera_groups: dict of group_id -> camera tuple
        intensities: list of float
        duty_cycles: list of float

    Returns:
        list of (group_id, intensity, duty_cycle) tuples
        mode_info dict with metadata
    """
    if mode is not None:
        if mode not in EXPERIMENT_MODES:
            raise ValueError(f'Unknown mode: {mode}. Available: {list(EXPERIMENT_MODES.keys())}')
        m = EXPERIMENT_MODES[mode]
        camera_groups = m['camera_groups']
        intensities = m['intensities']
        duty_cycles = m['duty_cycles']
        num_routes = m['num_routes']
    else:
        num_routes = NUM_ROUTES_DEFAULT

    conditions = []
    for gid in sorted(camera_groups.keys()):
        for intensity in intensities:
            for duty in duty_cycles:
                conditions.append((gid, intensity, duty))

    mode_info = {
        'mode': mode or 'custom',
        'num_groups': len(camera_groups),
        'num_intensities': len(intensities),
        'num_duty_cycles': len(duty_cycles),
        'num_conditions': len(conditions),
        'num_routes': num_routes,
        'total_routes': len(conditions) * num_routes,
    }
    return conditions, mode_info


def condition_to_dirname(group_id, intensity, duty_cycle):
    """Convert condition parameters to a directory name."""
    return f'{group_id}__i{intensity:.2f}__d{duty_cycle:.1f}'


def dirname_to_condition(dirname):
    """Parse a condition directory name back to (group_id, intensity, duty_cycle)."""
    parts = dirname.split('__')
    group_id = parts[0]
    intensity = float(parts[1].lstrip('i'))
    duty_cycle = float(parts[2].lstrip('d'))
    return group_id, intensity, duty_cycle


# ============================================================================
# Self-test
# ============================================================================

if __name__ == '__main__':
    import sys

    print('=' * 70)
    print('collapse_configs.py self-test')
    print('=' * 70)

    # 1. Camera group validation
    print('\n[1] Camera group validation')
    for gid, cams in CAMERA_GROUPS.items():
        for c in cams:
            assert c in ALL_CAMERAS, f'{gid}: unknown camera {c}'
        is_neg = any(c in NEGATIVE_CONTROL_CAMS for c in cams)
        is_bev = all(c in BEV_CONSUMED_CAMERAS for c in cams)
        tag = 'NEG_CTRL' if is_neg else ('BEV_ACTIVE' if is_bev else 'MIXED')
        print(f'  {gid:<25} cams={",".join(cams):<45} [{tag}]')
    print(f'  Total: {len(CAMERA_GROUPS)} groups')

    # 2. Experiment matrix sizes
    print('\n[2] Experiment matrix sizes')
    for mode_name, mode_info in EXPERIMENT_MODES.items():
        conds, info = generate_experiment_matrix(mode=mode_name)
        print(f'  {mode_name:<12} {info["num_conditions"]:>4} conditions x '
              f'{info["num_routes"]:>2} routes = {info["total_routes"]:>5} routes')

    # 3. Config generation test
    print('\n[3] Config generation test')
    cfg = make_attack_config('single_front_main', 0.25, 1.0)
    print(f'  attack_id: {cfg.attack_id}')
    print(f'  target_cameras: {cfg.target_cameras}')
    print(f'  patch_frac: {cfg.patch_frac}')
    print(f'  duty_cycle: {cfg.duty_cycle}')
    assert cfg.patch_frac == 0.25
    assert cfg.target_cameras == ('front_main',)

    cfg0 = make_attack_config('dual_front', 0.05, 0.5)
    print(f'  intensity=0.05 patch_frac: {cfg0.patch_frac}')
    assert cfg0.patch_frac == 0.05
    assert cfg0.duty_cycle == 0.5

    # Test clamping: intensity > 0.5 should clamp patch_frac to 0.5
    cfg_high = make_attack_config('single_front_main', 1.0, 1.0)
    print(f'  intensity=1.0 patch_frac: {cfg_high.patch_frac} (clamped to 0.5)')
    assert cfg_high.patch_frac == 0.5

    # 4. Directory naming roundtrip
    print('\n[4] Directory naming roundtrip')
    dname = condition_to_dirname('single_front_main', 0.25, 1.0)
    gid, inten, duty = dirname_to_condition(dname)
    print(f'  {dname} -> ({gid}, {inten}, {duty})')
    assert gid == 'single_front_main'
    assert abs(inten - 0.25) < 1e-6
    assert abs(duty - 1.0) < 1e-6

    print('\n' + '=' * 70)
    print('[PASS] All tests passed')
    sys.exit(0)
