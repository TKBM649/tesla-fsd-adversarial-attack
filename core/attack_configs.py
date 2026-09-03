"""
attack_configs.py — 对抗攻击场景参数化配置（Phase 2 Step 1）

设计原则：
- 自包含：不依赖 carla / BEVFormer / carla_bev_adapter，可在无 GPU 环境跑单测
- 常量硬编码 + 注释说明来源，避免隐式依赖
- 所有字段 JSON 可序列化，便于 CLI 文件驱动
- 监控失明预检：在配置阶段就标出哪些场景会让 k=15 监控器结构性失明（H4 实验自变量）
"""

import json
from dataclasses import dataclass, asdict
from typing import Tuple, Optional

# ============================================================================
# 常量区（硬编码，注释说明来源）
# ============================================================================

FRAME_RATE = 20  # dt=0.05s，与 collect_baseline.py 同步模式一致

# Phase 1 定案判据：cos<0.999740 AND L2>0.027718 持续 15 帧才报警
MONITOR_PERSIST_FRAMES = 15

# 8 路相机键名（快照 dict 的键，target_cameras 必须用这套命名）
# 来源：tesla_camera_layout.TESLA_CAMERAS（WSL ~/carla-adversarial/scripts/）
ALL_CAMERAS = (
    'front_wide', 'front_main', 'front_narrow',
    'side_front_left', 'side_front_right',
    'side_rear_left', 'side_rear_right',
    'rear'
)

# 真正进入 BEVFormer 的 6 路（TESLA_TO_NUSCENES 值集）
# 来源：carla_bev_adapter.TESLA_TO_NUSCENES
# front_narrow 和 side_rear_right 在推理链路之外，攻击这两路 BEV 应零响应（阴性对照）
BEV_CONSUMED_CAMERAS = (
    'front_main', 'front_wide',
    'side_front_left', 'side_front_right',
    'side_rear_left', 'rear'
)

ATTACK_TYPES = ('patch_sticker', 'occlusion', 'glare', 'blur', 'dropout')


# ============================================================================
# AttackConfig 数据类
# ============================================================================

@dataclass
class AttackConfig:
    attack_id: str
    attack_type: str
    target_cameras: Tuple[str, ...]
    
    # Patch 参数（patch_sticker / occlusion 用）
    patch_frac: float = 0.25          # 占画面面积比
    patch_anchor: str = 'center_lower'  # v1 固定；'sign_projection' 预留 v1.5
    opacity: float = 1.0
    
    # Blur 参数
    kernel_size: int = 11             # 奇数 3-15
    
    # Glare 参数
    glare_intensity: float = 0.8      # 加性高光场幅度
    
    # 时序参数（帧单位）
    onset_frame: int = 60             # 攻击起始帧（ABAB 设计默认 60）
    duration_frames: int = 100        # 攻击总窗口
    duty_cycle: float = 1.0           # 1.0=持续；<1 间歇
    period_frames: int = 100          # 间歇周期
    
    # 图案 RNG 种子（保证逐帧可复现）
    pattern_seed: int = 42
    
    # 自由文本
    notes: str = ''
    
    def __post_init__(self):
        # 硬校验
        if self.attack_type not in ATTACK_TYPES:
            raise ValueError(f'attack_type={self.attack_type!r} not in {ATTACK_TYPES}')
        
        invalid_cams = set(self.target_cameras) - set(ALL_CAMERAS)
        if invalid_cams:
            raise ValueError(f'target_cameras contains invalid names: {invalid_cams}')
        
        if not (0.05 <= self.patch_frac <= 0.5):
            raise ValueError(f'patch_frac={self.patch_frac} out of [0.05, 0.5]')
        
        if not (0.0 <= self.opacity <= 1.0):
            raise ValueError(f'opacity={self.opacity} out of [0, 1]')
        
        if self.kernel_size % 2 == 0 or not (3 <= self.kernel_size <= 15):
            raise ValueError(f'kernel_size={self.kernel_size} must be odd in [3, 15]')
        
        if not (0.0 <= self.glare_intensity <= 1.0):
            raise ValueError(f'glare_intensity={self.glare_intensity} out of [0, 1]')
        
        if self.onset_frame < 0:
            raise ValueError(f'onset_frame={self.onset_frame} must be >= 0')
        
        if self.duration_frames <= 0:
            raise ValueError(f'duration_frames={self.duration_frames} must be > 0')
        
        if not (0.0 < self.duty_cycle <= 1.0):
            raise ValueError(f'duty_cycle={self.duty_cycle} out of (0, 1]')
        
        if self.period_frames < 4:
            raise ValueError(f'period_frames={self.period_frames} must be >= 4')
        
        # 语义归一：duty_cycle==1.0 时强制 period_frames == duration_frames
        if self.duty_cycle == 1.0 and self.period_frames != self.duration_frames:
            self.period_frames = self.duration_frames
    
    @property
    def max_active_run(self) -> int:
        """单次攻击最长连续活跃帧数（用于监控可见性判定）"""
        if self.duty_cycle == 1.0:
            return self.duration_frames
        return int(self.duty_cycle * self.period_frames)
    
    @property
    def blind_to_monitor(self) -> bool:
        """
        是否会让 k=15 监控器结构性失明（H4 实验自变量）
        条件：间歇攻击（duty<1）且单次连续活跃帧 < 15
        """
        if self.duty_cycle == 1.0:
            return False
        return self.max_active_run < MONITOR_PERSIST_FRAMES
    
    @property
    def active_frames_per_period(self) -> int:
        """每个周期内活跃帧数"""
        return self.max_active_run
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['target_cameras'] = list(d['target_cameras'])
        d['max_active_run'] = self.max_active_run
        d['blind_to_monitor'] = self.blind_to_monitor
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'AttackConfig':
        d = d.copy()
        d['target_cameras'] = tuple(d['target_cameras'])
        d.pop('max_active_run', None)
        d.pop('blind_to_monitor', None)
        return cls(**d)


# ============================================================================
# 预置场景表
# ============================================================================

ATTACK_SCENARIOS = {
    'sign_patch_front': AttackConfig(
        attack_id='sign_patch_front',
        attack_type='patch_sticker',
        target_cameras=('front_main',),
        patch_frac=0.25,
        opacity=1.0,
        onset_frame=60,
        duration_frames=100,
        duty_cycle=1.0,
        notes='路标对抗贴纸（前视单相机，持续攻击）'
    ),
    
    'road_mark_intermittent': AttackConfig(
        attack_id='road_mark_intermittent',
        attack_type='patch_sticker',
        target_cameras=('front_main', 'front_wide'),
        patch_frac=0.15,
        opacity=1.0,
        onset_frame=60,
        duration_frames=100,
        duty_cycle=0.3,
        period_frames=100,
        notes='路面标记间歇攻击（双前视，duty 0.3，run 30≥15 可见）'
    ),
    
    'fast_intermittent_blind': AttackConfig(
        attack_id='fast_intermittent_blind',
        attack_type='patch_sticker',
        target_cameras=('front_main',),
        patch_frac=0.25,
        opacity=1.0,
        onset_frame=60,
        duration_frames=100,
        duty_cycle=0.5,
        period_frames=10,
        notes='快速间歇失明探针（duty 0.5, period 10, run 5<15 结构性失明）'
    ),
    
    'camera_degradation_blur': AttackConfig(
        attack_id='camera_degradation_blur',
        attack_type='blur',
        target_cameras=('front_main',),
        kernel_size=11,
        onset_frame=60,
        duration_frames=100,
        duty_cycle=1.0,
        notes='摄像头降级攻击（高斯模糊，模拟 NHTSA 事故场景）'
    ),
    
    'glare_dazzle_front': AttackConfig(
        attack_id='glare_dazzle_front',
        attack_type='glare',
        target_cameras=('front_main',),
        glare_intensity=0.8,
        onset_frame=60,
        duration_frames=100,
        duty_cycle=1.0,
        notes='眩光致盲攻击（加性高光场）'
    ),
    
    'dropout_dead_cam': AttackConfig(
        attack_id='dropout_dead_cam',
        attack_type='dropout',
        target_cameras=('front_narrow',),  # 阴性对照：不在 BEV_CONSUMED_CAMERAS 内
        onset_frame=60,
        duration_frames=100,
        duty_cycle=1.0,
        notes='阴性对照（攻击 front_narrow，BEV 应零响应）'
    ),
}


# ============================================================================
# 序列化 + 批量校验
# ============================================================================

def save_json(config: AttackConfig, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)

def load_json(path: str) -> AttackConfig:
    with open(path, 'r', encoding='utf-8') as f:
        return AttackConfig.from_dict(json.load(f))

def validate_all() -> list:
    """遍历所有预置场景，返回 [(attack_id, is_valid, error_msg)]"""
    results = []
    for aid, cfg in ATTACK_SCENARIOS.items():
        try:
            # 重新触发 __post_init__ 校验，过滤掉派生属性
            d = cfg.to_dict()
            d.pop('max_active_run', None)
            d.pop('blind_to_monitor', None)
            AttackConfig(**d)
            results.append((aid, True, ''))
        except Exception as e:
            results.append((aid, False, str(e)))
    return results


# ============================================================================
# 自测块（1.2）
# ============================================================================

if __name__ == '__main__':
    import sys
    
    print('=' * 70)
    print('attack_configs.py 自测')
    print('=' * 70)
    
    # 1. 打印场景摘要表
    print('\n[1] 预置场景摘要表')
    print(f'{"attack_id":<30} {"type":<15} {"cameras":<30} {"blind":<6}')
    print('-' * 85)
    for aid, cfg in ATTACK_SCENARIOS.items():
        cams = ', '.join(cfg.target_cameras)
        blind = 'YES' if cfg.blind_to_monitor else 'no'
        print(f'{aid:<30} {cfg.attack_type:<15} {cams:<30} {blind:<6}')
    
    # 2. 三个负例必须抛异常
    print('\n[2] 负例校验（必须抛异常）')
    negative_cases = [
        ('非法相机名', dict(attack_id='bad', attack_type='patch_sticker',
                            target_cameras=('cam_99',), patch_frac=0.25,
                            kernel_size=11, glare_intensity=0.8,
                            onset_frame=60, duration_frames=100,
                            duty_cycle=1.0, period_frames=100)),
        ('duty_cycle=0', dict(attack_id='bad', attack_type='patch_sticker',
                              target_cameras=('front_main',), patch_frac=0.25,
                              kernel_size=11, glare_intensity=0.8,
                              onset_frame=60, duration_frames=100,
                              duty_cycle=0.0, period_frames=100)),
        ('kernel_size 偶数', dict(attack_id='bad', attack_type='blur',
                                  target_cameras=('front_main',), patch_frac=0.25,
                                  kernel_size=10, glare_intensity=0.8,
                                  onset_frame=60, duration_frames=100,
                                  duty_cycle=1.0, period_frames=100)),
    ]
    for desc, kwargs in negative_cases:
        try:
            AttackConfig(**kwargs)
            print(f'  [FAIL] {desc}: did not raise (FAIL)')
            sys.exit(1)
        except ValueError as e:
            print(f'  [PASS] {desc}: {e}')
    
    # 3. 批量校验
    print('\n[3] 批量校验 validate_all()')
    results = validate_all()
    all_ok = True
    for aid, is_valid, err in results:
        status = '[PASS]' if is_valid else '[FAIL]'
        print(f'  {status} {aid}: {err if err else "OK"}')
        if not is_valid:
            all_ok = False
    
    if not all_ok:
        print('\n[FAIL] Batch validation failed')
        sys.exit(1)
    
    # 4. Adapter 一致性检查（可选，需 WSL venv）
    if '--check-adapter' in sys.argv:
        print('\n[4] Adapter 一致性检查')
        try:
            from carla_bev_adapter import TESLA_TO_NUSCENES
            from tesla_camera_layout import TESLA_CAMERAS
            
            adapter_cams = set(TESLA_TO_NUSCENES.values())
            layout_cams = set(TESLA_CAMERAS.keys())
            
            if adapter_cams == set(BEV_CONSUMED_CAMERAS):
                print(f'  [PASS] BEV_CONSUMED_CAMERAS matches adapter')
            else:
                print(f'  [FAIL] BEV_CONSUMED_CAMERAS mismatch: expected {adapter_cams}, got {set(BEV_CONSUMED_CAMERAS)}')
                all_ok = False
            
            if layout_cams == set(ALL_CAMERAS):
                print(f'  [PASS] ALL_CAMERAS matches layout')
            else:
                print(f'  [FAIL] ALL_CAMERAS mismatch: expected {layout_cams}, got {set(ALL_CAMERAS)}')
                all_ok = False
            
        except ImportError as e:
            print(f'  [WARN] import failed (must run in WSL venv): {e}')
    
    print('\n' + '=' * 70)
    if all_ok:
        print('[PASS] All tests passed')
        sys.exit(0)
    else:
        print('[FAIL] Some tests failed')
        sys.exit(1)
