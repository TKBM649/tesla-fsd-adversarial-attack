"""
attack_injector.py — 图像空间对抗攻击注入器（Phase 2 Step 2）

设计原则：
- 纯图像空间操作，不依赖 CARLA / BEVFormer
- 消费 attack_configs.AttackConfig，支持 5 种攻击类型
- 间歇逻辑：duty_cycle × period 控制活跃窗口
- 确定性：相同 pattern_seed → 相同输出
- 返回 (modified_snapshot, metadata)，metadata 含 attack_active + active_cameras
"""

import numpy as np
import cv2
from attack_configs import AttackConfig


class AttackInjector:
    def __init__(self, config: AttackConfig):
        self.config = config
        self.rng = np.random.RandomState(config.pattern_seed)
    
    def apply(self, snapshot: dict, frame_idx: int) -> tuple:
        """
        注入攻击到快照
        
        Args:
            snapshot: dict mapping camera_name → BGR image (numpy array)
            frame_idx: 当前帧索引
        
        Returns:
            (modified_snapshot, metadata)
            metadata = {'attack_active': bool, 'active_cameras': list[str]}
        """
        # 间歇逻辑
        if frame_idx < self.config.onset_frame:
            return snapshot, {'attack_active': False, 'active_cameras': []}
        
        if self.config.duty_cycle == 1.0:
            active = True
        else:
            elapsed = frame_idx - self.config.onset_frame
            active = (elapsed % self.config.period_frames) < self.config.active_frames_per_period
        
        if not active:
            return snapshot, {'attack_active': False, 'active_cameras': []}
        
        # 应用攻击
        modified = {k: v.copy() for k, v in snapshot.items()}
        active_cams = []
        
        for cam in self.config.target_cameras:
            if cam not in modified:
                continue
            img = modified[cam]
            modified[cam] = self._apply_attack(img, cam)
            active_cams.append(cam)
        
        return modified, {'attack_active': True, 'active_cameras': active_cams}
    
    def _apply_attack(self, img: np.ndarray, cam_name: str) -> np.ndarray:
        """根据 attack_type 应用具体攻击"""
        if self.config.attack_type == 'patch_sticker':
            return self._apply_patch(img)
        elif self.config.attack_type == 'occlusion':
            return self._apply_occlusion(img)
        elif self.config.attack_type == 'blur':
            return self._apply_blur(img)
        elif self.config.attack_type == 'glare':
            return self._apply_glare(img)
        elif self.config.attack_type == 'dropout':
            return self._apply_dropout(img)
        else:
            raise ValueError(f'Unknown attack_type: {self.config.attack_type}')
    
    def _apply_patch(self, img: np.ndarray) -> np.ndarray:
        """patch_sticker: 在指定区域叠加对抗图案（随机色块）"""
        h, w = img.shape[:2]
        
        # 计算 patch 尺寸（保持宽高比，基于面积比例）
        patch_area = h * w * self.config.patch_frac
        patch_h = int(np.sqrt(patch_area))
        patch_w = patch_h  # 正方形
        
        # v1 锚点：center_lower（画面中心偏下 60% 处）
        y0 = int(h * 0.6)
        x0 = (w - patch_w) // 2
        
        # 边界检查
        y0 = max(0, min(y0, h - patch_h))
        x0 = max(0, min(x0, w - patch_w))
        
        # 生成图案（随机色块）
        patch = self.rng.randint(0, 256, (patch_h, patch_w, 3), dtype=np.uint8)
        
        # 叠加
        region = img[y0:y0+patch_h, x0:x0+patch_w]
        blended = cv2.addWeighted(region, 1 - self.config.opacity, patch, self.config.opacity, 0)
        img[y0:y0+patch_h, x0:x0+patch_w] = blended
        
        return img
    
    def _apply_occlusion(self, img: np.ndarray) -> np.ndarray:
        """occlusion: 区域填充黑色"""
        h, w = img.shape[:2]
        patch_area = h * w * self.config.patch_frac
        patch_h = int(np.sqrt(patch_area))
        patch_w = patch_h
        
        y0 = int(h * 0.6)
        x0 = (w - patch_w) // 2
        y0 = max(0, min(y0, h - patch_h))
        x0 = max(0, min(x0, w - patch_w))
        
        img[y0:y0+patch_h, x0:x0+patch_w] = 0
        return img
    
    def _apply_blur(self, img: np.ndarray) -> np.ndarray:
        """blur: 高斯模糊（整图）"""
        return cv2.GaussianBlur(img, (self.config.kernel_size, self.config.kernel_size), 0)
    
    def _apply_glare(self, img: np.ndarray) -> np.ndarray:
        """glare: 加性高光场（整图增亮）"""
        glare = np.ones_like(img, dtype=np.float32) * (255 * self.config.glare_intensity)
        return np.clip(img.astype(np.float32) + glare, 0, 255).astype(np.uint8)
    
    def _apply_dropout(self, img: np.ndarray) -> np.ndarray:
        """dropout: 全零（模拟摄像头失效）"""
        return np.zeros_like(img)


# ============================================================================
# 自测块（2.2）
# ============================================================================

if __name__ == '__main__':
    import sys
    from attack_configs import ATTACK_SCENARIOS, ALL_CAMERAS
    
    print('=' * 70)
    print('attack_injector.py 自测')
    print('=' * 70)
    
    # 1. 构造假快照（8 相机，900x1600x3 uint8）
    dummy_snapshot = {}
    for cam in ALL_CAMERAS:
        dummy_snapshot[cam] = np.random.randint(0, 256, (900, 1600, 3), dtype=np.uint8)
    
    # 2. 遍历 6 个预置场景，验证输出
    print('\n[1] 攻击类型验证')
    for aid, cfg in ATTACK_SCENARIOS.items():
        injector = AttackInjector(cfg)
        
        # 测试 onset 前（应无变化）
        mod, meta = injector.apply(dummy_snapshot, frame_idx=0)
        assert not meta['attack_active'], f'{aid}: frame 0 should be inactive'
        
        # 测试 onset 后（应有变化）
        mod, meta = injector.apply(dummy_snapshot, frame_idx=cfg.onset_frame + 10)
        assert meta['attack_active'], f'{aid}: frame {cfg.onset_frame+10} should be active'
        assert len(meta['active_cameras']) > 0, f'{aid}: no active cameras'
        
        # 验证形状不变
        for cam in cfg.target_cameras:
            if cam in mod:
                assert mod[cam].shape == dummy_snapshot[cam].shape, f'{aid}: shape mismatch for {cam}'
        
        # 验证 dropout 全零
        if cfg.attack_type == 'dropout':
            for cam in meta['active_cameras']:
                assert np.all(mod[cam] == 0), f'{aid}: dropout should be all zeros for {cam}'
        
        # 验证 blur 像素变化
        if cfg.attack_type == 'blur':
            for cam in meta['active_cameras']:
                assert not np.array_equal(mod[cam], dummy_snapshot[cam]), f'{aid}: blur should change pixels'
        
        # 验证 glare 像素增亮
        if cfg.attack_type == 'glare':
            for cam in meta['active_cameras']:
                assert mod[cam].mean() > dummy_snapshot[cam].mean(), f'{aid}: glare should brighten'
        
        print(f'  [PASS] {aid}: {cfg.attack_type}')
    
    # 3. 间歇逻辑验证
    print('\n[2] 间歇逻辑验证')
    cfg = ATTACK_SCENARIOS['fast_intermittent_blind']  # duty 0.5, period 10
    injector = AttackInjector(cfg)
    
    # frame 60 (onset) → active
    _, meta = injector.apply(dummy_snapshot, 60)
    assert meta['attack_active'], 'frame 60 should be active'
    
    # frame 65 (elapsed 5, period 10, duty 0.5 → active run 5) → inactive
    _, meta = injector.apply(dummy_snapshot, 65)
    assert not meta['attack_active'], 'frame 65 should be inactive'
    
    # frame 70 (elapsed 10, next cycle) → active
    _, meta = injector.apply(dummy_snapshot, 70)
    assert meta['attack_active'], 'frame 70 should be active'
    
    print('  [PASS] Intermittent logic correct')
    
    # 4. 确定性验证
    print('\n[3] 确定性验证')
    cfg = ATTACK_SCENARIOS['sign_patch_front']
    inj1 = AttackInjector(cfg)
    inj2 = AttackInjector(cfg)
    
    mod1, _ = inj1.apply(dummy_snapshot, cfg.onset_frame)
    mod2, _ = inj2.apply(dummy_snapshot, cfg.onset_frame)
    
    for cam in cfg.target_cameras:
        if cam in mod1:
            assert np.array_equal(mod1[cam], mod2[cam]), f'{cam}: not deterministic'
    
    print('  [PASS] Deterministic (same seed -> same output)')
    
    print('\n' + '=' * 70)
    print('[PASS] All tests passed')
    sys.exit(0)
