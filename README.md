# 🚗 Tesla FSD Adversarial Attack

> **纯视觉自动驾驶感知系统的对抗攻击实验平台**
> **纯Python易复现**
> CARLA 0.9.16 仿真 · BEVFormer-tiny 感知代理 · Tesla HW3.0 八相机环视布局复刻
> 图像空间攻击注入 · 崩溃点扫描 · Windows + WSL2 混合部署（8GB VRAM 可跑通）

[![Python 3.10.12](https://img.shields.io/badge/python-3.10.12-blue.svg)](https://www.python.org/downloads/release/python-31012/)
[![CARLA 0.9.16](https://img.shields.io/badge/CARLA-0.9.16-00B4D8.svg)](https://github.com/carla-simulator/carla/releases/tag/0.9.16)
[![PyTorch 1.11.0+cu117](https://img.shields.io/badge/pytorch-1.11.0%2Bcu117-EE4C2C.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-P1✅_P2✅-brightgreen.svg)](#-实验矩阵)
[![License](https://img.shields.io/badge/license-none-lightgrey.svg)](#-license)

---

## ✨ 项目特点

| 特点 | 说明 |
|---|---|
| 🚗 **Tesla HW3.0 八相机复刻** | 按官方参数配置 3 前视（120°/50°/35°）+ 2 侧前（B 柱）+ 2 侧后（前翼子板）+ 1 后视（140°），含安装位姿、FOV、探测距离 50–250 m |
| 🧠 **BEVFormer 感知代理** | BEVFormer-tiny（ResNet-50 backbone，BEV 50×50，nuScenes 预训练）；8 路 Tesla 相机经 `TESLA_TO_NUSCENES` 映射为 6 路 nuScenes 相机序，另 2 路作阴性对照 |
| 🖼️ **图像空间攻击注入** | 5 类攻击（`patch_sticker`/`occlusion`/`glare`/`blur`/`dropout`）纯图像域实现，不改动 CARLA 与 BEVFormer 内部 |
| ⏱️ **间歇攻击 + 确定性复现** | `duty_cycle`/`period_frames` 控制间歇节奏；`pattern_seed=42` 保证逐帧图案可复现；`blind_to_monitor` 属性预判 k=15 监控器是否结构性失明 |
| 🧪 **崩溃点扫描矩阵** | 15 相机组 × 7 强度 × 5 占空比 → 5 档预设模式（24 / 56 / 105 / 60 / 525 条件），内建断点续扫 |
| 📏 **鲁棒阈值标定** | median ± 3·MAD 标定 BEV 监控阈值，单帧误报 43 → 叠加 15 帧持续性判据后误报事件 **0** |
| 🪟 **8GB VRAM 混合部署** | Windows CARLA 服务端 + WSL2 Ubuntu 22.04 客户端；分辨率 1280×960 → 640×480 降级后 GPU 利用率从 0%（冻结）恢复到 70% |
| 📊 **实验数据全量入库** | 108 个跟踪文件 / 78.0 MB：2000 帧基线 + 1956 帧攻击 + 崩溃扫描元数据 + 42 张图 |

---

## 🏗️ 架构

```
tesla-fsd-adversarial-attack/
├── config/       # 车辆与传感器定义
│   └── tesla_camera_layout.py     TESLA_CAMERAS：8 相机位姿 / FOV / 分辨率 / 探测距离
├── core/         # 攻击内核（不依赖 CARLA 与 BEVFormer，可离线自检）
│   ├── attack_configs.py          AttackConfig 数据类 + 6 个预置场景 + __post_init__ 硬校验
│   ├── attack_injector.py         AttackInjector：5 类图像空间攻击 + 间歇逻辑 + 种子化 RNG
│   └── collapse_configs.py        15 相机组 × 7 强度 × 5 占空比 + 5 档实验模式
├── pipeline/     # CARLA 采集主流程
│   ├── setup_tesla_cameras.py     spawn_tesla_cameras()：|yaw| ≥ 45° 条件取反修正
│   ├── bev_stitching.py           环视 BEV 拼接
│   ├── carla_bev_adapter.py       CARLA → BEVFormer 桥接（800×450 + ImageNet 归一化）
│   ├── run_bevformer_carla.py     在线/离线推理主程序（--test / --online / --offline）
│   ├── run_bev_demo.py            BEV 可视化 demo
│   ├── collect_baseline.py        Phase 2 基线采集（断点续采 + 冻结恢复）
│   ├── collect_attack.py          Phase 2 攻击采集（ABAB 设计 + 逐帧对比图）
│   └── run_collapse_experiment.py Phase 3B 扫描 CLI（--mode / --resume）
├── analysis/     # 统计与可视化
│   ├── analyze_baseline.py             阈值标定（robust median ± 3·MAD）
│   ├── finalize_baseline_stats.py      基线统计定稿
│   ├── analysis_baseline_vs_attack.py  基线 vs 攻击对比 + 5 图
│   ├── collapse_point_scanner.py       扫描引擎（复用 collect_attack 帧循环）
│   └── analyze_collapse_points.py      崩溃点检测（derivative/threshold/ratio）+ 6 图
├── utils/        # 13 个自检与调试脚本（CARLA 连通性 / 环境 / 断点续扫 / BEV 离线）
└── scripts/      # 3 个 Shell 辅助（fix_resolution / gather_info / optimize_scan）
```

**数据流**：

```
CARLA 0.9.16 (Windows · Town10HD_Opt · 同步模式 dt=0.05s / 20 FPS)
   ↓  8 路相机 BGR 帧（累积式 frame_buffer，回调只 attach 一次并跨路线共享）
AttackInjector.apply(snapshot, frame_idx)     ← onset_frame / duty_cycle / period_frames + seed=42
   ↓  carla_bev_adapter：resize 800×450 → PAD_SIZE_DIVISOR=32 补齐 → ImageNet 归一化 → 8→6 相机映射
BEVFormer-tiny (WSL2 · PyTorch 1.11.0+cu117)
   ↓  300 个检测 slot（NMS-free head 固定输出）+ BEV embed
指标层：det_count@score_thr=0.05 · bev_self_sim(cosine) · bev_l2_dist · speed
   ↓
results/*.json  →  analysis/  →  图表 + 量化报告
```

---

## 🧪 实验矩阵

| 阶段 | 名称 | 规模 | 状态 | 结果位置 |
|---|---|---|---|---|
| **Phase 1** | 八相机 spawn + BEV 拼接 + 在线推理闭环 | 8 相机 → 6 路 BEV 输入 | ✅ 完成 | `output/`（23 PNG） |
| **Phase 2** | 基线采集（无攻击） | 10 路线 × 200 帧 = **2000 帧** | ✅ 完成 | `results/baseline/` |
| **Phase 2** | `sign_patch_front` 攻击采集 | 10 路线 = **1956 帧**（攻击活跃 1356 / 非活跃 600） | ✅ 完成 | `results/attack_sign_patch_front/` |
| **Phase 2** | 基线 vs 攻击对比分析 | 5 图 + 量化报告 | ✅ 完成 | `results/analysis/` |
| **Phase 3B** | 崩溃点扫描 `h1_quick` | 24 条件 × 3 路线 × 100 帧 = 72 路线 | ✅ **24/24** | `results/collapse_scan/` |

> 矩阵规模由 `python core/collapse_configs.py` 自检**实算输出**，非文档估算（源码 docstring 中标称的 64/120/600 条件为过期值，`INTENSITY_LEVELS` 实为 7 档）。

---

## 📊 核心结果

### Phase 2 基线（10 路线 / 2000 帧 / `score_thr=0.05`）

仓库内存在两种统计口径，均由实际数据复算验证：

| 指标 | 全帧池化<br>（2000 帧，= `analysis_report.json`） | 按路线均值再平均<br>（排除 12 个场景跳变帧，= `baseline_stats.json`） |
|---|---|---|
| det_count_mean | **1.4935** | **1.4935** ± 0.9804 |
| det_count_raw_mean | 300.0（NMS-free 固定输出） | 300.0 |
| det_score_mean | — | 0.037692 ± 0.015501 |
| bev_cosine_mean | 0.99019318 | 0.998174 ± 0.002468 |
| bev_l2_mean | 0.015294 | 0.015427 ± 0.005256 |
| speed_mean | — | 5.211 m/s |

两者 cosine/L2 差异来自 12 个场景跳变帧（传送重置瞬间，单帧 cosine 低至 0.4994）：池化口径保留，路线均值口径剔除。

**监控阈值标定**（`analyze_baseline.py`，robust median ± 3·MAD）：

| 项 | 值 |
|---|---|
| BEV cosine 阈值 | < **0.999740** |
| BEV L2 阈值 | > **0.027718** |
| 持续性判据 | 15 连续帧 |
| 单帧误报 | 43 帧 |
| 叠加持续性后误报事件 | **0** |

### Phase 2 攻击（`sign_patch_front`）

攻击配置：`patch_sticker` / `front_main` / `patch_frac=0.25` / `opacity=1.0` / `onset_frame=60` / `duration_frames=100` / `duty_cycle=1.0` / `pattern_seed=42`

下列数值由仓库内 10 个 `attack_sign_patch_front_route_*.json` 与 10 个 `baseline_route_*.json` **逐帧复算**得出：

| 指标 | 基线<br>(n=2000) | 攻击 ON<br>(n=1356) | 攻击 OFF<br>(n=600) | Δ(ON − 基线) |
|---|---|---|---|---|
| det_count_mean | 1.4935 | **19.1755** | 18.4533 | **+17.68（×12.84）** |
| bev_cosine_mean | 0.99019318 | 0.95574673 | 0.98332928 | −0.03444645 |
| bev_l2_mean | 0.015294 | 0.003064 | 0.002632 | −0.012230 |
| **监控检出率** | — | **0 / 10 路线（0.0%）** | — | 完全漏检 |
| 误报（基线事件 / 攻击帧） | — | 0 / 0 | — | — |

> `analysis_report.json` 与 `archive_manifest.json` 记录的攻击组帧数为 1965（ON 1365 / OFF 600），比仓库内原始 JSON 求和多 **9 帧**，对应指标记录值为 det 19.1495 / cos 0.95603836 / L2 0.0030864 —— 与复算值在第 3–4 位有效数字上一致。基线侧则完全吻合（n=2000，det 1.4935，cos 0.99019318，L2 0.015294）。详见「已知限制」第 9 条。

**逐路线明细**（`attack_stats_sign_patch_front.json`）：

| 路线 | 帧数 | det_count_mean | det_max | BEV cosine | BEV L2 | speed_mean (m/s) | 攻击帧 | recoveries |
|---|---|---|---|---|---|---|---|---|
| route_0 | 188 | 0.000 | 0 | 0.99999606 | 0.002712 | 0.0048 | 128 | 3 |
| route_1 | 195 | 1.000 | 1 | 0.99999679 | 0.002386 | 0.0091 | 135 | 3 |
| route_2 | 198 | 18.854 | 19 | 0.99999467 | 0.003133 | 0.0108 | 138 | 3 |
| **route_3** | 198 | **51.949** | 53 | 0.99999397 | 0.003268 | 0.0108 | 138 | 3 |
| route_4 | 194 | 1.000 | 1 | 0.99999537 | 0.002867 | 0.0077 | 134 | 3 |
| route_5 | 195 | 5.021 | 6 | 0.99999534 | 0.002902 | 0.0098 | 135 | 3 |
| **route_6** | 199 | **53.729** | 55 | 0.99999290 | 0.003541 | 0.0118 | 139 | 3 |
| route_7 | 199 | 2.000 | 2 | 0.99999460 | 0.003025 | 0.0140 | 139 | 3 |
| route_8 | 191 | 0.576 | 1 | 0.99999501 | 0.002979 | 0.0070 | 131 | 3 |
| **route_9** | 199 | **52.698** | 53 | 0.99999308 | 0.003550 | 0.0118 | 139 | 3 |

对照基线同路线 det_count_mean：R3 = 4.175、R6 = 1.27、R9 = 1.49（`archive_manifest.json` 标记的高敏感路线 `["R3","R6","R9"]` 与此一致）。

**关键发现**：

1. **攻击效果高度场景依赖** —— R3/R6/R9 检测数放大到 50+，而 R0/R4 几乎无变化（0.0 / 1.0），R2 中等（18.85）。均值 19.18 不代表典型路线，中位数仅约 1.0。
2. **BEV 特征监控器对图像空间攻击完全失效** —— 攻击下 BEV L2 ≈ 0.003，远低于阈值 0.027718；BEV cosine 反而高于基线（0.999994 vs 0.998174）。说明 BEV 编码器鲁棒，脆弱点在**检测头**，Phase 4 防御应针对检测头异常而非特征漂移。
3. **⚠️ 检测数抬升主要是会话级效应，而非逐帧 patch 效应** —— 攻击 **OFF** 段（patch 未激活，n=600）det_count_mean 已达 **18.45**，与 ON 段 19.18 仅差 3.9%，而两者都是基线 1.49 的 12 倍以上。patch 的边际贡献远小于采集会话本身带来的状态差异（车辆近乎静止 + 每路线 3 次冻结恢复传送）。**引用 +17.68 这一增益时必须同时说明此混淆因素。**

### Phase 3B 崩溃点扫描（`h1_quick`，24/24 条件完成）

来源：`experiment_meta.json` + `scan_progress.json` + `analysis/collapse_report.json`

| 项 | 值 |
|---|---|
| 模式 | `h1_quick`：8 单相机组 × 3 强度 [0.05, 0.5, 1.0] × 1 占空比 [1.0] |
| 条件完成数 | **24 / 24** |
| 每条件规模 | 3 路线 × 100 帧（warmup 8 帧，`score_thr=0.05`） |
| 起止时间 | 2026-09-03 18:12:48 → 19:11:29 |
| 实际耗时 | **3520.4 s（≈ 58.7 min）** |
| 已分析条件 | 8（derivative 法） |

**崩溃点定位**（一阶导最大处对应的强度网格中点）：

| 相机组 | 是否进入 BEV 链路 | 崩溃强度（derivative） | threshold 法 | ratio |
|---|---|---|---|---|
| `single_front_main` | ✅ | **0.275** | — | — |
| `single_front_wide` | ✅ | **0.275** | — | — |
| `single_side_fr` | ✅ | 0.275 | — | — |
| `single_side_rl` | ✅ | 0.275 | 0.5 | 0.5 |
| `single_rear` | ✅ | 0.750 | — | — |
| `single_side_fl` | ✅ | 0.750 | — | — |

**假设检验结果**：

| 假设 | 结果 |
|---|---|
| **H1** 前视相机最关键 | `most_vulnerable = single_front_main`，`least_vulnerable = single_rear` → **部分支持**（前视主/广角崩溃强度最低 0.275，但 `single_side_fr`/`single_side_rl` 同为 0.275） |
| **H2** 多相机协同效应 | `median_collapse_single = 0.275`；`median_collapse_dual = null`、`median_collapse_triple = null`、`confirmed = null` → **未验证**（需 `h2_full`） |
| **H3** 占空比影响 | `h3_full` 未执行 → **无数据** |

---

## 🚀 快速开始

### 环境要求

| 组件 | 版本 / 规格 |
|---|---|
| OS | Windows 10/11（CARLA 服务端）+ WSL 2 Ubuntu 22.04（Python 客户端） |
| GPU | NVIDIA **RTX 4060+（8GB VRAM 起）**；推荐 RTX 3070/3080/3090/4090（16GB+）以支持更高分辨率 |
| RAM / 存储 | 16GB+ / 50GB+（CARLA 7.3GB + BEVFormer 2GB + env 3GB + results） |
| CARLA | **0.9.16**（Windows，端口 2000，地图 `Town10HD_Opt`） |
| Python | 3.10.12（虚拟环境 `~/bevformer-env`） |
| PyTorch / CUDA | **1.11.0+cu117**（必须 ≤ 1.13，mmcv-full 1.4.8 兼容性硬约束）/ 11.7 |
| mmcv-full / mmdet / mmdet3d | 1.4.8 / 2.14.0 / 0.17.1 |

### 安装

```bash
# 1) WSL2 + Ubuntu 22.04（Windows PowerShell 管理员）
wsl --install -d Ubuntu-22.04

# 2) 系统依赖（WSL 内）
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev python3-venv \
  python3-pip libvulkan1 libgl1-mesa-glx libglib2.0-0

# 3) Python 虚拟环境
python3 -m venv ~/bevformer-env
source ~/bevformer-env/bin/activate

# 4) PyTorch（版本必须严格匹配，否则 mmcv 编译失败）
pip install torch==1.11.0+cu117 torchvision==0.12.0+cu117 torchaudio==0.11.0 \
  --index-url https://download.pytorch.org/whl/cu117

# 5) OpenMMLab 栈
pip install mmcv-full==1.4.8 -f \
  https://download.openmmlab.com/mmcv/dist/cu117/torch1.11.0/index.html
pip install mmdet==2.14.0
pip install mmdet3d==0.17.1

# 6) CARLA Python API 与分析依赖
pip install carla==0.9.16 numpy opencv-python pygame
pip install scipy matplotlib scikit-learn
# 国内用户建议加 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 7) BEVFormer 源码 + tiny 权重
git clone https://github.com/fundamentalvision/BEVFormer.git ~/BEVFormer
mkdir -p ~/BEVFormer/ckpts      # 放入 bevformer_tiny_epoch_24.pth

# 8) Windows 侧启动 CARLA 服务端
#    D:\CARLA\CarlaUE4.exe -quality-level=Low -windowed -ResX=1280 -ResY=720

# 9) 验证 WSL → CARLA 连通（期望输出 Map: Town10HD_Opt）
python -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(10.0); \
  print('Connected. Map:', c.get_world().get_map().name)"
```

完整 13 步（含把本仓库脚本同步到 `~/carla-adversarial/scripts/` 与 md5 一致性校验）见 [`issues_solutions_reproducibility.txt`](./issues_solutions_reproducibility.txt) §2。

### BEVFormer 源码补丁（必做）

| 文件 | 修改内容 |
|---|---|
| `projects/mmdet3d_plugin/__init__.py` | 注释 dd3d 相关 import |
| `projects/mmdet3d_plugin/datasets/__init__.py` | 注释 v2 数据集 import |
| `projects/mmdet3d_plugin/datasets/pipelines/__init__.py` | 注释 `dd3d_mapper` import |
| `projects/mmdet3d_plugin/bevformer/detectors/bevformer_fp16.py` | 移除 `tkinter` import（headless 服务端报错） |

另需两处运行时补丁：`numba.errors` 伪模块（numba ≥ 0.53 已移除该子模块）、`mmcv.utils.Registry._register_module` monkey-patch 为 `force=True`（mmdet3d 0.17.1 重复注册）。

### 分辨率降级（8GB VRAM 必做）

```bash
bash scripts/fix_resolution.sh    # tesla_camera_layout.py: 1280×960 → 640×480
```

### 运行实验

```bash
source ~/bevformer-env/bin/activate
cd ~/carla-adversarial/scripts

# 内核自检（无需 CARLA / GPU）
python attack_configs.py           # 6 预置场景校验 + 3 个负例（非法相机名 / duty=0 / 偶数 kernel）
python collapse_configs.py         # 15 相机组合法性 + 5 档矩阵规模实算

# 适配器自检 / 离线 / 在线推理
python run_bevformer_carla.py --test
python run_bevformer_carla.py --offline --img-dir ~/carla-adversarial/output
python run_bevformer_carla.py --online --host localhost --port 2000 --num-frames 50

# Phase 2 基线采集（10 路线 × 200 帧，约 2–3 小时）
python collect_baseline.py --num-routes 10 --frames-per-route 200 \
  --warmup-frames 8 --score-thr 0.05 \
  --output-dir ~/carla-adversarial/results --host localhost --port 2000

# Phase 2 攻击采集（含逐帧对比图，每 30 帧保存一张）
python collect_attack.py --attack-config sign_patch_front \
  --num-routes 10 --frames-per-route 200 \
  --warmup-frames 8 --score-thr 0.05 \
  --output-dir ~/carla-adversarial/results \
  --host localhost --port 2000 --save-images --save-interval 30

# 断点续采（从 route_5 继续，文件名自动对齐 route_5.json …）
python collect_attack.py --attack-config sign_patch_front \
  --num-routes 10 --start-route-id 5 ...

# Phase 3B 崩溃点扫描（--resume 必带，否则从头重跑）
python run_collapse_experiment.py --mode h1_quick --frames-per-route 100
python run_collapse_experiment.py --mode h1_quick --frames-per-route 100 --resume

# 分析与出图
python analyze_collapse_points.py --scan-dir ~/carla-adversarial/results/collapse_scan/h1_quick
python analysis_baseline_vs_attack.py
```

### 攻击预设（`core/attack_configs.py`，6 个，自检全部 PASS）

| attack_id | 类型 | 目标相机 | 关键参数 | 对 k=15 监控器 |
|---|---|---|---|---|
| `sign_patch_front` | patch_sticker | front_main | frac 0.25, duty 1.0, onset 60, dur 100 | 可见 |
| `road_mark_intermittent` | patch_sticker | front_main, front_wide | frac 0.15, duty 0.3, period 100 → run **30** | 可见（30 ≥ 15） |
| `fast_intermittent_blind` | patch_sticker | front_main | frac 0.25, duty 0.5, period **10** → run **5** | **结构性失明（5 < 15）** |
| `camera_degradation_blur` | blur | front_main | kernel_size 11, duty 1.0 | 可见 |
| `glare_dazzle_front` | glare | front_main | glare_intensity 0.8, duty 1.0 | 可见 |
| `dropout_dead_cam` | dropout | front_narrow（阴性对照） | duty 1.0 | 可见 |

`AttackConfig.__post_init__` 硬校验：`patch_frac ∈ [0.05, 0.5]`、`opacity ∈ [0,1]`、`kernel_size` 为 `[3,15]` 内奇数、`glare_intensity ∈ [0,1]`、`duty_cycle ∈ (0,1]`、`period_frames ≥ 4`、相机名必须落在 8 相机白名单内；`duty_cycle == 1.0` 时强制 `period_frames = duration_frames`。

---

## 📁 项目结构

```
tesla-fsd-adversarial-attack/              108 个跟踪文件 / 78.0 MB
├── README.md                              ← 本文件
├── README.txt                             早期纯文本目录说明（历史保留）
├── issues_solutions_reproducibility.txt   复现指南 + 35+ 问题与方案（27.6 KB / 554 行）
├── 部分操作指南和数据分析.pdf               中文操作指南与数据分析（3.0 MB）
├── .gitignore
├── config/    (1)   tesla_camera_layout.py
├── core/      (3)   attack_configs · attack_injector · collapse_configs
├── pipeline/  (8)   spawn → BEV 拼接 → 适配器 → 推理 → 基线/攻击采集 → 扫描 CLI
├── analysis/  (5)   阈值标定 · 统计定稿 · 对比分析 · 扫描引擎 · 崩溃点检测
├── utils/     (13)  自检与调试脚本
├── scripts/   (3)   fix_resolution.sh · gather_info.sh · optimize_scan.sh
├── results/   (48 = 29 JSON + 19 PNG)
│   ├── baseline/                  10 路线 JSON + baseline_stats.json + archive_manifest.json
│   ├── attack_sign_patch_front/   10 路线 JSON + attack_stats + images/（8 张逐帧对比图）
│   ├── analysis/                  analysis_report.json + fig1–fig5
│   ├── collapse_scan/             experiment_meta + scan_progress + analysis/（report + fig1–fig6）
│   └── smoke_test/                1 路线冒烟测试数据
└── output/    (23 PNG)
    ├── raw_cameras/         (8)   Tesla 八相机原始帧
    ├── bevformer_offline/   (6)   离线逐相机检测结果
    ├── bevformer_online/    (6)   在线逐相机检测结果
    └── grids/               (3)   合成网格图
```

代码统计：**30 个 Python 文件 + 3 个 Shell 脚本**；数据与图像：**42 PNG + 29 JSON + 2 TXT + 1 PDF**。

---

## 📚 文档

| 文档 | 内容 |
|---|---|
| [`issues_solutions_reproducibility.txt`](./issues_solutions_reproducibility.txt) | **Part A** 复现指南（系统要求 / 13 步环境搭建 / BEVFormer 补丁 / 快速启动 / 目录结构 / 关键阈值）· **Part B** 9 大类 35+ 问题与解决方案 · 11 条已验证失败方案 · 附录 10 条关键经验 |
| `部分操作指南和数据分析.pdf` | 部分操作指南与数据分析（中文，3.0 MB） |
| [`results/baseline/archive_manifest.json`](./results/baseline/archive_manifest.json) | Phase 2 归档清单：39 文件 / 26,293,346 字节 / `code_commit = 77af60b` / `archive_date = 2026-09-02` |
| `README.txt` | 早期纯文本目录说明（已由本文件取代，保留作历史记录） |

---

## 🛠️ 技术栈

| 层 | 技术 |
|---|---|
| 仿真 | CARLA **0.9.16**（Windows 服务端 · 同步模式 `dt=0.05s`（20 FPS）· 地图 `Town10HD_Opt` · TrafficManager 端口 8000，由 Python API 内置启动，无独立可执行文件） |
| 感知代理 | **BEVFormer-tiny**（ResNet-50 backbone · BEV 50×50 · `bevformer_tiny_epoch_24.pth` · nuScenes 预训练 · NMS-free head 固定输出 300 slot） |
| 深度学习栈 | PyTorch 1.11.0+cu117 · mmcv-full 1.4.8 · mmdet 2.14.0 · mmdet3d 0.17.1 · CUDA 11.7 · numba 兼容补丁 |
| 图像攻击 | OpenCV（`addWeighted` 图案混合 · `GaussianBlur` · 加性高光场 · 纯黑遮挡）· NumPy `RandomState(seed=42)` 保证逐帧确定性 |
| 数据桥接 | `carla_bev_adapter`：resize 至 800×450 → `PAD_SIZE_DIVISOR=32` 补齐 → ImageNet 归一化（mean `[123.675, 116.28, 103.53]`，std `[58.395, 57.12, 57.375]`）→ `TESLA_TO_NUSCENES` 8→6 映射 |
| 分析可视化 | matplotlib（11 张分析图）· scipy · scikit-learn；统计口径 **median ± IQR**（检测数分布严重右偏，均值不具代表性） |
| 跨平台 | WSL 2 Ubuntu 22.04 客户端 + 3 个 Bash 辅助脚本 + Windows↔WSL md5 一致性校验纪律 |

### 相机映射（`carla_bev_adapter.TESLA_TO_NUSCENES`）

| nuScenes 相机 | Tesla 相机 | FOV / 探测距离 | 安装位置 |
|---|---|---|---|
| `CAM_FRONT` | `front_main` | 50° / 150 m | 挡风玻璃中央 |
| `CAM_FRONT_RIGHT` | `front_wide` | 120° / 60 m | 挡风玻璃偏副驾侧 |
| `CAM_FRONT_LEFT` | `side_front_left` | 40° / 80 m | 左 B 柱 |
| `CAM_BACK` | `rear` | 140° / 50 m | 车牌上方 |
| `CAM_BACK_LEFT` | `side_rear_left` | 40° / 100 m | 左前翼子板 |
| `CAM_BACK_RIGHT` | `side_front_right` | 40° / 80 m | 右 B 柱 ⚠️ 命名左右不对称（已识别，未修正） |
| — | `front_narrow`（35° / 250 m）<br>`side_rear_right`（40° / 100 m） | | **不进入 BEV 链路 → 用作阴性对照** |

---

## 🔑 数据源

本项目**不依赖任何外部真实驾驶数据集**（无 nuScenes / KITTI / Waymo 原始数据），全部观测数据由仿真自产：

| 数据 | 来源 | 说明 |
|---|---|---|
| **Tesla 八相机参数** | Tesla 官方规格（`tesla.cn/autopilot`） | 在 `config/tesla_camera_layout.py` 文件头声明；含安装位姿、FOV（35°/40°/50°/120°/140°）、探测距离（50–250 m） |
| **仿真场景与路线** | CARLA 0.9.16 · 地图 `Town10HD_Opt` | 由 `world.get_map().get_spawn_points()` 经 `np.random.permutation` 采样；Phase 2 十条路线实际使用的 `spawn_index` = `123 / 135 / 43 / 147 / 120 / 71 / 108 / 145 / 59 / 22`，已逐一记录在 `results/baseline/baseline_stats.json` 的 `per_route` 中 |
| **感知模型权重** | BEVFormer 官方仓库（nuScenes 预训练） | `~/BEVFormer/ckpts/bevformer_tiny_epoch_24.pth`，因体积与许可原因**未入库**，需自行下载 |
| **攻击图案** | 程序生成（非外部素材） | `np.random.RandomState(pattern_seed=42)` 生成随机彩色块，`opacity` 加权混合，逐帧完全确定 |
| **实验观测数据** | 本仓库自产 | `results/` 下 29 个 JSON（基线 2000 帧 + 攻击 1956 帧 + 崩溃扫描元数据）+ 19 张 PNG；`output/` 下 23 张可视化 PNG |
| **环境事实与阈值** | `issues_solutions_reproducibility.txt` §6 | BEV 监控阈值、分辨率降级、同步模式 dt 等均来自该文件实测记录 |

**可复现性边界**：攻击图案（`pattern_seed=42`）与崩溃扫描条件矩阵（`collapse_configs.py` 纯枚举，无随机性）完全确定；但**路线采样未设随机种子**，重跑采集会得到不同的 `spawn_index` 组合 —— 复现时应直接引用仓库 JSON 中已记录的 spawn_index，或自行补种。

---

## ⚙️ 关键参数

| 参数 | 值 | 来源 | 说明 |
|---|---|---|---|
| BEV Cosine 阈值 | **0.999740** | Phase 1 标定 | 低于 = 异常 |
| BEV L2 阈值 | **0.027718** | Phase 1 标定 | 高于 = 异常 |
| 持续帧数 | 15 | 监控器设计 | 需连续异常帧才告警 |
| 检测分数阈值 | 0.05 | 经验值 | Domain Gap 下必须降低才有足量框 |
| 攻击起始帧 | 60 | 实验设计 | ABAB 设计的预热段 |
| 攻击窗口 | 100 帧 | 实验设计 | `duration_frames` |
| 图案种子 | 42 | 复现性 | 逐帧图案完全确定 |
| 相机分辨率 | **640×480** | VRAM 约束 | 由 1280×960 降级（`scripts/fix_resolution.sh`） |
| 同步模式 dt | 0.05 s | CARLA 设置 | 固定步长，20 FPS |
| 预热帧 | 8 | 经验值 | 采集前丢弃，等待回调就绪 |
| BEVFormer 输入 | 800×450 → pad(32) | 适配器常量 | 原始 1600×900 × 0.5 缩放 |

---

## 📄 License

本仓库当前**未包含 LICENSE 文件**。在明确添加许可证之前，默认保留所有权利（All Rights Reserved）。

---

## 📧 联系

- **GitHub**: [@TKBM649](https://github.com/TKBM649)
- **Issues**: [提交问题](https://github.com/TKBM649/tesla-fsd-adversarial-attack/issues)

---

<p align="center">
  <sub>Built with ❤️ for autonomous-driving perception security research</sub>
</p>
