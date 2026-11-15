# Isaac-PathCover

Isaac-PathCover 用原生 Python 实现 PathCover，并通过 L2F 电机及旋翼动力学接入 Isaac Lab 3.0 / Newton。输入点云经过稀疏地图、障碍膨胀、全局搜索和凸走廊，生成可跟踪的 MINCO 多项式轨迹。

官方算法固定为 [PathCover 8723a05d](https://github.com/kunalnk123690/PathCover/tree/8723a05d1993bf0199a453b47dc6ee167d493c23)。质量、惯量、电机和旋翼参数来自原 Isaac-L2F。来源提交及文件 SHA256 见 [docs/sources.json](docs/sources.json)。本仓库独立运行，不导入原 Isaac-L2F，也不包含 CORTO 规划或历史训练代码。

## 运行演示

在仓库根目录运行以下命令。它依次完成 medium 和 dense 飞行，并保存结果。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m isaac_pathcover.demo --scene both --seeds 0 --duration 60 --output results/demo
```

`--duration` 是每次试验的最大仿真时长。满足到达条件后提前结束。批量验收使用固定种子：

默认运行模式为**异步规划**：独立 CPU 进程承担地图更新和 GCOPTER 求解，Newton 在规划期间持续推进。使用 GCOPTER 原有收敛规则，并记录真实求解耗时；允许求解结果在原定交接时刻后接入，前提是与当前实测状态匹配且满足轨迹约束。验收关注实际求解耗时、有效交接频率和无碰撞到达率。旧版暂停仿真时间的同步对照可使用 `--synchronous`。实现见[异步重规划](docs/async_replanning.md)。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m isaac_pathcover.demo --scene both --seeds 0 1 2 3 4 --duration 60 --output results/validation
```

每个 `results/demo/<scene>/seed000/` 包含：

| 文件 | 内容 |
|---|---|
| `flight.html` | 可离线打开的障碍、初始走廊、初始轨迹和完整飞行路径 |
| `flight.npz` | 100 Hz 控制轨迹、400 Hz 物理状态、参考、接触力、转子状态和指令 |
| `result.json` | 到达、碰撞、跟踪误差、动力学统计和耗时 |
| `planning.json` | 每次求解的状态、代价、迭代、约束残差、交接和执行拒绝原因 |
| `mapping.json` | 每次传感器扫描、点云融合、占据变化和耗时 |
| `plans/NNNN/trajectory.npz` | 每次求解成功的多项式系数、分段时长和搜索路径 |
| `plans/NNNN/corridor.json` | 对应多面体、种子点和局部终点 |
| `trajectory.npz`、`corridor.json` | 首次接受的轨迹和走廊 |

可以使用 [`scripts/record_rerun.py`](scripts/record_rerun.py) 将已录制的 Newton 轨迹转成 Rerun `.rrd`，交互回放障碍物、无人机姿态、实际轨迹、参考轨迹和速度曲线。使用说明见 [Rerun 回放](docs/rerun_recording.md)。

场景目录中的 `summary.json` 汇总所有种子。`source_hashes.json` 固定本次运行的 Python 源码。候选求解成功但执行检查失败时，候选仍会保存；以 `handoff.accepted` 判断是否执行。

## 安装

使用 Python 3.12。规划及数值测试依赖 NumPy 2.5.1、SciPy 1.17.0 和 PyTorch 2.11.0。完整仿真还需要兼容的 NVIDIA 驱动和 CUDA GPU。

```bash
python3 scripts/install_simulation.py
```

仅安装规划及测试环境：

```bash
uv venv --python 3.12
uv pip install '.[test]'
```

只生成两图的轨迹和走廊，无须启动 Newton：

```bash
.venv/bin/python -m isaac_pathcover.demo --scene both --plan-only --output results/planning
```

运行程序在 NumPy 导入前默认设置单线程 BLAS。性能复现时显式设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。如果调用者已经设置这些变量，程序保留调用者的值。

完整安装使用固定提交的官方源码布局，只安装必需的 core、Newton、contrib 和 PhysX 配置组件。后两个组件由官方场景模块导入，实际物理后端仍只有 Newton。安装不会添加 RSL、TensorBoard 或 Transformers。详细步骤和版本见 [安装说明](docs/installation.md)。

## 实现范围

| 模块 | 实现 |
|---|---|
| `mapping/nanovox.py` | 稀疏 8³ 体素块、整数对数几率、命中/未命中融合、占据及距离查询 |
| `mapping/inflation.py` | 官方椭球膨胀和邻域外壳，增加支持删除占据体素的精确增量更新 |
| `planning/search.py` | 3D JPS、距离势场、DMP 路径细化；规划流程优先复用当前地图中安全的旧路径段 |
| `planning/risp.py` | 原 RISP 随机分割、切平面及点集更新 |
| `planning/corridor.py` | 路径与多面体求交、连续重叠走廊、局部终点及起始区域选择 |
| `trajectory/minco.py` | S3NU/S4NU 约束系统、分段多项式、能量与解析伴随梯度 |
| `trajectory/flatness.py` | 含阻力的微分平坦性映射及解析反向传播 |
| `trajectory/gcopter.py` | 原时空参数化、目标与惩罚、梯形积分、联合时间优化；新增基于上一条轨迹的热启动 |
| `trajectory/lbfgs.py` | 原 cautious L-BFGS 和 Lewis–Overton 线搜索 |
| `control/` | 官方几何控制、轨迹交接、L2F 旋翼分配及电机状态 |
| `simulation/` | Newton/MJWarp 刚体、精确障碍实体、机载激光雷达和结果统计 |

优化器没有调用 `scipy.optimize.minimize`。SciPy 用于线性代数、线性规划和几何计算。正式执行不调用自定义原生扩展、ROS、Gazebo 或 MOSEK。

默认冷启动保留官方目标权重、梯度、约束、时间参数化、初始化和停止流程。重规划增加基于上一条轨迹的热启动，但不修改目标函数与求解器。仅将物理参数匹配到 L2F 的 0.752 kg 无人机。随机分割使用持续的 NumPy 随机生成器；数值对照可注入保存的官方随机序列。

GCOPTER 的“收敛”状态不保证软约束满足。Newton 交接增加了独立执行检查，超限候选保留原求解记录并被拒绝。检查复用原惩罚宽度 `μ=0.01` 作为明确的软残差容许带。姿态、角速度和推力采用采样检查。完整规则见 [执行与验收](docs/execution.md)。

## 场景与时钟

medium 使用已有的 100 障碍场景。dense 使用已有的 180 障碍重建场景。两者的体素、起终点和原始文件哈希均保存在 [场景来源](src/isaac_pathcover/simulation/scenes/provenance.json)。dense 的来源标记为 reconstructed。

| 场景 | 占据体素 | Newton 障碍盒 | 起点 / m | 终点 / m |
|---|---:|---:|---|---|
| medium | 59,870 | 762 | `[-11.04, -2.11, 1.28]` | `[12.68, -3.47, 1.04]` |
| dense | 103,876 | 1,372 | `[-14, -6.5, 0.49]` | `[6.8, 5.2, 0.49]` |

演示从已知静态场景的表面点云初始化地图。飞行中继续融合机载扫描。物理频率为 400 Hz，控制频率为 100 Hz，扫描和重新规划频率为 2 Hz。

默认异步模式不在求解时暂停 Newton；以前的同步模式仍可通过 `--synchronous` 复现。2 Hz 是仿真时间中的请求上限，繁忙时跳过请求，不等于已实现 2 Hz 墙钟实时规划。实际规划频率、墙钟延迟和仿真推进量均保存在报告中。

## 验证与复现

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

测试直接读取固定官方输入与输出。常规测试不需要 C++ 编译器。`tests/reference/` 中的独立驱动用于重新编译官方头文件并复算数值证据；它们不进入 Python 安装包。

- [地图与全局搜索对照](docs/mapping_search_validation.md)：占据、膨胀、JPS/DMP 和配对性能。
- [RISP 与走廊对照](docs/corridor_validation.md)：平面、随机序列、连续覆盖和性能。
- [轨迹数值对照](docs/numerical_validation.md)：MINCO、平坦性、目标、梯度、非凸收敛及性能。
- [执行与验收规则](docs/execution.md)：传感器、时钟、失败处理、交接门限和指标定义。
- [Newton 闭环结果](docs/newton_validation.md)：两图各 5 个固定种子的完整数据，以及原交接方式的对照。
- [2026-11-15 工程优化](docs/engineering_optimizations.md)：原仓库增量/热启动机制、精确增量膨胀、旧路径复用和 GCOPTER 热启动的实现及种子 0 实测对照。
- [异步重规划验收](docs/async_replanning.md)：SUPER 延迟预测、轨迹交接和 Newton 连续仿真的实测对照。

数值测试保留正常与异常结果。近零比推力或倒置姿态会使原平坦性映射病态。两种语言的非凸优化也可能沿不同浮点路径停止。完整目标和梯度对照用于判断移植一致性；Newton 到达和碰撞率单独验证。

## 已完成的飞行验收与限制

2026-11-15 的历史异步验收：medium/dense 在种子 0–4 下均 **5/5 无碰撞到达**。当时使用了已删除的 1.5 s 人为截止时间，因此 23 次规划被提前终止或拒绝。完整墙钟规划周期中位数分别为 1.376/1.208 s。最新实现不再通过固定墙钟预算拒绝轨迹，改用实测状态和轨迹约束决定交接。详细指标见[异步重规划验收](docs/async_replanning.md)。

下述统计对应 2026-11-14 的五种子基线。2026-11-15 优化后各场景额外验证了种子 0，决策周期 P50 降至 medium 0.913 s、dense 1.216 s；两次均无碰撞到达，但仍未达到墙钟 2 Hz。完整对照见[工程优化报告](docs/engineering_optimizations.md)。

2026-11-14，medium 和 dense 的种子 0–4 均无碰撞到达。两图的合并跟踪均方根误差分别为 1.34 cm 和 1.98 cm，最大误差为 5.31 cm 和 8.00 cm。它们是本次 10 次试验的观测结果。

仍有以下未达标项：

- 完整周期的 p50/p95 为 medium 2.33/9.21 s、dense 2.42/4.19 s。234 次周期均超过 0.5 s，未达到墙钟 2 Hz。
- 实际闭环状态仍有规划界限超调。两图最大角速度范数为 5.55/5.22 rad/s，高于规划器的 3 rad/s。电机状态推算的 100 Hz 推力峰值为 10.25/13.64 N，高于规划上界 10 N。详情见飞行报告。
- 一个真实近奇异姿态的梯度缩放差为 `0.00124633`，没有通过正常 `3e-8` 阈值。该例在常规测试中标为严格 `xfail`，完整输入及 C++ 对照保留在仓库中。

执行检查恢复了本组试验中的无碰撞到达，但没有将软约束优化和采样检查变为严格的连续时间可行性证明。

## 许可证

主许可证为 BSD-3-Clause。移植模块保留对应来源的许可说明，见 [licenses/](licenses/)。
