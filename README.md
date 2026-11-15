# Isaac-PathCover

在现有 Isaac Lab / Newton 四旋翼上运行原生 Python SUPER。地图、搜索、轨迹优化和执行状态机来自 SUPER `2ad3419`。CIRI 的几何生成替换为 PathCover/RISP `8723a05d`。

执行层提供 Ideal 和 TurboMPC 两种模式。两种模式共享 SUPER + RISP 规划、CmdTraj、状态机与传感器。旧几何控制器的 9 次飞行保留在 [历史验收记录](docs/super_validation.md)。

本次 [10D TurboMPC 跟踪修复与对照](docs/controller_tracking_fix.md) 已完成：修复后的 NMPC 与已有 Ideal 基线在 Medium seed2、Dense seed4 均无碰撞到达。NMPC 最大跟踪误差为 0.290 m、0.327 m。修复涉及旋翼分配、角速度反馈和输入成本。规划参数、时间判断和状态机保持原样。[初版对照及失败记录](docs/controller_comparison.md) 完整保留。

## 运行

在仓库根目录运行。规划在独立 CPU 进程执行，Newton 持续推进。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m isaac_pathcover.demo \
  --controller nmpc --scene medium --seeds 2 --duration 90 --output results/turbompc/medium
```

`--controller ideal` 直接执行 SUPER 参考状态。`--controller nmpc` 使用 TurboMPC 控制 Newton 刚体，默认选择 NMPC。`--duration` 是仿真观察时长。到达终点后结束。仅检查规划器：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m isaac_pathcover.demo \
  --scene both --plan-only --output results/super_risp_planning
```

地图更新使用现有场景表面点云与机载激光雷达。medium 保留 100 个障碍物场景，dense 保留 180 个障碍物重建场景。场景文件、实体碰撞几何和起终点不变。

| 场景 | 起点 / m | 终点 / m |
|---|---|---|
| medium | `[-11.04, -2.11, 1.28]` | `[12.68, -3.47, 1.04]` |
| dense | `[-14, -6.5, 0.49]` | `[6.8, 5.2, 0.49]` |

## 单一流水线

| 模块 | 实现与源语义 |
|---|---|
| `mapping/rog_map.py` | ROG-Map 对数几率、增量占据计数、膨胀、虚拟高度、射线和地图查询 |
| `planning/search.py` | SUPER A*、占据起点逃逸、距离启发式与路径长度视野 |
| `planning/corridor.py` | SUPER 路径分段、重叠、简化与局部目标；RISP 生成点种子和线段种子多面体 |
| `planning/risp.py` | PathCover 随机分割；线段种子和机器人半径适配 CIRI 接口 |
| `trajectory/minco.py` | MINCO S4NU、解析伴随梯度、分段多项式截取与拼接 |
| `trajectory/super_opt.py` | ExpTrajOpt、BackupTrajOpt、YawTrajOpt、时间和位置参数化、原惩罚与收敛规则 |
| `trajectory/lbfgs.py` | cautious L-BFGS 与 Lewis–Overton 线搜索 |
| `planning/pipeline.py` | PlanFromRest、ReplanOnce、带时间戳的轨迹引导、旧路径复用、主轨迹与备用轨迹生成 |
| `control/execution.py` | ExpTraj、BackupTraj、CmdTraj、旧前缀拼接、备用轨迹切换、SUPER 状态机 |
| `control/async_planner.py` | 独立规划进程、共享仿真时钟和最新测量状态 |
| `control/ideal.py` | SUPER perfect_drone_sim 的位置、速度与平坦性姿态赋值 |
| `control/nmpc.py` | 未来参考采样、10 维状态跟踪、滚动求解并输出推力加速度和机体系角速度 |
| `control/solvers/` | 从 Isaac-L2F 复制的 Python SQP、ADMM 与 nvmath/cuDSS 求解组件 |
| `control/rate.py` | 角速度内环，使用角速度反馈和实测角加速度阻尼生成力矩 |
| `control/actuator.py` | 保持力矩方向的旋翼分配，以及原电机一阶响应 |
| `simulation/` | 保留的 Newton 刚体、场景与传感器 |

规划使用 Python、NumPy、SciPy 和 Numba。数值核由 Numba 在 CPU 编译并缓存，保留一套 Python 算法源码。运行时不调用 ROS、C++ 对照程序或自定义 C++/CUDA 扩展。现有 Newton 仿真使用 Isaac Lab、PyTorch、Warp 和 Newton 官方依赖。旧 NanoVox、JPS/DMP、GCOPTER 规划器与同步演示入口已经移除。

## 原始时间与执行语义

默认规划参数来自 SUPER 的 `super_planner/config/static_dense.yaml`。物理频率为 400 Hz，环境步进为 100 Hz。NMPC 每 20 ms 仿真时间求解一次。ROS 2 的 15 Hz 重规划参数在源码中转换为 66 ms 墙钟定时器。正在执行的规划请求不会排成积压队列。

规划进程读取持续推进的 Newton 仿真时钟。新命令保持 SUPER 自身的起始时间，收到命令时不重置轨迹时间。ReplanOnce 拼接旧轨迹前缀，BackupTraj 进入同一个 CmdTraj。

SUPER 源码保留以下时间判断：

- `generateExpTraj`：生成期间仿真时间增加超过 `replan_forward_dt` 时返回失败。
- `ReplanOnce`：墙钟耗时超过 `0.9 × replan_forward_dt` 时返回失败。
- A*：使用源搜索超时和原时钟语义。

默认 `replan_forward_dt=0.1 s`。这些是源码行为。性能报告记录其实际触发次数，不把 15 Hz、0.1 s 或实时因子作为额外验收门槛。

默认 `goal_vel_en=false`。局部轨迹的末端速度为零。连续飞行依靠及时重规划与前缀接续；备用轨迹负责原算法中的制动。

## 飞行数据

每个 `<output>/<scene>/seed000/` 保存：

| 文件 | 内容 |
|---|---|
| `flight.npz` | 实际状态、400 Hz 物理状态、PVAJ 参考、电机、旋翼推力、接触力、备用标志、状态机和 NMPC 求解耗时 |
| `speed.csv` | 仿真时间、实际速度、参考速度、备用轨迹标记与状态机 |
| `planning.json` | 每次请求、求解时刻、模块耗时、优化状态、原时间判断、轨迹提交与参考跳变量 |
| `mapping.json` | 扫描与地图融合记录 |
| `fsm.json` | 原状态机切换的时间、前后状态和回调来源 |
| `timeline.json` | 中途低速区间、并发规划记录、备用轨迹区间、模块 P50/P95 |
| `plans/NNNN/` | 主轨迹、偏航、备用轨迹、实际命令、搜索路径与走廊 |
| `result.json` | 控制器配置、到达、碰撞、跟踪误差、规划失败数、NMPC P50/P95、软件版本和硬件 |
| `source_hashes.json` | 该次运行使用的 Python 源文件 SHA256 |
| `flight.html` | 原始障碍、初始走廊和实际飞行路径 |

低速区间是诊断统计，不参与规划接受。原始速度曲线和规划时间线用于区分启动等待、参考制动、规划失败与跟踪偏差。

## Rerun

```bash
/home/tong/.local/share/pipx/venvs/rerun-sdk/bin/python scripts/record_rerun.py \
  --scene both --source results/turbompc_tracking_20261115_222728/nmpc \
  --output results/turbompc_tracking_20261115_222728/rerun/nmpc
/home/tong/.local/bin/rerun results/turbompc_tracking_20261115_222728/rerun/*/*.rrd
```

回放包含无人机姿态、主轨迹、备用轨迹、实际与参考速度、规划耗时、NMPC 求解耗时、状态机和接触力。录制名称包含控制器，旧记录标为历史几何控制器。

[打开本次 4 条回放](http://100.92.244.71:9118?url=rerun%2Bhttp%3A%2F%2F100%2E92%2E244%2E71%3A9898%2Fproxy) · [下载录制与统计](http://100.92.244.71:9119/)。其中 Ideal 使用原基线录制。生成方法与历史入口见 [Rerun 说明](docs/rerun_recording.md)。

## 验证与来源

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/check_controllers.py \
  --controller ideal --output results/controller_checks/ideal
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/check_controllers.py \
  --controller nmpc --output results/controller_checks/nmpc
```

Ideal 检查直接执行与几何碰撞。NMPC 检查受扰动悬停与旋翼推力限制。闭环对照使用 Medium seed2 和 Dense seed4，两种控制器各飞行一次。固定规划数值夹具仍保存在 `tests/fixtures/`，本次不扩展离线 C++ 热启动对照。

SUPER 参数中与现有场景范围、Newton 执行接口有关的适配在模块文档中说明。来源提交和文件哈希见 [docs/sources.json](docs/sources.json)。历史实验位于 [docs/history](docs/history)，原始实验数据保留在 `results/`。

安装沿用现有 [安装说明](docs/installation.md)。使用 `.venv/bin/python`。

## 许可

SUPER 移植代码按 LGPL-3.0-or-later 分发。MINCO/L-BFGS 和 TurboMPC 相关代码保留其 MIT 许可。PathCover 和 Newton 接口保留原来源许可。完整条款及声明位于 [licenses](licenses)。
