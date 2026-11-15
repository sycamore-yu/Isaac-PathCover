# SUPER 规划与执行移植

源码固定为 SUPER `2ad3419c127a617c6d7df6925e81a14175a9c096`。
主入口为 `planning/pipeline.py::Planner`，执行状态位于 `control/execution.py`。

## 源码对应

| Python 模块 | SUPER 源码 | 保留行为 |
|---|---|---|
| `Planner.plan_from_rest` | `super_planner.cpp::PlanFromRest` | 起点移至非占据单元；静止状态初始化；Exp、Backup 与 Cmd 组合 |
| `Planner.replan_once` | `super_planner.cpp::ReplanOnce` | 保留旧命令；前向预测；新轨迹成功后提交 |
| `Planner._generate_exp` | `super_planner.cpp::generateExpTraj` | 轨迹引导、时间戳、路径复用、A* 扩展、走廊、S4 优化、Yaw 优化及旧前缀拼接 |
| `Planner._generate_backup` | `super_planner.cpp::generateBackupTrajectory` | 视线检查、种子回退、单走廊、制动优化和切换时间 |
| `ExpTraj`、`BackupTraj`、`CmdTraj` | `data_structure/*_traj.h` | 原始起始时刻、轨迹拼接、备用区间及严格边界判断 |
| `ExecutionFSM` | `fsm.cpp`、`ros_interface/ros2/fsm_ros2.hpp` | INIT、WAIT_GOAL、GENERATE_TRAJ、FOLLOW_TRAJ、EMER_STOP 与回调顺序 |
| `simple_pm_time_allocator` | `geometry_utils.h::simplePMTimeAllocator` | 原分支与二次方程；未修改公式 |

重规划从当前命令的未来状态开始。新命令先拼入旧命令从请求时刻至预测时刻的前缀。
`CmdTraj.start_time` 仍为请求时刻。Newton 收到命令后按当前仿真时刻采样，不重置轨迹时钟。

`static_dense.yaml` 原默认 `goal_vel_en=false`。因此局部轨迹末端使用零速度、零加速度和零 jerk。
连续飞行依靠末端之前完成重规划和轨迹接续。
备用轨迹已拼入提交命令，控制端到达备用起始时刻后直接执行该段。

## 原始时间规则

以下规则来自原 C++，并非新增验收条件。

- `generateExpTraj` 在仿真时间耗时超过 `replan_forward_dt` 时返回失败。源码行 757–760。
- `ReplanOnce` 在墙钟耗时超过 `0.9 * replan_forward_dt` 时返回失败。源码行 233–237。
- `static_dense.yaml` 中 `replan_forward_dt=0.1 s`。
- ROS 2 重规划定时器周期为 `int(1000 / 15)=66 ms` 墙钟时间。
- 原主状态机和命令定时器周期为 `10 ms`。Newton 命令适配使用现有 `100 Hz` 控制循环。
- 主进程独立推进 Newton，规划进程读取共享仿真时钟。规划繁忙时不累积互斥回调。

没有加入旧 GCOPTER 执行检查、额外残差阈值、晚到候选匹配规则或规划竞争。

## 固定输入对照

运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_super_execution.py
```

实测结果为 **13 个测试通过**。

| 对照项 | 输入数量 | 结果 |
|---|---:|---|
| CmdTraj 前缀、备用标记、结束标记、PVAJ 与 yaw | 9 个时刻 | PVAJ 最大绝对误差 `7.11e-15`；yaw 误差 `0` |
| 非零起始时刻截取与跨分段拼接 | 5 个时刻 | PVAJ 最大绝对误差 `2.27e-13` |
| 原路径时间分配公式 | 36 组输入 | 时间和速度最大绝对误差 `0` |
| 原 FSM 顺序 | 22 个步骤 | 状态、new_goal、finish_plan、首次重规划跳过标记及调用次数全部一致 |
| 规划期间穿越目标阈值 | 2 个交错输入 | PubCmd 保留 Main 缓存位置；下一次 Main 才刷新；状态与原 C++ 一致 |
| 轨迹采样时刻 | 3 组起点与原 `sample_traj_dt` | 逐次加法结果完全一致 |

额外测试覆盖：提交时保留原时钟；`NO_NEED` 分支保留已提交备用命令；原重规划超时分支拒绝提交。
状态对照还覆盖重规划计算期间旧命令结束的回调穿插。返回结果按发起回调处理。
原 `fsm.cpp` 行 95 刷新 FSM 的位置快照。
`fsm_ros2.hpp` 行 383 的 PubCmd 终点判断使用该快照。
规划阻塞 Main 时，新测量位置不会提前改变 PubCmd 的状态切换。
备用入口测试确认 Exp 计算结束后读取最新测量位置。
备用标志只来自命令发布回调或成功提交。测量更新不会按时间重新计算该标志。
接近目标的早退分支只修改返回的 Exp 副本，成功提交前保留旧 Exp 标志。

固定输出保存在 `tests/fixtures/super_execution.json`。
环境和误差保存在 `tests/fixtures/super_execution_validation.json`。

离线重新生成：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/reference/super_execution/regenerate.py
```

离线程序直接包含原 `ExpTraj`、`BackupTraj`、`CmdTraj` 头文件，并编译原 `trajectory.cpp`、`piece.cpp`。
FSM 对照从原文件提取原方法体。仅替换 ROS 发布、日志和规划器返回值。
方法体 SHA256 保存在固定输出中。常规运行与测试不编译 C++。

## Python 实现适配

位置、速度、地图类型采用批量 NumPy 查询。
空间采样仍按源码顺序逐个选择，保留原跳过距离和停止条件。
采样时刻采用 `cumsum` 保留原 `t += dt` 的舍入顺序。
对 `results/super_risp_development_20261115/profile/backup_input.json` 的固定输入复算后，
备用优化器的 `t0`、`te`、初始切换时刻、终点、时长和走廊平面差值均为 `0`。
首次批量化检查耗时为 `0.249 s`，原始数据在同目录 `batched_backup_frontend.json`。

后续实测确定视线查询仍为主要前端开销。
`mapping/raycast.py` 用 Numba CPU 编译原 DDA 和邻域循环，禁止浮点重排。
备用入口按原顺序查询采样线段，在首个不可见点停止。
同一次批量查询缓存重复单元的邻域结果。
初始化阶段完成编译预热，耗时单独记为 `raycast_initialization_s`。

固定原输入复测 5 次，备用前端 P50 为 `0.007947 s`，P95 为 `0.008097 s`。
时间与终点误差不超过 `4.44e-15`，走廊平面误差不超过 `2.67e-14`。
这组误差来自恢复原逐次时间加法的舍入顺序。
原始数据与采样分析位于同目录 `backup_raycast_batch.json` 和 `.pstats`。
复现脚本为 `tests/reference/super_execution/benchmark_backup.py`。
`tests/test_super_search.py` 还对照 44 个原 C++ 视线输入。

Newton 通过 `state_reader` 提供当前 PVAJ、姿态和 yaw。
规划器在原源码读取测量值的位置刷新状态。
备用生成入口保存新的位置快照，不能复用 Exp 优化前的测量位置。
`backup_state_reader` 单独读取最后发布命令的备用标志。
GENERATE_TRAJ 期间停止命令发布，因此保留该标志。

原 `generateBackupTrajectory` 行 1025–1034 为日志再次调用相同优化器，第二次输出未用于执行。
Python 删除这次重复求解。每次首次求解都会重新初始化优化器，因此不改变下一周期的初值。

`PlannerConfig` 删除原源码只加载但不使用的 `backup_traj_en`、`yaw_mode`、`mpc_horizon`。
同时删除 CorridorGenerator 未使用的 `obs_skip_num`，以及 CIRI 专用的 `iris_iter_num`。
备用轨迹始终沿用该提交中的执行分支。

原备用前端可能在空采样数组上调用 `back()`。
Python 将该具体情况记录为 `BACKUP_EMPTY_VISIBLE_INTERVAL`，并返回原失败通道。
视场裁切和感知距离裁切沿用源逻辑；默认分别为 `false` 和 `-1`。

## LiDAR 计算耗时

Newton 原 LiDAR 的体素遍历由 `simulation/lidar.py` 用 Numba CPU 编译。
光束生成、姿态旋转、射线边界初始化和端点计算仍使用原 NumPy/SciPy 表达式。
遍历保留原 `argmin` 的轴平局顺序、逐次加法和距离边界。
该函数独立于 ROG-Map 的射线更新，未改变传感器或地图算法。

使用 `newton_timed` 保存的 medium 17 帧和 dense 136 帧，每帧交替重复 3 次。
与 git `ded657cf1856393def3fce237fe3c4f94c4975f8` 原函数相比，
全部传感器原点、端点数组、返回顺序和字节均相同。

| 场景 | 原 NumPy P50 / P95 | 编译后 P50 / P95 | P50 加速 |
|---|---:|---:|---:|
| medium | 77.36 / 104.82 ms | 4.99 / 7.15 ms | 15.50 倍 |
| dense | 90.97 / 94.77 ms | 5.54 / 6.14 ms | 16.41 倍 |

耗时包含旋转和端点计算。首次编译调用单独耗时 1.169 s。
规划进程在发出 ready 前预热 LiDAR，并单列 `lidar_initialization_s`。
数据保存在 `results/super_risp_development_20261115/lidar_jit/benchmark.json`。
离线复现脚本为 `tests/reference/lidar/benchmark.py`。
固定测试保留两图各 6 个实际传感器帧，并核验轴平局和最大距离边界。
`tests/test_scene.py` 与 `tests/test_lidar.py` 共 5 个测试通过。
