# SUPER 轨迹数值实现与对照

## 来源和运行路径

轨迹模块移植自 SUPER 提交 `2ad3419c127a617c6d7df6925e81a14175a9c096`。
运行路径只有 `ExpTrajOpt → BackupTrajOpt → YawTrajOpt`。
`gcopter.py` 和旧轨迹诊断模块已删除。

| Python 文件 | SUPER 来源 | 职责 |
| --- | --- | --- |
| `trajectory/numerics.py` | `src/utils/minco.cpp`、`src/traj_opt/exp_traj_optimizer_s4.cpp`、`backup_traj_optimizer_s4.cpp` | MINCO_S4NU、目标函数、解析梯度、时间和顶点坐标映射 |
| `trajectory/lbfgs.py` | `src/utils/lbfgs.cpp` | 原始有限内存 BFGS、谨慎更新、Lewis–Overton 线搜索 |
| `trajectory/super_opt.py` | 两个 S4 优化器和 `src/traj_opt/yaw_traj_opt.cpp` | 初始化、参数、源接受条件、航向插值 |
| `trajectory/minco.py` | MINCO 和 `utils/optimization/polynomial_interpolation.h` | 通用最小导数插值、轨迹求值、精确裁剪和拼接 |
| `trajectory/numerics.py::flatness_jacobian` | `utils/geometry/quadrotor_flatness.hpp` | 唯一微分平坦映射及解析 Jacobian，供约束核直接调用 |

CPU 数值核由 Numba 编译 Python 源码。所有核均设置 `fastmath=False`。
标量除零采用 `error_model="numpy"`，与源 C++ 的 IEEE Inf/NaN 传播相同。
运行时不编译或加载自定义 C++、CUDA 扩展，也不使用 ROS。
L-BFGS、MINCO_S4NU 和约束积分没有第二个运行实现。
通用 MINCO 插值器使用 SciPy LAPACK，供航向插值和数值查询使用。
S4 优化目标内部使用源顺序的无主元带状 LU 分解和转置求解。
两者求解同一 MINCO 边界与连续性方程。
SciPy 的主元选择仅改变线性系统的数值求解方式。

## 算法契约

位置轨迹每段为 7 次多项式。边界状态是 `(3, 4)` 的位置、速度、加速度和加加速度。
系数数组形状为 `(段数, 8, 3)`，幂次升序。
轨迹对象提供任意阶导数、4 阶状态批量求值、精确裁剪和拼接。
多项式外推和分段边界的选择遵循 SUPER。
命令层负责使用源时间规则限制求值时间。

ExpTrajOpt 保留走廊简化、相邻交集内切球、最近引导点的时间戳、`max(0.01, Δt) × 0.8` 初始化和 6 位小数截断。
初始中间点是交集内切球中心。
`pos_constraint_type=1` 使用直接位置变量。
`pos_constraint_type=2` 使用原平方归一化顶点参数化及其范数惩罚。
`static_dense.yaml` 的注释与实际 `switch` 有差异；移植按代码执行类型 1。
优化器会原地更新传入的走廊列表，保留执行层对简化结果的观察。

目标包含 snap 积分、时间项、梯形积分约束，以及 Exp 的中间点吸引项。
位置、速度、加速度、加加速度、角速度和推力项使用源平滑 L1。
默认关闭的项仍支持其解析梯度。
角速度和推力项按源码要求同时启用。
时间、MINCO 消元、自由终点和备用切换时间全部使用解析链式导数。

BackupTrajOpt 保留 2 段制动轨迹、自由终点和有界切换时间。
切换时间用正弦映射到 `[t0, te]`。
其头部状态来自 Exp 轨迹，切换时间梯度包含 Exp 的速度至 snap。
均匀时间模式只有 1 个总时长变量。
目标保留 `penna_ts × (te − ts)` 项。

YawTrajOpt 保留时间分配、0.5 s 前视、航向展开和 3/5/7 次最小导数插值。
默认使用 3 次最小加速度插值。
恰好相差 `±π` 时保留原侧。
源航向超限分支只记录日志，不拒绝轨迹；Python 也不新增拒绝条件。

## 默认值和停止规则

下表直接对应 SUPER `config/static_dense.yaml`。
Newton 的质量和重力适配由上层配置传入。

| 参数 | Exp | Backup |
| --- | ---: | ---: |
| 位置参数化类型 | 1 | 2 |
| 禁用 snap 能量项 | false | true |
| 均匀时间 | false | true |
| 积分分辨率 | 15 | 12 |
| 平滑系数 | 0.01 | 0.01 |
| 相对目标停止精度 | 5e-6 | 1e-5 |
| 时间权重 | 1e7 | 100 |
| 位置权重 | 5e9 | 1e10 |
| 速度权重 | 5e8 | 1e8 |
| 加速度权重 | 5e8 | 1e8 |
| 加加速度权重 | -5e8 | -1e6 |
| 吸引项权重 | 5e8 | -1 |
| 角速度/推力权重 | -1 / -1 | -1 / -1 |

两者的速度、加速度和加加速度上限分别为 `8 m/s`、`15 m/s²` 和 `120 m/s³`。
备用切换时间权重为 `1e7`。
L-BFGS 保留 `mem_size=256`、`past=3`、`g_epsilon=0`、`min_step=1e-32` 和无限制迭代数。
线搜索最多评估 64 次，使用 Armijo 和弱 Wolfe 条件。
初始驻点使用 `≤ g_epsilon`；迭代中的驻点使用 `< g_epsilon`。
没有人为求解截止时间或放宽停止精度。

优化后的接受条件逐项保留源代码。
Exp 的速度拒绝分支在原源码中被注释，Python 没有重新启用。
Backup 额外使用原始 `1.2 × max_vel/max_acc` 连续轨迹极值检查。
连续极值通过多项式驻点求根获得。

## 热启动和源状态依赖

Exp 的 `init_times/init_points` 直接初始化时间和中间点。
该入口按源重载从边界、中间点和累计段时长重建引导路径。
普通执行入口使用带时间戳的旧轨迹引导路径。
这两种入口均保留相同目标和停止条件。

Backup 的显式 `init_times/init_points` 入口支持非均匀时间和端点初始化。
均匀时间的总时长由 `heu_duration` 显式传入。
SUPER 的对应重载没有重新写入 `total_time`，因此读取上一次调用的总时长。
重放该重载时，`heu_duration` 应传入上一次的总时长。
Python 用显式参数表示这项状态依赖。

SUPER 的主执行流程仅调用普通 `heu_duration` 入口。
显式热启动重载只出现在 `super_planner.cpp:1025` 的日志重复求解。
本项目移除了该次日志重复求解。
下一轮普通调用会重新设置总时长和边界，因此移除它不改变主执行状态。

## C++ 固定输入对照

`tests/reference/super_trajectory/regenerate.py` 从上述提交提取原目标函数和约束函数。
它直接编译原 MINCO、带状系统、L-BFGS、轨迹和航向插值源码。
只替换 ROS/颜色日志头文件。
编译输出保存在忽略目录 `.reference/super_trajectory/`。
运行时规划器不调用这个程序。

已保存 6 组固定输入：Exp 两种位置参数化各覆盖能量开启/关闭，Backup 覆盖均匀/非均匀时间。
测试同时启用角速度、推力和加加速度约束，验证默认关闭的分支。
输入包含归一化半空间、顶点顺序和初始变量，因此复测不依赖线性规划退化解的选择。
fixture 保存原源码 SHA-256、原输出、Python 误差和完整收敛记录。

固定目标误差定义为 `|f_py − f_cpp| / max(1, |f_cpp|)`。
梯度及其余数组采用逐项 `|a − b| / (1 + |b|)` 的最大值。
完整求解轨迹在共同有效时间内的 101 个等距时刻比较。

| 案例 | 固定目标误差 | 固定梯度误差 | 完整求解最大位置差 |
| --- | ---: | ---: | ---: |
| Exp 类型 1，含能量 | 5.83e-15 | 1.75e-14 | 3.27e-14 m |
| Exp 类型 1，无能量 | 1.72e-16 | 1.68e-14 | 7.56e-11 m |
| Exp 类型 2，含能量 | 2.05e-14 | 5.59e-14 | 2.77e-14 m |
| Exp 类型 2，无能量 | 9.17e-15 | 5.68e-14 | 7.43e-8 m |
| Backup 非均匀时间 | 2.41e-14 | 3.72e-11 | 2.75e-3 m |
| Backup 均匀时间 | 1.48e-13 | 1.09e-13 | 1.28e-4 m |

各案例的停止状态均与 C++ 相同。
非均匀 Backup 的最终目标相对差为 `9.21e-6`，总时长差为 `1.71e-4 s`。
其他案例的最终目标相对差小于 `1.74e-8`。
非凸优化的最终迭代点会受浮点运算影响。
浮点运算顺序会影响线搜索和历史更新。
固定目标、解析梯度、MINCO 系数及边界伴随导数另设严格对照。

`tests/test_super_trajectory.py` 验证上述固定结果、完整收敛、航向系数，以及裁剪拼接的 0–3 阶连续性。
原有 MINCO 和微分平坦映射 fixtures 继续保留。
旧 `flatness.py` 已删除。其非零阻力、航向和四元数梯度 fixture 直接验证运行数值核。

### MINCO 与完整航向阶数

6 组 SUPER 固定输入的 MINCO 系数最大尺度误差为 `3.61e-14`。
能量误差为 `2.48e-13`。
系数偏导最大误差为 `1.44e-12`。
时间、路径点及首尾状态伴随导数最大误差为 `2.05e-12`。
这些误差统一使用 `max(|a−b|/(1+|b|))`。

航向对照现已覆盖 3、5、7 次插值。
每个阶数使用 6 条相同的 C++ 位置轨迹。
三者的时间分配误差全部为 0。

| 航向阶数 | 案例数 | 系数最大尺度误差 | 101 点最大航向差 |
| --- | ---: | ---: | ---: |
| 3 | 6 | 1.21e-15 | 7.22e-16 rad |
| 5 | 6 | 2.35e-15 | 1.89e-15 rad |
| 7 | 6 | 2.02e-14 | 1.26e-14 rad |

C++ 的 5 次轨迹保存 6 项系数。
Python 接口统一保存 8 项，末尾补 0。
对照在补 0 后比较同一多项式。
汇总原始数据位于 `profile/trajectory_numeric_evidence_summary.json`。

### 显式热启动与数值精度限制

`regenerate_hotstart.py` 编译原公开显式重载。
它还调用原几何预处理和接受检查。
目标函数外增加记录包装，保存首次和最终变量。
ROS 日志及时间读取使用离线桩。
原 QuickHull 的缺失标准头通过 `-include cstdint` 补入。
原 C++ 算法文件保持不变。

新增 4 组输入，共执行 5 次显式热启动。
Exp 覆盖两种位置参数化。
Backup 覆盖非均匀时间及两次均匀时间调用。
均匀时间序列为普通初始化、显式初始化、再次显式初始化。
Python 使用 C++ 的固定顶点顺序，隔离枚举顺序差异。
连续调用由同一 C++ 对象执行。
Python 每次传入该阶段的 C++ 继承总时长，以比较同一个问题。
该对照不表示两侧独立连续调用会得到相同序列。
用户给定的 `init_times/init_points` 记录误差全部为 0。
相同初始变量的目标及梯度误差分别不超过 `1.23e-14/1.07e-13`。

均匀时间重载确实沿用前一次总时长。
两次继承值为 `0.6853168041297044 s` 和 `0.6160733952472788 s`。
给定段时长仍为 `[0.82, 1.05] s`。
两者与实际均匀初始时长不同，已单独验证。

下表的目标尺度差定义为 `|f_py−f_cpp|/(1+|f_cpp|)`。
轨迹先按共同有效局部时间的 101 点比较。
全部 5 次调用的成功状态相同。
L-BFGS 都以原相对目标停止条件返回状态 `1`。

| 显式热启动 | Python / C++ 迭代数 | 最终目标尺度差 | 最大位置差 |
| --- | ---: | ---: | ---: |
| Exp 类型 1 | 12 / 12 | 2.23e-10 | 6.12e-9 m |
| Exp 类型 2 | 114 / 136 | 1.69e-5 | 7.84e-3 m |
| Backup 非均匀 | 123 / 139 | 6.47e-4 | 5.02e-3 m |
| Backup 均匀，第 1 次 | 461 / 138 | 4.15% | 0.676678 m |
| Backup 均匀，第 2 次 | 2054 / 1335 | 4.43% | 0.030961 m |

第 1 次均匀 Backup 的切换时刻均约为 `1.4 s`。
二者差值仅 `1.43e-12 s`。
C++ / Python 总时长为 `0.6160733952 / 0.6115146330 s`。
尾端位置差为 `0.6766780175 m`。
在同一 Exp 绝对时轴拼接旧前缀后，1001 点最大差仍为 `0.6766780217 m`。
第 2 次的绝对时轴最大差为 `0.0309612496 m`。
完整输出存在这些差异，不能称为轨迹一致。
此显式重载仅用于原执行中的日志重复求解。
实际规划入口使用普通初始化重载。

对两侧最终变量共完成 10 次交叉固定评估。
目标误差不超过 `9.02e-13`，MINCO 系数误差不超过 `5.96e-13`。
C++ 从捕获初值重跑后，5 个最终变量均逐位复现公开重载结果。
但 9 个终点梯度未满足原 `3e-9` 严格匹配界限。
这些固定输入在测试中使用 `strict=True` 的预期失败标记。
它们不计入通过数，且没有扩大容差。

最差点来自第 2 次均匀热启动。
每段仅 `0.1486379033 s`，MINCO 系统条件数为 `5.44e9`。
最大梯度分量尺度差为 `2.49e-4`。
该分量的 C++ / Python 值为 `1.06235892 / 1.06287338`。
取消 JIT 后，原 Python 双精度结果逐位相同。
只将相同表达式及数组提升到 `longdouble` 后，该值为 `1.06228156`。
步长 `3e-12` 的五点有限差分给出 `1.06228177`。
此证据支持双精度舍入和消去放大的判断。
较大的终点差异及严格梯度匹配限制均保留在交付中。

原始记录和复现入口：

- `tests/fixtures/super_trajectory_hotstart.json`
- `tests/fixtures/super_trajectory_precision_diagnostic.json`
- `tests/reference/super_trajectory/regenerate_hotstart.py`
- `tests/reference/super_trajectory/precision_diagnostic.py`
- `tests/test_super_trajectory_hotstart.py`

离线数值入口默认使用 `.runtime/numba_hotstart` 缓存。
测试入口使用 `.runtime/numba_tests` 缓存。
测试及临时回调与运行缓存保持独立。
本次轨迹专项测试结果为 `62 passed, 9 xfailed`。
其中 9 项表示已知固定终点的严格梯度匹配限制。
这 9 项不计入通过数。
日志保存在 `.runtime/super_trajectory_final_tests.log`。

## 真实 medium 输入和性能

环境：Python `3.12.14`、NumPy `2.5.1`、SciPy `1.17.0`、Numba `0.68.1`、llvmlite `0.50.0`。
测量设置 `OPENBLAS_NUM_THREADS=1`。
统计来自同一真实 medium 输入的 10 次完整优化，包含初始化和源接受检查。
编译与缓存加载由工作进程启动前的 `warmup()` 单独记录。

| 模块 | 原 NumPy 实现，带 cProfile | Numba P50 | Numba P95 | Python 迭代/评估 | C++ 迭代/评估 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exp | 3.52 s | 67.42 ms | 67.75 ms | 710 / 1538 | 602 / 1378 |
| Backup | 4.48 s | 33.12 ms | 35.21 ms | 527 / 1058 | 711 / 1372 |

前后两列的计时条件不同：原数值来自分析器，Numba 数值未开启分析器。
表中不计算据此推导的加速倍数。
原分析器显示 L-BFGS Python 循环、积分临时数组和批量算子调用占主要时间。
优化只改变这些计算的 Python 执行方式。
最终源码保留原标量循环和解析式，复用积分临时数组。

真实输入的固定目标/梯度误差分别为：Exp `1.86e-14 / 1.19e-13`，Backup `1.65e-15 / 1.08e-12`。
完整求解的差异比小案例明显。
Exp/Backup 最终目标相对差为 `0.496% / 0.692%`。
最大位置差为 `4.61 cm / 4.55 cm`，最大速度差为 `0.284 / 0.221 m/s`。
总时长差为 `0.203 / 1.184 ms`。
原 C++ 同样需要数百次迭代，不能把高迭代数直接归因为 Python 初始化错误。
这些差异和两种求解结果均保存在原始记录中。

该测量不是完整规划的 P50/P95，也不代表 Newton 飞行结果。
完整规划还包含地图、搜索、RISP 和执行状态更新。
原执行源码的时序条件仍由执行模块保留。

原始记录：

- `tests/fixtures/super_trajectory_reference.json`
- `results/super_risp_development_20261115/profile/exp_input.json`
- `results/super_risp_development_20261115/profile/backup_input.json`
- `results/super_risp_development_20261115/profile/planning.pstats`
- `results/super_risp_development_20261115/profile/trajectory_numba_cpp.json`

复现命令：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/reference/super_trajectory/regenerate.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/reference/super_trajectory/regenerate_hotstart.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/reference/super_trajectory/precision_diagnostic.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_super_trajectory.py tests/test_super_trajectory_hotstart.py \
  tests/test_trajectory_reference.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/reference/super_trajectory/benchmark.py \
  results/super_risp_development_20261115/profile \
  --cpp-oracle .reference/super_trajectory/oracle
```

## 保留的实现差异和限制

RISP 的半空间由规划模块输入。轨迹模块没有 CIRI 或旧 GCOPTER 选择分支。
内切球使用 SciPy 内置 HiGHS，顶点枚举使用现有 SciPy 几何实现。
相同退化几何可能返回不同中心或顶点顺序；保存的固定输入消除了该影响。

源 Backup 未写入最大位置违约日志，Python 保留该日志值为 0。
源关闭角速度/推力时仍读取未初始化的诊断局部变量。
Python 将这两个关闭项的日志写为 0；相应权重关闭时不影响目标或接受条件。
仅供日志的重复求解和未使用的独立走廊顶点枚举已移除。

当前 S4 带状运算使用 NumPy 密集存储，运算范围仍限制在源带宽 8。
大段数的存储开销为平方增长。
实际录制的局部规划段数较小，目前没有证据表明它是主要耗时。
数值核首次编译约 50 s；已有缓存的新进程加载约 18 s。
工作进程在完成预热后才宣布就绪，飞行规划计时不包含此启动工作。
删除旧 flatness 模块后的 3 次回放保持相同成本和迭代数。
该次 Exp/Backup 中位数为 `66.69 / 32.96 ms`，缓存加载为 `20.37 s`。
记录保存在同目录的 `trajectory_unique_flatness.json`。
