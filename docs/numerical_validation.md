# 轨迹算法数值验证

固定输入对照支持 MINCO、平坦性映射和 GCOPTER 目标及梯度的数学一致性。非凸优化的最终解仍会随浮点运算路径变化。成功状态也不保证软约束严格满足。

本文记录 2026-11-14 的轨迹模块验证。完整机器可读汇总见 [trajectory_validation.json](../tests/fixtures/trajectory_validation.json)。飞行成功率和碰撞率属于独立的 Newton 实验，不由本文的函数对照推断。

## 1. 来源与实验条件

参考 [官方 PathCover](https://github.com/kunalnk123690/PathCover/tree/8723a05d1993bf0199a453b47dc6ee167d493c23)，固定提交：

```text
8723a05d1993bf0199a453b47dc6ee167d493c23
```

| 模块 | 官方源码，均相对 src/trajectory_server/include |
|---|---|
| GCOPTER 3D | gcopter_solver.hpp 的 PATHCOVER_DIM == 3 分支 |
| MINCO S3NU、S4NU | gcopter/minco.hpp |
| 平坦性映射 | gcopter/flatness.hpp |
| L-BFGS | gcopter/lbfgs.hpp |
| 顶点重建及过滤 | gcopter/geo_utils.hpp |

汇总文件保存上述源码和 Python 运行文件的完整 SHA256。每个对照数据文件还保存独立 C++ 程序的源码 SHA256。

实验使用 Python 3.12.14、NumPy 2.5.1、SciPy 1.17.0。数值计算采用 float64，并设置 `OPENBLAS_NUM_THREADS=1`。C++ 使用系统 Eigen 3 和 GCC 13.3.0，编译参数为：

```text
-std=c++17 -O3 -DPATHCOVER_DIM=3
```

C++ 对照程序直接包含官方头文件。[离线驱动](../tests/reference/trajectory/README.md)通过标准输入输出交换数值，编译产物默认写入 `.reference/`。正式 Python 运行和回归测试不依赖这些可执行文件，也不使用 C++ 绑定。

Python 使用 SciPy 的 LAPACK 带状 LU 分解。它求解原 MINCO 约束矩阵及转置伴随系统。LAPACK 使用选主元分解，官方使用不选主元的带状消元。两者的浮点运算次序不同。

基础 GCOPTER 案例使用官方物理参数：

```text
mass = 0.716 kg
gravity = 9.81 m/s²
horizontal_drag = vertical_drag = parasitic_drag = 0
speed_regularization = 0.001
```

medium、dense 及飞行故障回归只将质量和重力改为实际动力学参数：

```text
physical_parameters = (0.752, 9.8066, 0, 0, 0, 0.001)
```

其余配置保持一致：时间权重 100，位置惩罚权重 10000，速度权重 100，角速度、倾角及推力权重各 10000。速度上限为 3 m/s，角速度上限为 3 rad/s，倾角上限为 0.7854 rad，推力范围为 2–10 N。每段使用 16 个梯形积分区间。初始化速度仍为速度上限的 3 倍。

## 2. 如何解释误差

同一组变量的比较使用以下定义：

```text
目标绝对误差 = |f_python - f_cpp|
目标相对误差 = |f_python - f_cpp| / max(1, |f_cpp|)
梯度绝对误差 = max_i |g_python[i] - g_cpp[i]|
梯度缩放误差 = max_i |g_python[i] - g_cpp[i]| / max(1, |g_cpp[i]|)
```

缩放误差避免大梯度量级掩盖比较结果。它也避免对接近零的分量直接除零。

Qhull 和官方 QuickHull 的顶点枚举顺序可能不同。GCOPTER 使用顶点坐标定义优化变量，因此函数对照向双方提供完全相同的顶点矩阵和变量向量。对照程序先独立构造官方几何，再替换顶点坐标以进行函数比较。

因此，记录中的“初始变量按位一致”表示双方实际评估了相同数值。它不表示两个几何库独立生成的顶点顺序或初始变量按位一致。两个实测故障案例另行检查官方独立生成的顶点数量。

最终解的比较则让 Python 和 C++ 从相同初值各自运行 L-BFGS。它考察优化行为。仅凭最终代价不同，无法判断目标或梯度存在漏项。

## 3. MINCO 与平坦性映射

### MINCO

固定案例含 3 段轨迹，时长为 `[1.1, 1.4, 1.2]` s。内部位置为 `[1, 0.5, 0.3]` 和 `[2, 1.5, 0.7]`。起终状态和全部参考输出见 [trajectory_reference.json](../tests/fixtures/trajectory_reference.json)。

| 检查项 | S3NU 最大绝对误差 | S4NU 最大绝对误差 |
|---|---:|---:|
| 多项式系数 | 1.066e-14 | 1.270e-13 |
| 能量积分 | 1.194e-12 | 8.640e-12 |
| 对系数的偏梯度 | 1.364e-12 | 5.239e-10 |
| 对时长的偏梯度 | 1.876e-12 | 4.422e-11 |
| 内部位置总梯度 | 1.535e-12 | 1.746e-10 |
| 时长总梯度 | 3.126e-12 | 1.053e-9 |

[基础回归测试](../tests/test_trajectory_reference.py)还检查边界条件、段间连续性及伴随梯度的中心有限差分。S3NU 检查 0–4 阶导数连续；S4NU 检查 0–6 阶导数连续。S3、S4 的完整 GCOPTER 梯度也接受有限差分检查。

### 平坦性映射

对照使用非零阻力参数：

```text
(mass, gravity, horizontal_drag, vertical_drag, parasitic_drag, speed_regularization)
= (0.752, 9.8066, 0.1, 0.15, 0.08, 0.001)
```

推力和姿态四元数与参考值的误差为 0。角速度最大误差为 `2.78e-17`。反向传播各项梯度的最大误差为 `1.11e-16`。测试同时检查批量计算与逐个状态计算的一致性。

完整输入、输出和误差见 [基础参考数据](../tests/fixtures/trajectory_reference.json)及 [汇总中的 reference_errors](../tests/fixtures/trajectory_validation.json)。

## 4. L-BFGS 的流程与失败契约

Python 实现保留官方 cautious L-BFGS 和 Lewis–Overton 线搜索。线搜索使用 Armijo 条件及弱 Wolfe 条件，没有调用 SciPy minimize。

GCOPTER 的参数如下：

| 参数 | 数值 |
|---|---:|
| 历史向量数 mem_size | 256 |
| 相对目标回看距离 past | 3 |
| 相对目标阈值 delta | 1e-4 |
| 梯度停止阈值 g_epsilon | 0 |
| 最小步长 | 1e-32 |
| 最大迭代数 | 1000 |
| Armijo 系数 | 1e-4 |
| 弱 Wolfe 系数 | 0.9 |
| cautious_factor | 1e-6 |

相对目标停止条件为：

```text
|f[k - 3] - f[k]| / max(1, |f[k]|) < 1e-4
```

基础双盒走廊中，双方均执行 26 次迭代。Python 最终目标为 `346.1426448664393`，C++ 为 `346.14264495058273`，差值约 `8.41e-8`。

达到迭代上限时，Python 返回官方错误码 `-1008`。GCOPTER 丢弃候选轨迹并报告失败。相应回归测试使用最大迭代数 1，验证不会将该候选轨迹标记为可执行。

证据见 [基础测试](../tests/test_trajectory_reference.py)、[基础参考数据](../tests/fixtures/trajectory_reference.json)和 [性能记录](../tests/fixtures/trajectory_performance.json)。

## 5. medium 与 dense 的同输入对照

两组数据固定随机种子 0。medium 按官方起始多面体选择规则裁掉第 1 个区域，保留 6 段。dense 从第 1 个区域开始，保留 7 段。裁剪前的旧 medium 缓存不属于本节正式输入。

| 场景 | 起始区域索引，0 起算 | 段数 | 优化变量数 |
|---|---:|---:|---:|
| medium | 1 | 6 | 241 |
| dense | 0 | 7 | 225 |

### 初值处的函数一致性

| 场景 | 目标绝对误差 | 目标相对误差 | 梯度绝对误差 | 梯度缩放误差 |
|---|---:|---:|---:|---:|
| medium | 1.669e-5 | 5.405e-14 | 4.825e-4 | 9.131e-11 |
| dense | 6.080e-6 | 5.707e-15 | 1.502e-5 | 1.818e-11 |

初始目标很大，因此绝对误差和相对误差应一起阅读。两组梯度的相对 L2 误差分别为 `6.86e-13` 和 `1.88e-14`。

### 各自优化后的结果

| 场景 | Python 最终目标 | C++ 最终目标 | Python / C++ 迭代数 | 最终目标相对差 |
|---|---:|---:|---:|---:|
| medium | 536.237488901 | 534.866175543 | 111 / 132 | 0.2564% |
| dense | 470.781746909 | 470.808627378 | 150 / 148 | 0.005709% |

双方均按相对目标变化停止。没有为减小差异而修改终止条件。

进一步将 Python 最终变量交给 C++，再将 C++ 最终变量交给 Python：

| 场景及变量来源 | 同变量目标绝对误差 | 梯度绝对误差 | 梯度缩放误差 |
|---|---:|---:|---:|
| medium，Python 最终变量 | 7.958e-13 | 4.373e-9 | 1.062e-10 |
| medium，C++ 最终变量 | 9.095e-13 | 1.767e-9 | 3.972e-11 |
| dense，Python 最终变量 | 0 | 1.433e-10 | 1.166e-10 |
| dense，C++ 最终变量 | 5.684e-14 | 2.603e-11 | 1.288e-11 |

这些结果支持目标和解析梯度一致。非凸优化的迭代路径仍受浮点运算次序影响。

证据：[场景汇总](../tests/fixtures/trajectory_scene_validation.json)、[medium 完整输入](../tests/fixtures/trajectory_medium_reference.json)、[dense 完整输入](../tests/fixtures/trajectory_dense_reference.json)、[场景回归测试](../tests/test_trajectory_scenes.py)。

## 6. 官方顶点量化规则的修复

早期实现使用了走廊可视化 `con2vert` 的 `1e-8` 距离去重。GCOPTER 实际调用 `geo_utils.hpp::enumerateVs/filterVs`，两者规则不同。

修复后的实现按官方顺序执行：

1. 根据内部点构造对偶点。
2. 用对偶三角形的叉积还原原始顶点。
3. 在加回内部点之前量化每个坐标。
4. 每个量化单元保留第一个顶点。
5. 最后加回内部点。

量化分辨率为：

```text
mag = max(abs(raw_vertices))
res = mag * max(1e-6 / mag, float64_epsilon)
```

四舍五入使用远离零的规则，与 Eigen 一致。源规则不使用 NumPy 默认的偶数舍入。

实测 medium 的 7.5 s 计划和 dense 的 6.0 s 计划被固定为回归输入。初始位置、速度及加速度来自保存的 Newton 状态；加速度由相邻 0.01 s 速度差得到。

修复后，这两组输入的全部原多面体和相邻交集顶点数均与官方独立构造结果一致。代表性变化包括 medium 的一个原多面体从 48 点降至 46 点，以及 dense 的一个交集从 36 点降至 34 点。

**收敛改善不能全部归因于减少优化变量。** medium 改变顶点数量的区域不是交集；每区域 1 段时，该区域顶点不直接参数化内部航点。只读快照比较显示，medium 旧新变量数均为 209，初始变量最大差为 `3.74e-14`。其改善体现了浮点迭代路径的敏感性。dense 的变量数则确实从 217 降至 215。该补充核查使用 `.reference/old_smoke_*_trajectory.json` 与已保存的新对照数据。

同时移除了非官方的 `acos` 输入裁剪。裁剪可能在近奇异姿态下将非有限目标改成有限目标。当前实现保留源函数的定义域行为。

### 固定故障输入的修复结果

| 输入 | 修复前 Python 目标 | 修复后 Python 目标 | 修复前 / 后总时长 |
|---|---:|---:|---:|
| medium，7.5 s | 4,559,628.261 | 466.216 | 1.419 / 3.973 s |
| dense，6.0 s | 1,085,744.564 | 1,249.154 | 4.280 / 4.047 s |

修复后 Python 轨迹的诊断结果：

| 输入 | 最大速度 | 最大角速度 | 最大倾角 | 推力范围 | 最小走廊余量 |
|---|---:|---:|---:|---:|---:|
| medium | 3.0229 m/s | 1.0511 rad/s | 0.2573 rad | 6.255–9.096 N | 0.1484 m |
| dense | 2.2314 m/s | 3.0052 rad/s | 0.5559 rad | 5.619–9.452 N | 0.0311 m |

这里仍有小幅软约束超限：medium 速度超过 3 m/s；dense 角速度超过 3 rad/s。表格保留实测值，没有将成功状态解释为严格满足约束。

同一组修复后初始变量下，C++ 的 medium 最终目标为 `1,153,276.699`，dense 为 `1,199.343`。Python 分别为 `466.216` 和 `1,249.154`。前者表明官方流程同样可能停在高代价区域。

证据：[完整故障回归数据](../tests/fixtures/trajectory_replan_reference.json)保存输入、多面体、顶点坐标、官方顶点数、旧新结果和双方最终变量。[回归测试](../tests/test_trajectory_scenes.py)检查顶点数及修复后轨迹的物理量。

## 7. 奇异姿态与数值限制

原平坦性映射先归一化比推力方向，再使用 `1 + z_z` 计算姿态和角速度。接近零比推力或接近倒置方向时，该映射病态。

旧 medium 坏解的一个积分节点满足：

```text
a + g e3 = (-0.00244877, -0.00041181, -11.37436678) m/s²
z_z = -0.9999999761700155
1 + z_z = 2.3829984541201554e-8
```

将相同 float64 比推力分量转换为 80 位 Decimal 后，得到：

```text
1 + z_z = 2.3829984501821431018608054796595...e-8
```

两者相对差为 `1.65e-9`。此检查仅验证该分母的差分相消；它不是整个优化器的高精度对照。

修复后 C++ 的 medium 高代价终点也接近倒置方向。在相同最终变量处，目标相对差为 `5.05e-11`，梯度绝对差为 `2.16e3`，梯度缩放差为 `2.53e-7`。该点不能宣称达到正常案例中的梯度精度。

旧 dense 坏解的比推力分量约为 `(-0.0002172, 0.0022116, -0.0035342)` m/s²，接近零向量。此处应关注归一化的条件数，不能仅观察倒置分母。

上述输入保留在 [奇异性诊断](../tests/fixtures/trajectory_singularity_diagnostics.json)及 [故障回归数据](../tests/fixtures/trajectory_replan_reference.json)。本次没有改写平坦性公式、添加备用求解器、加大惩罚或改变停止条件。

轨迹诊断采用以下方法：

- 速度、加速度和每段走廊平面残差：在归一化时间内求多项式驻点，再比较端点与驻点。
- 角速度、倾角和推力：每段采样 101 个点。

后者是采样统计，不是连续时间极值证明。对有限测试输入通过验证，也不构成任意输入下的收敛保证。

### 本次 medium 执行失败的独立复现

`validation_final_20261114/medium/seed000` 在使用机载激光雷达的本次验收中发生碰撞。首个显著超限轨迹出现在 3.5 s，计划索引为 7。

以实测位置、速度、相邻 0.01 s 速度差及保存的终点恢复输入后，Python 完全复现了保存的多项式系数。双方 11 个原多面体或交集的顶点数一致。初值处的目标相对差为 `9.52e-15`，梯度缩放差为 `4.86e-11`。

| 指标 | Python | 官方 C++，相同初值 |
|---|---:|---:|
| 最终目标 | 269388.700 | 173997.974 |
| 迭代数 | 114 | 128 |
| 最大速度 | 5.670 m/s | 4.973 m/s |
| 最大角速度，采样 | 13.567 rad/s | 11.294 rad/s |
| 最大倾角，采样 | 2.866 rad | 2.780 rad |
| 最小推力，采样 | 0.215 N | 0.078 N |
| 最大推力，采样 | 13.324 N | 10.767 N |

两者均按相对代价变化停止，却违反原有软限制。这是官方算法同样产生不可执行轨迹的实例。

该 Python 终点的同变量目标相对差为 `2.82e-10`，但梯度缩放差达到 **0.00124633**，相对 L2 差为 `0.00124490`。这一数值退化远大于正常输入，不能将其记为普通精度下的函数一致性通过。

另用保存轨迹首尾导数恢复边界。起点误差为 0；终点最大误差仅 `7.11e-15`。这使双方最终代价变为 `270077.552` 和 `82325.370`。两者仍超限，但双方这些终点的交叉梯度缩放差不超过 `4.99e-9`。该结果进一步表明，微小边界误差可以改变进入的非凸区域。

[trajectory_execution_failure.json](../tests/fixtures/trajectory_execution_failure.json)保存完整精确输入、顶点矩阵、初始变量、双方最终变量、交叉梯度及紧凑诊断。原飞行文件的 SHA256 也保存在该文件中。C++ 轨迹诊断由 Python 对 C++ 最终变量重建 MINCO 后计算，方法与前文一致。

后续新增的 [execution_check](../src/isaac_pathcover/control/execution.py)属于执行策略，GCOPTER 的数学定义、返回状态和代价不变。该策略分别记录优化成功与交接接受状态。它用 `0.01` 检查速度平方、角速度平方、倾角和居中推力平方的原始残差，并用 `1e-6 m` 检查原走廊残差。拒绝时保留旧轨迹。

这个 `0.01` 来自惩罚平滑宽度，但并非官方可行性容差。姿态与推力检查仍基于采样。该策略的飞行结果必须单独验证，不能据此改写上述失败记录。

## 8. 性能

基础性能案例使用 2 个重叠盒和 2 段 MINCO 轨迹。每段有 16 个积分区间。每次试验计算目标及梯度 100 次，共运行 5 次。

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| 单次目标及梯度耗时中位数 | 2.564 ms | 1.963 ms |
| 加速比 | — | 1.306 倍 |

性能修改包括批量平坦性正反向计算、缓存导数基函数，以及批量计算 MINCO 端点导数。完整试验数据见 [trajectory_performance.json](../tests/fixtures/trajectory_performance.json)。

基础案例的完整优化耗时为 Python 67.96 ms、C++ 0.540 ms。真实场景记录如下：

| 场景 | Python 3 次耗时 | Python 中位数 | C++ 单次耗时 |
|---|---|---:|---:|
| medium | 0.4314、0.4670、0.4752 s | 0.4670 s | 0.01486 s |
| dense | 0.7059、0.6635、0.6569 s | 0.6635 s | 0.01799 s |

这些数值不包含地图、全局搜索、走廊生成或优化器初始化。Python 的 `optimize()` 计时包含最终轨迹重建；C++ 计时围绕 `lbfgs_optimize`。采样次数和计时边界存在差异，因此本表只报告观测耗时。

Python 优化阶段本身已接近或超过 0.5 s。该结果不支持完整系统达到 2 Hz 实时重新规划的结论。完整周期耗时应读取 Newton 运行日志。

## 9. 回归命令与证据索引

在仓库根目录运行：

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest \
  tests/test_trajectory_reference.py \
  tests/test_trajectory_scenes.py \
  tests/test_corridor.py \
  tests/test_risp_reference.py -q
```

原始汇总记录为 38 项通过，耗时 8.68 s。加入 medium0007 病态终点后，上述测试组为 **38 passed、1 xfailed**，耗时 9.27 s。其中 trajectory 为 29 项通过和 1 项严格 xfail。

该 xfail 从完整 fixture 重算 Python 梯度，再与保存的 C++ 梯度比较。它沿用正常 `3e-8` 缩放误差阈值，实际误差为 `0.00124633`。标记只接受断言失败；设置错误和非有限计算仍正常报错。若此点以后通过，严格 xfail 会使测试失败，要求重新审查标记。已知数值差异因此出现在常规 pytest 结果中。

命令直接读取已保存的官方输出，不要求安装 C++ 编译器。需要重新编译官方对照时，使用 [离线验证入口](../tests/reference/trajectory/README.md)。附加同一病态输入时，离线入口也保留 FAIL 报告。

| 证据 | 内容 |
|---|---|
| [trajectory_validation.json](../tests/fixtures/trajectory_validation.json) | 版本、源码及运行文件 SHA256、误差和性能汇总 |
| [trajectory_reference.json](../tests/fixtures/trajectory_reference.json) | MINCO、非零阻力平坦性及基础 GCOPTER 的完整参考值 |
| [trajectory_medium_reference.json](../tests/fixtures/trajectory_medium_reference.json) | 正式 medium 的全部数值输入及交叉输出 |
| [trajectory_dense_reference.json](../tests/fixtures/trajectory_dense_reference.json) | 正式 dense 的全部数值输入及交叉输出 |
| [trajectory_scene_validation.json](../tests/fixtures/trajectory_scene_validation.json) | 两图目标、梯度、迭代数、耗时及轨迹诊断 |
| [trajectory_replan_reference.json](../tests/fixtures/trajectory_replan_reference.json) | 两个实测异常计划、源顶点数量和修复回归 |
| [trajectory_singularity_diagnostics.json](../tests/fixtures/trajectory_singularity_diagnostics.json) | 比推力、倒置分母及 80 位 Decimal 检查 |
| [trajectory_execution_failure.json](../tests/fixtures/trajectory_execution_failure.json) | 本次 medium 3.5 s 超限计划的完整输入与 C++ 失败证据 |
| [trajectory_performance.json](../tests/fixtures/trajectory_performance.json) | 5 次优化前后微基准和基础优化耗时 |

