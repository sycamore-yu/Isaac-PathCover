# SUPER 地图、搜索与 RISP 走廊

## 来源与模块

SUPER 源码固定为 `2ad3419c127a617c6d7df6925e81a14175a9c096`。
PathCover 源码固定为 `8723a05d1993bf0199a453b47dc6ee167d493c23`。

| Python 模块 | 原始实现 |
|---|---|
| `mapping/rog_map.py` | ROG-Map ProbMap、CounterMap、InfMap、SlidingMap、RayCaster、ROGMap |
| `mapping/esdf.py` | ESDFMap 局部距离场与插值梯度 |
| `mapping/raycast.py` | 原始 DDA 线段查询的 CPU 编译 |
| `planning/search.py` | `super_core/astar.cpp` |
| `planning/search_kernel.py` | 原始 A* 扩展循环与优先队列的 CPU 编译 |
| `planning/corridor.py` | `super_core/corridor_generator.cpp` 和 `Polytope::SimplifySFC` |
| `planning/risp.py` | PathCover `RISP` 的随机分割与冗余面消除 |
| `planning/geometry.py` | 半空间几何、内切球线性规划与顶点枚举 |

旧 NanoVox、JPS/DMP 和 PathCover 整条路径覆盖算法已移除。

## 地图语义

概率使用源代码的 float32 对数几率。
一批更新中，同一格点只要存在命中，就忽略该格点的射线未命中。
格点越过占据或已知自由阈值时，更新粗分辨率计数。
只有粗格点类别改变时，才更新其球形膨胀邻居。
当膨胀步数为 1 时，保留原始 27 格立方邻域。

地图保留原始两套查询的区别。
`get_inf_grid_type` 与 `is_occupied_inflate` 对虚拟地板和天花板使用不同的阈值。
Python 按源表达式的运算顺序计算边界。
例如 `id * resolution + 0.5 * resolution` 不能替换为
`(id + 0.5) * resolution`，否则边界浮点判定可能改变。

射线保留原始终点格排除规则。
若多轴同时到达格点边界，顺序为 z、y、x。
实现按每轴原始累计加法生成事件时间，再稳定排序。
该变化消除了 Python 的逐格循环。

运行时使用 NumPy 有界数组。
地图滑动复制重叠区域，并清空离开区域。
源代码使用环形索引；存储布局不进入规划接口。
地图大小和垂直范围由现有 Newton 场景适配。
其余默认参数来自 SUPER `static_dense.yaml`。

移植包含规划使用的概率更新、占据与未知计数、射线融合、批量更新、
地图滑动、边界查询、盒搜索和最近格点查询。
可选 ESDF 与 frontier extraction 已移植，默认仍关闭。
frontier 保留 27 格邻域的已知自由计数和未知格筛选。
ESDF 使用 SciPy 的欧氏距离变换。
它保留局部更新区域、未知格作为非占据、三线性插值、
一阶梯度和原二阶梯度公式。
ROS 可视化由现有 Rerun 接口替代。

固定提交的 `esdf_map.cpp:225–236` 合并正负距离时，
直接使用局部索引，没有应用前面距离变换的环形偏移。
本移植保留该实际行为，并用非零地图原点对照。
该源问题不会影响默认关闭 ESDF 的规划流程。

## 搜索与走廊

A* 保留 26 邻接、原启发函数、`1 + 1e-5` tie breaker、
搜索距离终止、escape search，以及原概率地图邻居搜索分支。
优先队列保留 C++ 可变节点的更新方式。
EUCL 启发函数保留 `Eigen::Vector3i::norm()` 的整数截断。
A* 扩展循环与可变节点优先队列由 Numba 编译。
数组索引替代 Python 节点字典，邻接顺序和队列顺序不变。
已关闭节点先跳过查询。其他格点的分类在一次搜索内复用。
地图与查询参数在该次搜索内不变；返回时清空全部分类记录。
A* 的原始超时使用调用者传入的仿真时钟。
搜索默认值为 0.1 s，escape 为 0.2 s。
每次扩展后仍执行原检查。
Newton 共享时钟通过原子读访问，避免编译器把读取移出循环。
固定输入测试仍可使用 Python 时钟回调。
编译和首个完整扩展在规划器初始化时完成。

CorridorGenerator 保留原始路径分段、线段长度判定、点走廊修补、
相邻区域重叠、局部冗余区域移除与局部终点。
点种子与线段种子均进入同一 RISP 几何生成器。
备用轨迹也使用该生成器。
普通路径保留 `[9999, 9999, 9999]` 起点哨兵。
只有跳过占据前缀时，才记录实际移后的起点。
因此普通备用走廊使用实时机器人位置，符合原调用契约。

RISP 线段种子的几何扩展如下。
对随机选中的障碍点 `p`，将其投影到种子线段，得到 `q`。
令 `n = p - q`，构造：

```text
n · x <= n · q + (1 - alpha) ||n||² - robot_radius ||n||
```

这是 CIRI → RISP 的几何替换部分。
机器人半径在生成分离平面时进入约束。
被平面排除的障碍球同时从候选集合移除。
没有生成后的第二层轨迹或走廊接受规则。
当线段退化为点且半径为 0 时，结果精确退化为原始 RISP。

原 CorridorGenerator 保存但没有读取 `obs_skip_num`。
该无效配置已从运行接口移除。
`iris_iter_num` 只控制被替代的 CIRI 迭代，因此也不进入 RISP 接口。

## 固定输入数值验证

离线驱动直接编译 7 个原始 ROG-Map C++ 源文件。
唯一头文件适配移除未使用的 ROS PCL 转换依赖。
测试没有重新实现一份 C++ 地图。

`tests/fixtures/super_mapping_reference.json` 包含：

- 射线融合开启和关闭两种配置。
- 初始化、固定点云导入、4 次扫描后的 3 个状态。
- 每个状态 102 个查询点，共 612 组查询。
- 普通与膨胀占据查询、格点类型、对数几率和盒搜索。
- 零长度、负坐标、正反方向、相同边界时刻的射线。
- 开启 ESDF/frontier 的 4 个阶段，共 408 组额外地图查询。
- 216 个 ESDF 距离、插值、一阶梯度和二阶梯度输入。
- 408 个 frontier 查询，以及 4 个阶段的 frontier 盒搜索。

本机对照结果为：

| 项目 | 结果 |
|---|---:|
| 对数几率最大绝对误差 | 0 |
| 普通格点类型差异 | 0 / 1,020 |
| 膨胀格点类型差异 | 0 / 1,020 |
| 普通与膨胀占据布尔差异 | 0 |
| 盒搜索坐标与射线访问序列 | 一致 |
| Frontier 布尔差异 | 0 / 408 |
| ESDF 距离、插值和两阶梯度最大绝对误差 | 0 |

紧凑结果和夹具哈希保存在 `tests/fixtures/super_mapping_validation.json`。

点种子 RISP 仍使用保存的官方随机序列验证。
重新编译原始头文件后，归一化平面最大误差为 `8.326672684688674e-16`，
每次分割剩余点数完全一致。

`tests/fixtures/super_search_reference.json` 由原始 `astar.cpp` 生成。
18 个路径与状态场景、4 个最近格点输入均通过。
原始线段查询的 44 个输入也比较布尔结果和局部终点。

其他测试覆盖：删除占据后的增量膨胀、命中优先、地图滑动、
线段种子覆盖、机器人球排除、搜索距离终止、连续走廊、
点种子备用走廊、起点哨兵与占据前缀两分支，
以及 SimplifySFC 的起终区域选择。
地图更新后的搜索测试验证分类记录不会跨搜索保留。

重建地图夹具：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/reference/super_mapping/regenerate.py \
  /home/tong/tongworkspace/reference_repos/SUPER
```

运行当前验证：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_super_mapping.py tests/test_super_search.py \
  tests/test_super_corridor.py tests/test_risp_reference.py
```

## 已测量的性能改动

搜索首先查询起点所在格。
该格满足原条件时，直接返回，避免构造大球邻居列表。
A* 后续进一步编译原扩展循环；结果见下节的真实失败输入。
批量线段检查在一次调用中缓存重复格点的邻域结果。

内切球仍由 SciPy 安装的 HiGHS 求解。
实现直接传递固定维度模型，删除 `linprog` 每次重复的通用输入解析。
目标、半空间、自由变量和有符号半径规则保持一致。
HiGHS 使用 1 个线程，求解器实例在当前规划进程中复用。
单次内切球 LP 在固定输入中从 1.78 ms 降至 0.103 ms。
完整 Newton 飞行性能见总验收报告。

### Newton 停顿的搜索输入重放

`results/super_risp_development_20261115/search_failure` 保存真实失败输入。
medium 的 14 次扫描逐帧匹配原点、点数和地图版本。
停止位置为 `[2.1848917, -2.4660051, 1.3468595]`。
该位置存在 7.066 m 路径，需要 2,954 次扩展。
因此该次持续停顿来自搜索超时，而非不存在路径。

dense 初始输入存在 7.035 m 路径，需要 11,590 次扩展。
原 Python 搜索的 profile 为 2.275 s。
其中地图与邻域查询为 1.700 s。
CPU 编译和原子时钟读取后，3 次测量为 46.91–47.79 ms。
56 个路径点和扩展次数与改动前逐位一致。
所有原始超时值和检查位置保持不变。

medium 同输入在原子时钟版本中为 12.52–12.92 ms。
独立进程持续更新共享时钟时，搜索在仿真耗时 0.11 s 返回 `TIME_OUT`。
该次执行扩展 3,609 个节点，墙钟耗时 17.42 ms。

后续 dense 飞行在 `[2.6312754, 4.2243991, 1.4146289]` 再次停顿。
`dense_stopped/input.npz` 固定了该时刻的地图和搜索输入。
16 次扫描的原点、点数与地图版本均完全重现。
固定时钟下，原搜索需要 68,765 次扩展，生成 60 个路径点。
终点路径长度为 7.001972184338276 m。
所有测量使用相同输入和固定零共享时钟，仍执行原超时检查。

| 实现 | 3 次耗时 / ms |
|---|---|
| 优化前 | 290.90、287.52、281.99 |
| 先跳过关闭节点 | 183.97、184.44、181.26 |
| 复用格点分类 | 75.11、73.79、72.41 |

优化后的路径、扩展数和代价与固定基准逐位相同。
输入、原始测量和源码哈希均保存在该目录。
`tests/reference/super_mapping/benchmark_search.py` 可重复上述比较。
数据保存在 `search_failure/shared_clock_validation.json`。

重放命令：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/reference/super_mapping/replay_flight_search.py \
  results/super_risp_development_20261115/newton_jit/medium/seed000 \
  --until 24.2 --output results/super_risp_development_20261115/search_failure/medium_stopped
```
