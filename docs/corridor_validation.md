# RISP 与 PathCover 走廊验证

Python 移植与官方 C++ 在相同输入、相同随机抽样序列下，生成相同的走廊几何。归一化平面最大绝对误差为 `1.1102230246251565e-15`，种子点最大绝对误差为 `8.881784197001252e-16`。真实 `medium`、`dense` 场景的 RISP 性能优化保留了相同输出；两场景的走廊调用耗时中位数分别下降 40.54% 和 71.00%。本页只报告走廊生成和相关接口验证，不把这些结果当作 Newton 飞行验收结果。

## 1. 来源与数学定义

官方源码固定为提交 [`8723a05d1993bf0199a453b47dc6ee167d493c23`](https://github.com/kunalnk123690/PathCover/tree/8723a05d1993bf0199a453b47dc6ee167d493c23)。主要依据如下：

- [`PathCover.hpp`](https://github.com/kunalnk123690/PathCover/blob/8723a05d1993bf0199a453b47dc6ee167d493c23/src/corridor_planning/include/planner/PathCover.hpp)：随机迭代空间划分（RISP）、冗余平面消除和沿路径扩展走廊。
- [`trajectory_server.cpp`](https://github.com/kunalnk123690/PathCover/blob/8723a05d1993bf0199a453b47dc6ee167d493c23/src/trajectory_server/src/trajectory_server.cpp)：`findStartPolytope` 的前缀裁剪规则。
- [`risp.py`](../src/isaac_pathcover/planning/risp.py)、[`corridor.py`](../src/isaac_pathcover/planning/corridor.py)、[`pipeline.py`](../src/isaac_pathcover/planning/pipeline.py)：对应的 Python 实现。

所有多面体统一表示为 `A @ x <= b`。对于当前种子点 $y$ 和随机选中的障碍点 $p$，分离平面为

$$
a=p-y,\qquad b=a^\mathsf{T}y+(1-\alpha)\|a\|^2,\qquad \alpha=0.01.
$$

每轮保留满足 $a^\mathsf{T}p_i\le b$ 的点，保留点的相对顺序不变。处理完障碍点后，追加工作空间边界，并通过对偶凸包消除冗余平面。Python 实现未增加多面体收缩；机体净空由前置占据膨胀提供。

沿路径扩展也保留源码的具体行为：`size > max_horizon` 在新增多面体后检查。因此默认 `max_horizon=6` 最多生成 7 个区域。提前达到该上限时，轨迹终点使用源码发布的局部目标。终点判定中的 `.all()` 条件也保持源码定义。[对应回归测试](../tests/test_corridor.py) 检查这些行为。

## 2. 相同随机抽样的官方对照

版本化输入和 C++ 输出保存在 [`risp_corridor_reference.json`](../tests/fixtures/risp_corridor_reference.json)。汇总指标保存在 [`risp_validation.json`](../tests/fixtures/risp_validation.json)。夹具包含 436 个障碍点、7 个路径点，完整走廊包含 4 个区域。

离线参考程序直接包含官方 `PathCover.hpp`，分别调用 `RISP` 和 `pathCover`。它记录 C 库 `rand()` 的原始返回值。Python 测试用 `ReferenceRandom` 重放这些值，并按 `rand_value % current_point_count` 选择点。这样固定的是每次抽样，而不只是整数种子。正式 Python 运行使用 NumPy 随机数生成器；相同整数种子并不意味着其序列与 C 库 `rand()` 相同。

平面比较先除以法向量范数，再匹配平面行。该比较允许凸包库返回不同的行顺序，但不改变不等式方向。种子点保持路径顺序直接比较。每轮剩余点数逐项比较。

| 对照项 | 结果 |
| --- | ---: |
| 区域数 | 4，完全一致 |
| 归一化平面最大绝对误差 | `1.1102230246251565e-15` |
| 种子点最大绝对误差 | `8.881784197001252e-16` |
| 每轮剩余点数 | 完全一致 |

[单区域测试](../tests/test_risp_reference.py) 同时验证种子严格在区域内部，以及每个输入障碍点至少违反一个最终半空间。[完整走廊测试](../tests/test_corridor.py) 验证相邻区域交集的最大内切球半径大于 `1e-3`。

连续覆盖测试不是路径点采样。对于每一条路径线段，测试求出它与每个凸多面体相交的参数区间，再检查区间并集覆盖 `[0, 1]`。这验证了该无截断夹具的全部连续路径段。截断走廊另行测试区域上限和局部终点，不要求它覆盖尚未规划的全局路径后缀。

普通回归只读取已保存的官方结果，不需要 C++ 编译器、ROS 或 C++ 扩展。离线参考程序不参与正式运行。需要重新生成对照时，可使用仓库中的[独立驱动与编译说明](../tests/reference/corridor/README.md)。驱动只包含输入输出代码，直接调用固定提交的官方头文件。

## 3. 官方实现与 Python 的计时对照

下表是同一合成夹具的 30 次计时记录。它测于 RISP 性能优化之前；不能把这组数值当作当前实现的性能。C++ 计时范围是源码 `pathCover` 调用，不包含可执行程序启动和结果输出。原始汇总位于 [`official_fixture_comparison_before_performance_change`](../tests/fixtures/risp_validation.json)。

| 实现 | p50 / ms | p95 / ms |
| --- | ---: | ---: |
| 官方 C++ | 0.459503 | 0.611909 |
| Python，优化前 | 2.801261 | 3.621367 |

这组测量的 Python/C++ p50 耗时比为 `6.096286`。后续真实场景性能对照使用另一组输入，不与本表混算加速比。

## 4. medium 与 dense 的等价性能优化

优化仅改动 RISP 每轮点积和点集压缩。基线使用 `einsum` 与布尔索引；最终实现使用矩阵向量乘法和稳定压缩：

```python
distances = work @ normal
work = np.compress(distances <= offset, work, axis=0)
```

分离平面、抽样次数、点的顺序、冗余消除和 PathCover 遍历规则均未改变。测量对象是前缀裁剪前的完整走廊生成调用。`medium` 和 `dense` 均生成 7 个区域。

每个版本、每个场景测量 5 次，固定输入与随机种子。环境设置为 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。同一主机当时还有其他工作进程，绝对耗时包含资源争用；5 次样本的 p95 只用于描述本次测量。记录位于 [`matched_performance_change`](../tests/fixtures/risp_validation.json)。

| 场景 | 实现 | p50 / ms | p95 / ms |
| --- | --- | ---: | ---: |
| medium | `einsum` + 布尔索引 | 416.016 | 469.571 |
| medium | `einsum` + `compress` | 290.352 | 331.759 |
| medium | 矩阵乘法 + `compress` | 247.347 | 298.965 |
| dense | `einsum` + 布尔索引 | 484.741 | 517.733 |
| dense | `einsum` + `compress` | 441.088 | 479.006 |
| dense | 矩阵乘法 + `compress` | 140.582 | 174.100 |

基线/最终实现的 p50 耗时比分别为 `1.682` 和 `3.448`，对应耗时下降 `40.54%` 和 `71.00%`。这些数值只描述走廊阶段，不包含地图融合、膨胀、路径搜索、轨迹优化或飞行。

两场景优化前后的逐项检查结果如下：

| 场景 | 归一化平面最大绝对差 | 种子点最大绝对差 | 每轮剩余点数 |
| --- | ---: | ---: | --- |
| medium | 0 | 0 | 完全一致 |
| dense | 0 | 0 | 完全一致 |

证据 JSON 还保存了点云、路径、边界及优化前后函数的 SHA-256。两张真实地图对应的路径哈希分别以 `94542a8ba6b0262c`、`a2d6ffde80d42067` 开头；完整哈希以该 JSON 为准。这使性能对照与特定输入、特定改动绑定，而非仅凭场景名称判断输入相同。

## 5. 起始区域前缀裁剪修复

官方 `findStartPolytope` 选择包含当前位置的最后一个区域。若当前位置不在任何区域内，则选择归一化正半空间违反量的欧氏范数最小的区域；并列时选择序号较大的区域。只使用第一个区域会保留已经越过的走廊前缀，从而改变轨迹优化问题。

Python 流水线现在按该规则裁剪 `polytopes`、`seeds` 和 `remaining_counts`，并记录 `source_start_index`。随机种子为 0 的真实场景证据如下：

| 场景 | 原始区域数 | `source_start_index` | 传给 GCOPTER 的区域数 |
| --- | ---: | ---: | ---: |
| medium | 7 | 1 | 6 |
| dense | 7 | 0 | 7 |

裁剪后的完整几何、起终状态和官方数值结果，分别保存在 [`trajectory_medium_reference.json`](../tests/fixtures/trajectory_medium_reference.json) 与 [`trajectory_dense_reference.json`](../tests/fixtures/trajectory_dense_reference.json)。[`trajectory_scene_validation.json`](../tests/fixtures/trajectory_scene_validation.json) 同时记录种子、裁剪序号、段数和输入哈希。这里引用的是裁剪后的夹具，未将修复前的 7 段 medium 输入混作正式输入。

[`test_start_polytope_follows_source_containment_and_excess_ranking`](../tests/test_corridor.py) 覆盖重叠区选择、区域外距离排序和并列选择。真实场景的轨迹数值回归直接读取已裁剪的夹具，因此验证的是该输入的下游计算；它不会在每次测试时重新生成整张地图和原始走廊。这是当前回归覆盖的边界。

## 6. 失败处理与测试入口

| 行为 | 现有测试 |
| --- | --- |
| 空点云返回工作空间边界；种子与障碍重合或落在边界上时失败 | [`test_risp_reference.py`](../tests/test_risp_reference.py) |
| 随机种子可复现、障碍点排除、正体积交集、连续路径覆盖、horizon 和局部目标 | [`test_corridor.py`](../tests/test_corridor.py) |
| 点云到可执行 MINCO 轨迹；越界目标返回失败且无候选轨迹 | [`test_pipeline.py`](../tests/test_pipeline.py) |
| 地图占据变化使缓存失效；未变化时保留缓存；调用者状态格式错误继续抛出异常 | [`test_pipeline.py`](../tests/test_pipeline.py) |
| 新规划失败时保留旧轨迹及其时钟；首次失败时保持初始悬停点 | [`test_execution.py`](../tests/test_execution.py) |
| 裁剪后真实场景的目标函数与解析梯度数值对照 | [`test_trajectory_scenes.py`](../tests/test_trajectory_scenes.py) |

可在仓库根目录执行走廊及交接回归：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q \
  tests/test_risp_reference.py tests/test_corridor.py \
  tests/test_pipeline.py tests/test_execution.py
```

障碍排除测试检查输入膨胀点云。Newton 中的实体碰撞、跟踪误差、到达判据和完整规划周期耗时，由飞行验收单独记录。随机算法的其他种子可能生成不同几何；本页的逐元素数值结论仅适用于记录的同输入、同抽样对照。

