# 原 A* 与 ROG-Map 对照

`regenerate.py` 直接编译 SUPER `2ad3419` 的 `astar.cpp` 和 7 个 ROG-Map 源文件。
固定输出位于 `tests/fixtures/super_search_reference.json`。
该文件保存原提交和编译源文件的 SHA256。
运行时不链接或调用这些 C++ 程序。

仅用 `include/` 中的最小适配替换 ROS 时钟、可视化和发布接口。
地图仍使用原 `ROGMap`，搜索仍使用原 `Astar`。
输入概率地图分辨率为 `0.1 m`，膨胀地图分辨率为 `0.2 m`。
障碍是固定的 `21 × 20` 点墙。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/reference/super_search/regenerate.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_super_search.py
```

`--no-build` 可重用已生成的离线程序。构建需要系统 Eigen、PCL 和 yaml-cpp。
`--map-objects .runtime/super_mapping_oracle` 可复用映射对照生成的 7 个原源码目标文件。

## 对照范围

对角搜索分别启用和禁用。每种设置包含 9 个输入：

- 空旷路径和绕墙路径。
- 搜索距离终止。
- 概率地图的细邻域膨胀。
- 未知单元按占据处理。
- 空旷起点、膨胀占据起点、地图外起点的逃逸搜索。
- 原 `0.1 s` 搜索超时分支。该输入的模拟时钟每次调用增加 `0.051 s`。

其余输入使用不推进的模拟时钟，避免机器负载改变路径验收结果。
18 个输入的返回状态全部一致，路径逐点误差均在 `2e-15 m` 内。
4 个最近非占据单元输入覆盖空闲、膨胀占据、概率占据和虚拟地板。
这 4 个输出同样通过数值对照。

44 个原 `isLineFree` 输入验证空邻域、半径为 1 和 2 的邻域、长度限制、
未知单元、墙面、虚拟地板、地图外线段、反向射线和零长度线段。
布尔结果和返回的局部目标与原源码一致。
成功时局部目标为输入终点。失败时为首个阻塞单元中心。
批量查询按输入顺序停止，并只复用同一次调用内相同邻域的占据结果。

## 原源码中的实现细节

`astar.cpp:141` 的欧氏启发式使用 `Eigen::Vector3i.norm()`。
该返回值为整数，平方根会截断。Python 保留该截断。
如果直接使用浮点欧氏范数，固定绕墙输入会从 19 个点变为 18 个点。

原逃逸搜索始终使用 26 个邻居，不读取 `allow_diag`。
细邻域查询忽略 A* 局部搜索范围外的邻居。
可变节点优先队列保留 libstdc++ 的入堆、出堆和相同优先级处理顺序。

`rog_map.cpp:108–121` 的 `findNearestInfCellThat` 使用概率地图网格生成候选中心，
然后调用膨胀地图查询。返回位置因此属于原始分辨率网格，不能换成膨胀网格中心。

`mapping/raycast.py` 用 Numba 编译原串行 DDA 与邻域循环。
`fastmath=False` 保留浮点运算顺序。Numba 缓存仅保存编译结果，不包含另一套规划算法。
