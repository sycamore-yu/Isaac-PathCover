# 映射与搜索的离线官方对照

本目录仅保存独立测试驱动，不复制官方算法正文。驱动通过头文件和源码调用官方 C++ 算法，生成已经纳入 `tests/fixtures/` 的参考数据。运行包和正常 pytest 不编译、导入或调用这些程序。

## 固定来源

官方仓库：<https://github.com/kunalnk123690/PathCover>

固定提交：`8723a05d1993bf0199a453b47dc6ee167d493c23`。

`regenerate.py` 检查该提交，并拒绝算法目录中存在未提交修改的 checkout。源码路径由 `--pathcover` 指定，不依赖 Isaac-L2F 或本机原始工作目录。

| 文件 | 用途 |
|---|---|
| `mapping_search.cpp` | 两组 NanoVoxMap 融合输入、4 种膨胀配置、27 组 JPS 方向表、两种权重的小场景 JPS/DMP 对照 |
| `scene_search.cpp` | 从文本文件读取任意固定栅格和起终点，运行官方 JPS/DMP |
| `regenerate.py` | 编译驱动，从本仓库的 medium/dense 资产生成相同输入，保存新输出并与冻结夹具比较 |

两份 C++ 驱动与生成原夹具时使用的文件字节相同，其 SHA-256 分别为：

```text
mapping_search.cpp:
cc10e647efef36849c2b4713bd494e108b9bedbdf5c0dedc6b1c06f5d0976a4a

scene_search.cpp:
88001a9606ed5bc774ad4300a579377f60741feb21be3fe9d5650b8ea2b260a6
```

## 一条命令重新计算

前提是已安装本项目的 Python 依赖，并有 C++17 编译器、Eigen 3 和 Boost 头文件。原验证使用 GCC 13.3、系统 Eigen 3，未启用 OpenMP、CUDA 或 ROS。

先准备固定官方源码；已有对应 checkout 时不需要重新克隆：

```bash
git clone https://github.com/kunalnk123690/PathCover .reference/PathCover
git -C .reference/PathCover checkout 8723a05d1993bf0199a453b47dc6ee167d493c23
```

在本仓库根目录执行：

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python tests/reference/mapping/regenerate.py \
  --pathcover .reference/PathCover \
  --output .reference/mapping-recomputed
```

如果 Eigen 不在 `/usr/include/eigen3`，通过 `--eigen /path/to/eigen3` 指定。通过 `--cxx /path/to/compiler` 指定其他编译器。

输出目录包含：

- `build/mapping_search_oracle` 与 `build/scene_search_oracle`：独立可执行文件。
- `mapping_search.json`：重新生成的小场景官方输出。
- `medium.txt` 与 `dense.txt`：可独立复用的官方场景搜索输入。
- `search_scene_reference.json`：真实场景的官方输出及 Python 同输入对照。
- `comparison.json`：与已提交夹具的数值比较、完整编译命令和驱动哈希。

复算不会覆盖冻结夹具或历史性能数据。程序返回非零状态时，编译或数值比较失败。时间值随主机负载变化，不参与数值一致性判定。等代价路径允许采用不同坐标序列；程序记录路径是否逐点相等，并检查目标值与势场。

正常 Python 回归测试另行运行：

```bash
.venv/bin/python -m pytest tests/test_mapping.py tests/test_search.py -q
```

## 手工编译与运行

`regenerate.py` 使用以下命令结构。变量只指定官方源码位置；本目录没有替代算法实现。

```bash
PATHCOVER_SOURCE="$PWD/.reference/PathCover"
mkdir -p .reference/mapping-manual

g++ -O2 -std=c++17 -I/usr/include/eigen3 \
  -I"$PATHCOVER_SOURCE/src/mapping/nanovoxmap_ros/nanovoxmap/include" \
  -I"$PATHCOVER_SOURCE/src/mapping/occupancy_inflation/include" \
  -I"$PATHCOVER_SOURCE/src/third_party/jps_lib/include" \
  tests/reference/mapping/mapping_search.cpp \
  "$PATHCOVER_SOURCE/src/mapping/occupancy_inflation/src/point_cloud_inflater.cpp" \
  "$PATHCOVER_SOURCE/src/mapping/occupancy_inflation/src/voxel_bit_grid.cpp" \
  "$PATHCOVER_SOURCE/src/third_party/jps_lib/src/jps_planner/graph_search.cpp" \
  "$PATHCOVER_SOURCE/src/third_party/jps_lib/src/jps_planner/jps_planner.cpp" \
  "$PATHCOVER_SOURCE/src/third_party/jps_lib/src/distance_map_planner/graph_search.cpp" \
  "$PATHCOVER_SOURCE/src/third_party/jps_lib/src/distance_map_planner/distance_map_planner.cpp" \
  -o .reference/mapping-manual/mapping_search_oracle

.reference/mapping-manual/mapping_search_oracle \
  > .reference/mapping-manual/mapping_search.json

g++ -O2 -std=c++17 -I/usr/include/eigen3 \
  -I"$PATHCOVER_SOURCE/src/third_party/jps_lib/include" \
  tests/reference/mapping/scene_search.cpp \
  "$PATHCOVER_SOURCE/src/third_party/jps_lib/src/jps_planner/graph_search.cpp" \
  "$PATHCOVER_SOURCE/src/third_party/jps_lib/src/jps_planner/jps_planner.cpp" \
  "$PATHCOVER_SOURCE/src/third_party/jps_lib/src/distance_map_planner/graph_search.cpp" \
  "$PATHCOVER_SOURCE/src/third_party/jps_lib/src/distance_map_planner/distance_map_planner.cpp" \
  -o .reference/mapping-manual/scene_search_oracle

.reference/mapping-manual/scene_search_oracle \
  .reference/mapping-recomputed/medium.txt \
  > .reference/mapping-manual/medium.json
```

场景输入是空白分隔文本，依次包含：

1. `nx ny nz`。
2. `origin_x origin_y origin_z resolution start_x start_y start_z goal_x goal_y goal_z`。
3. `nx*ny*nz` 个 `int8` 栅格值，x 最快变化，即 NumPy `ravel(order="F")` 顺序。

medium/dense 输入由本仓库资产的表面点云，经 NanoVoxMap、障碍膨胀和 VoxelGrid 生成。输入的文本 SHA-256 和栅格字节 SHA-256 必须与冻结夹具一致。源参数为 JPS heuristic=`resolution`，DMP heuristic=`0`、cost weight=`0.1`、势场半径=`(1,1,1)`、搜索区域半径=`(2,2,2)`。

## 验证范围

脚本重新计算官方数值参考，不重放优化前后的历史 5 次性能试验。旧性能试验使用固定 `180×16`、`20 m` 观测输入；新飞行配置使用 `440×16`、`0.1–100 m` 和机体安装变换。两者不能混用。

详细误差、地图语义、随机性及计时限制见 [映射与搜索验证报告](../../../docs/mapping_search_validation.md)。
