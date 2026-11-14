# 官方 RISP / PathCover 离线对照

本目录只保存输入输出驱动和比较脚本。算法正文来自外部官方源码，未复制到本目录。正式 Python 包和 `pytest` 均不调用该可执行程序；不需要 Isaac-L2F、ROS、Gazebo、MOSEK、pybind11 或 ctypes。

固定来源：<https://github.com/kunalnk123690/PathCover>，提交 `8723a05d1993bf0199a453b47dc6ee167d493c23`。目标头文件是 `src/corridor_planning/include/planner/PathCover.hpp`，SHA-256 为 `811bcdd2482395be80e383148bde92b3a7e281785259a9f3dbd824fcdb89e427`。`verify.py` 会检查提交和头文件哈希。

## 构建

准备 C++17 编译器、Eigen 3 和 Qhull 开发库。Ubuntu/Debian 对应软件包是 `g++`、`libeigen3-dev`、`libqhull-dev`。Python 环境使用本仓库的 NumPy、SciPy 和 `isaac-pathcover` 安装。以下命令在本仓库根目录执行；上游路径可自行指定。

```bash
export PATHCOVER_ORACLE_SOURCE=/tmp/isaac-pathcover-official
git clone https://github.com/kunalnk123690/PathCover "$PATHCOVER_ORACLE_SOURCE"
git -C "$PATHCOVER_ORACLE_SOURCE" checkout 8723a05d1993bf0199a453b47dc6ee167d493c23
mkdir -p build/reference/corridor
g++ -std=c++17 -O3 -I/usr/include/eigen3 \
  -I"$PATHCOVER_ORACLE_SOURCE/src/corridor_planning/include/planner" \
  tests/reference/corridor/oracle.cpp -lqhullcpp -lqhull_r \
  -o build/reference/corridor/oracle
```

若 Eigen 安装在其他位置，请修改其 `-I` 路径。已经存在固定提交的官方源码目录时，直接设置 `PATHCOVER_ORACLE_SOURCE` 并从编译步骤开始。

## 运行与结果

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python tests/reference/corridor/verify.py \
  --upstream "$PATHCOVER_ORACLE_SOURCE" \
  --oracle build/reference/corridor/oracle \
  --fixture tests/fixtures/risp_corridor_reference.json \
  --repetitions 30 \
  --output results/corridor_official_validation.json
```

输入夹具包含 436 个障碍点、7 个路径点和 6 个工作空间平面。该夹具固定 `srand(37)`、`max_horizon=6`、源码默认 `alpha=0.01`。驱动直接调用官方 `RISP` 和 `pathCover`，输出半空间、种子、每轮剩余点数及 C `rand()` 返回值。Python 复现 `rand_value % remaining_count`，避免将两种随机数发生器的相同整数种子误认为相同随机序列。

脚本同时比较本次 C++ 输出和已保存的官方夹具。比较会匹配归一化平面行，允许凸包枚举顺序不同；种子点按路径顺序比较，剩余点数要求完全一致。几何误差门限为 `3e-13`。成功时 JSON 的 `passed` 为 `true`，退出码为 0。失败时保存错误指标并返回非零退出码；编译错误或无效输入直接报错。

报告包含逐次耗时、p50/p95、源码和二进制哈希、Python/数值库版本，以及 C 库版本。C++ 计时只围绕 `pathCover` 调用；Python 计时只围绕当前版本的 `path_cover` 调用，二者均不含进程启动或结果输出。该脚本复测当前实现，不覆盖历史性能证据。

保存的 `rand()` 序列来自验证主机的 C 库。若换用不同 C 库，`srand(37)` 不保证产生相同序列。此时本次 C++/Python 对照可能仍一致，但保存夹具对照或 `same_saved_random_draws` 会失败，报告会保留两组结果。

现有版本化证据：

- [原始输入与官方输出](../../fixtures/risp_corridor_reference.json)。
- [历史数值对照与真实场景性能优化](../../fixtures/risp_validation.json)。
- [验证方法、覆盖范围和前缀裁剪修复](../../../docs/corridor_validation.md)。

本目录文件名不匹配 `test_*.py`。常规单元测试读取 JSON 夹具，不编译或启动官方程序。
