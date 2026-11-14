# 轨迹算法的离线官方对照

此目录只含验证驱动。它直接调用官方 C++ 头文件，不复制算法正文，不提供 Python 扩展，也不进入安装包或常规 pytest。

固定官方提交：

```text
8723a05d1993bf0199a453b47dc6ee167d493c23
```

需要项目的 Python 3.12 数值环境、C++17 编译器和 Eigen 3 头文件。无需启动 Isaac Lab、Newton 或 ROS。

## 一条命令重算

先准备官方源码。已有固定版本 checkout 时可跳过：

```bash
git clone https://github.com/kunalnk123690/PathCover.git .reference/PathCover
git -C .reference/PathCover checkout --detach 8723a05d1993bf0199a453b47dc6ee167d493c23
```

在 Isaac-PathCover 根目录运行：

```bash
.venv/bin/python tests/reference/trajectory/compare.py \
  --upstream .reference/PathCover \
  --output .reference/trajectory_reproduce
```

脚本检查 commit 和头文件 SHA256，并拒绝已修改的官方头文件。随后编译独立可执行文件，重算 MINCO S3/S4、非零阻力平坦性映射，以及 5 组 GCOPTER 输入：基础双盒、medium、dense 和 2 组实测重新规划回归。

所有输入均来自 [tests/fixtures](../../fixtures) 或基础回归测试定义的固定向量。GCOPTER 对照使用相同的顶点矩阵和变量，分别检查初值、保存的 C++ 终值及保存的 Python 终值。

如需双方重新执行 L-BFGS：

```bash
.venv/bin/python tests/reference/trajectory/compare.py \
  --upstream .reference/PathCover \
  --output .reference/trajectory_reproduce \
  --optimize
```

优化报告保留双方状态、代价、迭代数、耗时和终点交叉误差。非凸优化的最终代价不要求相等；成功状态不表示严格满足软约束。

## 产物

所有产物写入 `--output`。原始 fixture、官方源码和运行算法均不改动。

| 路径 | 内容 |
|---|---|
| build/oracle | 独立 C++ 对照可执行文件 |
| inputs/*.txt | 每个 GCOPTER 评估点的完整数值输入 |
| cpp/*.json | 官方头文件的新计算输出 |
| report.json | Python/C++ 误差、旧 fixture 对比、版本和 SHA256 |

脚本将 OpenBLAS 和 OpenMP 线程数固定为 1。可用 `--cxx` 指定编译器，用 `--eigen-include` 指定 Eigen 头文件目录。完整编译命令写入报告。

实际编译命令等价于：

```bash
g++ -std=c++17 -O3 -DPATHCOVER_DIM=3 \
  -I/usr/include/eigen3 \
  -I.reference/PathCover/src/trajectory_server/include \
  tests/reference/trajectory/oracle.cpp \
  -o .reference/trajectory_reproduce/build/oracle
```

可直接重放生成的输入：

```bash
.reference/trajectory_reproduce/build/oracle gcopter \
  < .reference/trajectory_reproduce/inputs/medium__initial.txt
```

`oracle primitives` 输出 MINCO 和平坦性参考值；`oracle gcopter optimize` 从标准输入中的变量执行原 L-BFGS。非有限数值输出为 JSON null，原优化状态单独保留。

## 已记录的奇异输入

可附加场景格式的完整 fixture：

```bash
.venv/bin/python tests/reference/trajectory/compare.py \
  --upstream .reference/PathCover \
  --output .reference/trajectory_execution_failure \
  --extra-fixture tests/fixtures/trajectory_execution_failure.json
```

此例来自 medium 的 3.5 s 计划。Python 和 C++ 都返回超限轨迹。Python 终点附近的梯度缩放差约为 `0.001246`，超过正常对照阈值，因此该命令预期返回非零并写出 FAIL 报告。它用于重现数值退化，不属于通过基准。不要放宽阈值来消除该记录。

5 组常规输入的同变量梯度阈值为 `3e-8`。已有 medium 重新规划 C++ 奇异终点使用已记录的 `1e-6` 阈值。报告同时保留实际误差。

实现细节和历史结果见 [数值验证文档](../../../docs/numerical_validation.md)。

