# 原始 RISP 离线对照

固定来源为 PathCover 提交 `8723a05d1993bf0199a453b47dc6ee167d493c23`。
驱动直接编译原始 `PathCover.hpp` 并调用点种子 `RISP`。
SUPER 的路径分段、走廊重叠及简化由 `tests/test_super_corridor.py` 验证。

在仓库根目录执行：

```bash
mkdir -p .runtime/risp_oracle
g++ -std=c++17 -O3 -I/usr/include/eigen3 \
  -I.reference/PathCover/src/corridor_planning/include/planner \
  tests/reference/corridor/oracle.cpp -lqhullcpp -lqhull_r \
  -o .runtime/risp_oracle/oracle
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/reference/corridor/verify.py \
  --upstream .reference/PathCover --oracle .runtime/risp_oracle/oracle \
  --output results/risp_reference.json
```

输入保存在 `tests/fixtures/risp_corridor_reference.json`。
验证使用其中的 436 个障碍点、首个种子和 6 个边界平面。
C++ 记录 `srand(37)` 的随机序列，Python 按相同余数选点。
验证比较归一化平面与每次分割后剩余点数。
正常单元测试读取已有夹具，不编译 C++。
