# ROG-Map C++ 对照

`regenerate.py` 直接编译 SUPER 2ad3419 的 ROG-Map 源码。
依赖为 C++17、Eigen、PCL 与 yaml-cpp。
`include/pcl_conversions/pcl_conversions.h` 只替代未使用的 ROS 转换头文件。

在仓库根目录执行：

```bash
.venv/bin/python tests/reference/super_mapping/regenerate.py /home/tong/tongworkspace/reference_repos/SUPER
.venv/bin/python -m pytest -q tests/test_super_mapping.py
```

二进制和构建日志存入 `.runtime/super_mapping_oracle/`。
该目录缓存逐文件编译的 7 个原始源对象。
固定 C++ 输出写入 `tests/fixtures/super_mapping_reference.json`。
夹具包含射线开关、ESDF、frontier 查询及增量移除。
ESDF 还比较距离插值和一阶、二阶梯度。
正式 Python 包不读取或执行 C++ 源码及二进制。

`replay_flight_search.py` 从 Newton 飞行记录恢复扫描与冻结地图。
旧记录通过传感器原点匹配状态；新记录直接使用扫描位置和姿态。
输出保留点数、原点误差、地图版本、地图数组和搜索路径。
