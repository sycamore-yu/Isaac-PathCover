# SUPER + RISP 的 Rerun 录制

录制直接读取 Ideal 或 NMPC 飞行数据，不重新运行仿真。
每条录制使用同一批 `flight.npz`、`planning.json`、`fsm.json` 和轨迹多项式。

## 10D TurboMPC 修复后的对照

本次输出目录为 `results/turbompc_tracking_20261115_222728/rerun/`。
`nmpc/` 保存修复后的 Medium seed2 与 Dense seed4。
`ideal_baseline/` 保存原 Ideal 录制的副本，SHA256 与原文件一致。
两个场景的两种控制模式均无碰撞到达。

2 条新 NMPC 录制均通过 `rrd verify`。Ideal 沿用已验证的录制。
结果、固定轨迹诊断和验证见 [跟踪修复报告](controller_tracking_fix.md)。

- [本次 Rerun 回放](http://100.92.244.71:9118?url=rerun%2Bhttp%3A%2F%2F100%2E92%2E244%2E71%3A9898%2Fproxy)
- [本次录制与统计下载](http://100.92.244.71:9119/)

```bash
/home/tong/.local/share/pipx/venvs/rerun-sdk/bin/python scripts/record_rerun.py \
  --scene both --source results/turbompc_tracking_20261115_222728/nmpc \
  --output results/turbompc_tracking_20261115_222728/rerun/nmpc --fps 12
/home/tong/.local/bin/rerun results/turbompc_tracking_20261115_222728/rerun/*/*.rrd
```

## 初版 Ideal 与 10D TurboMPC 对照（保留）

原输出目录为 `results/ideal_turbompc_20261115_1927/rerun/`。
`ideal/` 和 `nmpc/` 分别保存 Medium seed2 与 Dense seed4。
录制名称包含控制器。NMPC 另有实际求解耗时曲线，单位为 ms。

4 条录制均通过 `rrd verify`。结果和归因见 [对照报告](controller_comparison.md)。

- [初版 Rerun 回放](http://100.92.244.71:9116?url=rerun%2Bhttp%3A%2F%2F100%2E92%2E244%2E71%3A9897%2Fproxy)
- [初版录制与统计下载](http://100.92.244.71:9117/)

```bash
/home/tong/.local/share/pipx/venvs/rerun-sdk/bin/python scripts/record_rerun.py \
  --scene both --source results/ideal_turbompc_20261115_1927/ideal \
  --output results/ideal_turbompc_20261115_1927/rerun/ideal --fps 12
/home/tong/.local/share/pipx/venvs/rerun-sdk/bin/python scripts/record_rerun.py \
  --scene both --source results/ideal_turbompc_20261115_1927/nmpc \
  --output results/ideal_turbompc_20261115_1927/rerun/nmpc --fps 12
/home/tong/.local/bin/rerun rrd verify results/ideal_turbompc_20261115_1927/rerun/*/*.rrd
/home/tong/.local/bin/rerun results/ideal_turbompc_20261115_1927/rerun/*/*.rrd
```

## 历史录制

原有 medium seeds 0–2、dense seeds 0–5，共 9 条录制保持原样。
到达、碰撞和越界状态分别保留。9 个文件均通过 `rrd verify`。

- [工作机 Rerun 回放](http://100.92.244.71:9114?url=rerun%2Bhttp%3A%2F%2F100%2E92%2E244%2E71%3A9896%2Fproxy)
- [录制文件与汇总](http://100.92.244.71:9115/)

### 历史数据的生成与检查

本机的独立 SDK 环境使用 Rerun 0.38.1。

```bash
/home/tong/.local/share/pipx/venvs/rerun-sdk/bin/python scripts/record_rerun.py \
  --scene both --source results/super_risp_validation_20261115 \
  --output results/rerun_super_risp_20261115 --fps 12
/home/tong/.local/bin/rerun rrd verify results/rerun_super_risp_20261115/*.rrd
/home/tong/.local/bin/rerun results/rerun_super_risp_20261115/*.rrd
```

其他机器可使用装有 NumPy 和 `rerun-sdk` 的 Python 环境执行录制脚本。
SDK 与 Isaac Lab 环境独立。
默认转换各场景下全部已保存的飞行。`--seeds` 可指定种子列表。

## 回放内容

| 内容 | 表示 |
|---|---|
| 场景 | 原障碍实体、原始起终点、SUPER 最近非占据格点调整后的目标 |
| 无人机 | Newton 实际位置、姿态、机体与旋翼位置 |
| 橙色路径 | 实际飞行路径 |
| 蓝色路径 | 控制器实际使用的参考位置 |
| 紫色轨迹 | 当时已提交的完整 CmdTraj |
| 黄色轨迹 | 同次提交的备用轨迹 |
| 速度与误差 | 实际速度、实际使用的参考速度、位置误差 |
| 规划 | 各次完整规划耗时、含传感器更新的工作进程耗时 |
| NMPC | 每次滚动求解的实际墙钟耗时，独立于规划耗时 |
| 执行 | 备用标志、FSM 状态和接触力 |
| 事件日志 | 规划状态、备用生成状态、FSM 切换与来源回调 |

轨迹按原始仿真时间显示。控制器保持最后发布的命令。
`flight.npz` 的 `command_time` 保存命令采样的仿真时间。
物理状态记录在该控制步结束时。
四元数从 Newton 的 `[w,x,y,z]` 转换为 Rerun 的 `[x,y,z,w]`。

12 帧/s 仅控制三维动画的抽样密度。
速度、误差、备用标志、状态机和接触力曲线保留全部 100 Hz 控制样本。
原始 100 Hz 控制记录和 400 Hz 物理记录仍保存在飞行数据中。
规划和状态机事件不按回放帧率丢弃。

`summary.json` 保存每条录制的源结果、帧数、字节数和轨迹提交数量。
`.rrd` 和原始飞行数据保存在 `results/`。
历史运行结果见 [SUPER 验收报告](super_validation.md)。
