# 安装与依赖验证

仿真安装使用 Isaac Lab v3.0.0 的 4 个官方源码组件。安装脚本准备固定提交的源码目录，再用 `uv.lock` 安装依赖。规划、轨迹优化和控制代码来自本仓库，不依赖 Isaac-L2F 的 Python 环境。

## 1. 安装仿真环境

需要 Python 3.12、Git、uv，以及可运行 PyTorch CUDA 13.0 的 NVIDIA 环境。在仓库根目录执行：

```bash
python3 scripts/install_simulation.py
```

如果 uv 安装在用户目录但没有加入 `PATH`，显式指定其位置：

```bash
python3 scripts/install_simulation.py --uv "$HOME/.local/bin/uv"
```

脚本执行以下步骤：

1. 在本仓库 `.runtime/isaaclab` 获取官方固定提交。
2. 稀疏检出 `apps` 和 4 个源码组件，保留根目录资源。
3. 验证提交、组件目录及已跟踪文件没有修改。
4. 执行 `uv sync --locked --python 3.12 --extra simulation --extra test`。

默认 Python 环境为本仓库 `.venv`。源码依赖及环境均不进入 Git 提交。安装脚本不修改 Isaac-L2F，也不使用其环境中的包。

NMPC 使用 `nvmath-python==1.0.0` 与 `nvidia-cudss-cu13==0.8.0.10`。这两个依赖已加入 `simulation` extra 和锁文件。控制求解器源码保存在本仓库。

完成后运行两图演示：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m isaac_pathcover.demo --scene both --seeds 0 --output results/demo
```

完整回归测试：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

只准备源码、不安装或修改 Python 环境：

```bash
python3 scripts/install_simulation.py --prepare-only
```

也可以把环境安装到本仓库的另一个目录：

```bash
python3 scripts/install_simulation.py \
  --venv .runtime/isolated-venv \
  --uv "$HOME/.local/bin/uv"
```

已有依赖 checkout 的提交不符合要求时，脚本报告错误。它不会强制覆盖已有提交或已修改的上游文件。

## 2. 只安装规划与测试

纯规划不需要 Isaac Lab 或 Newton。使用 uv 的 pip 接口安装基本依赖及测试包，可以避免项目级解析尚未准备的本地仿真源码目录。

```bash
uv venv --python 3.12 .runtime/planning-venv
uv pip install --python .runtime/planning-venv/bin/python '.[test]'

.runtime/planning-venv/bin/python -m pytest \
  tests/test_super_mapping.py tests/test_super_execution.py -q

.runtime/planning-venv/bin/python -m isaac_pathcover.demo \
  --scene both --seeds 0 --plan-only --output results/planning
```

基本依赖仍包含 PyTorch，因为本仓库的动力学参数与控制模块使用张量接口。仿真入口使用上一节的安装脚本；它负责四个源码组件的目录准备和可编辑安装。

## 3. 固定的 Isaac Lab 来源

官方仓库：<https://github.com/isaac-sim/IsaacLab>

发行版本：`v3.0.0`。

固定提交：

```text
e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4
```

该提交已从 GitHub 实际获取并验证。4 个源码组件来自同一个提交：

| 组件 | 源码目录 | distribution 版本 | `Requires-Dist` |
|---|---|---:|---|
| 核心接口 | `source/isaaclab` | `isaaclab==28.0.0` | 无 |
| Newton | `source/isaaclab_newton` | `isaaclab-newton==9.1.4` | `isaaclab` |
| 必需的配置类型 | `source/isaaclab_contrib` | `isaaclab-contrib==3.2.0` | `isaaclab` |
| 必需的配置类型 | `source/isaaclab_physx` | `isaaclab-physx==8.0.0` | `isaaclab` |

这些组件版本与发行版本 `3.0.0` 使用不同的编号。安装后的 `importlib.metadata.version("isaaclab")` 返回 `28.0.0`，对应本次固定的 v3.0.0 源码。

保留 contrib 和 physx 有实际调用依据。官方 `InteractiveScene` 在模块级导入 `VisuoTactileSensorCfg`，在创建场景实体时导入 `SurfaceGripperCfg`。这些配置入口在 Newton 环境中也会加载。保留 physx 配置包没有改变本项目的 Newton/MJWarp 仿真后端。

源码依据：

- [核心包元数据](https://github.com/isaac-sim/IsaacLab/blob/e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4/source/isaaclab/pyproject.toml)
- [Newton 包元数据](https://github.com/isaac-sim/IsaacLab/blob/e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4/source/isaaclab_newton/pyproject.toml)
- [场景中的配置导入](https://github.com/isaac-sim/IsaacLab/blob/e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4/source/isaaclab/isaaclab/scene/interactive_scene.py)

## 4. 为什么使用可编辑安装

官方发布的聚合 wheel 将全部 `isaaclab*` 模块合并，并把根项目的第三方基础依赖写入包元数据。`rsl-rl-lib`、`tensorboard`、`transformers` 都属于该聚合 wheel 的无条件依赖。官方 extras 没有提供只保留本任务组件的选择。[聚合 wheel 元数据生成逻辑](https://github.com/isaac-sim/IsaacLab/blob/e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4/tools/wheel_builder/gen_pyproject.py)

直接把 4 个子目录构建成普通 wheel 也无法启动本任务。独立环境实测在 `isaaclab/paths.py` 报错：`Could not locate the Isaac Lab root`。官方路径逻辑要求包内存在 `apps/`，或模块位于 `<root>/source/isaaclab/isaaclab`。普通核心 wheel 没有聚合 wheel 的资源布局。[官方路径解析实现](https://github.com/isaac-sim/IsaacLab/blob/e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4/source/isaaclab/isaaclab/paths.py)

本项目保留官方源码布局，并通过标准可编辑安装加载四个组件。没有修改上游路径解析，也没有添加伪造资源目录。`pyproject.toml` 中的四个 `tool.uv.sources` 都指向本仓库内的相对源码目录。

## 5. 实际运行依赖

第三方依赖来自完整 `FlightEnv` 启动及 10 步悬停的模块追踪。追踪期间禁止导入 `rsl_rl`、`tensorboard`、`transformers`、`stable_baselines3`、`rl_games` 和 `skrl`。运行成功，实际加载的 Isaac Lab 模块只有上述四个组件。

追踪得到以下 20 个第三方 distribution：

| distribution | 实测版本 |
|---|---|
| charset-normalizer | 3.3.2 |
| Farama-Notifications | 0.0.6 |
| filelock | 3.25.2 |
| glfw | 2.10.0 |
| gymnasium | 1.3.0 |
| lazy-loader | 0.5 |
| mujoco | 3.12.0 |
| mujoco-warp | 3.12.0 |
| newton | 1.6.1 |
| newton-usd-schemas | 0.5.0 |
| numpy | 2.5.1 |
| packaging | 26.0 |
| prettytable | 3.18.0 |
| rich | 14.3.4 |
| scipy | 1.17.0 |
| torch | 2.11.0+cu130 |
| tqdm | 4.70.0 |
| typing_extensions | 4.16.0 |
| usd-exchange | 3.0.1 |
| warp-lang | 1.17.0 |

NumPy、SciPy、Numba、PyTorch 在基本依赖中声明。Numba 0.68.1 / llvmlite 0.50.0 在 CPU 上编译 Python 数值核；源码仍保留一套 SUPER 目标、梯度和 L-BFGS。首次编译和缓存加载发生在规划进程初始化，逐次规划时间单独记录。其余仿真依赖在 `simulation` extra 中声明。结果可视化另需 Plotly，测试另需 pytest。传递依赖记录在 [uv.lock](../uv.lock)。

加入 NMPC 依赖后，锁文件记录 72 个包。安装环境的 70 个 distribution 已通过 `uv pip check`。锁文件包含项目、测试、可视化及平台条件依赖。

PyTorch 使用单独的 `pytorch-cu130` 索引。该索引设置 `explicit=true`，只绑定 `torch`，其他包继续使用 PyPI/NVIDIA 索引。实测中，将 cu130 索引作为所有包的首选曾导致 CUDA 依赖反复下载重试；限定索引后完成了安装。

这份依赖集合针对本仓库的无界面飞行、规划及结果导出。它不覆盖 Isaac Lab 的全部任务、训练和渲染功能。

## 6. 2026-11-14 安装验证记录

本节保留 SUPER 重构前的环境安装证据。当前算法验收使用 README 所列的 SUPER 测试和实验。

验证环境位于本仓库 `.runtime/dependency_probe/.venv`。第三方包独立安装，未向 `sys.path` 添加 Isaac-L2F 或既有 Isaac Lab 环境的 `site-packages`。

| 检查 | 实际结果 |
|---|---|
| 四个官方组件及运行依赖 | 59 个已安装 distribution |
| 包依赖一致性 | `uv pip check` 通过 |
| RSL-RL / TensorBoard / Transformers 安装数量 | 0 |
| medium 场景创建、Newton/MJWarp 启动 | 成功 |
| 悬停步数 | 10 个环境步，40 个动力学步，0.1 s 仿真时间 |
| 记录的位置漂移 | 0 m |
| 最大接触力 | 0 N |
| 末端竖直速度 | `6.340921743230865e-08 m/s` |
| 启动及短测试墙钟时间 | 39.63 s |

该 59 包记录属于加入项目、pytest 和 Plotly 前的运行依赖探测。它验证四个源码组件和必要第三方包可以启动实际 Newton 环境。完整飞行的到达率、碰撞率和跟踪误差使用仓库的飞行验收报告，不能由这次短悬停推断。

随后，主 `.venv` 已移除临时共享依赖路径，并用正式安装脚本安装 65 个 distribution。`uv pip check` 通过。该环境没有 RSL-RL、TensorBoard、Transformers、`isaaclab-rl` 或 `isaaclab-tasks`。

当时的回归结果为 `74 passed, 1 xfailed in 18.61s`。该记录对应已被替换的 GCOPTER 流水线，见[历史数值报告](history/numerical_validation.md)。

同一环境实际执行了上面的双图演示命令。medium 和 dense 的种子 0 均无碰撞到达。两图的时间、状态、参考、电机状态和 400 Hz 物理记录，与此前五种子试验中的同一种子逐元素完全一致。该安装验收单独记录，不重复计入五种子成功率。[命令、原始结果和数组对照](data/installed_demo_validation.json)

## 7. 打包范围

项目 wheel 只打包 `src/isaac_pathcover` 下的 Python 代码、动力学参数、两图几何资产及许可证。官方 Isaac Lab 源码通过安装脚本获取，不嵌入 wheel。

2026-11-14 的旧 wheel 核对记录保存在 [data/package_validation.json](data/package_validation.json)。该文件仅记录当时的安装结果。

源代码归档还包含安装脚本、锁文件、测试夹具、离线对照驱动和验证文档。

`tests/reference/super_mapping/`、`super_trajectory/`、`super_execution/` 的 C++ 驱动仅用于离线源码对照。它们不进入运行包，常规 pytest 读取已保存的输入输出。正式运行使用 Python、标准数值库和现有 Newton 官方依赖。
