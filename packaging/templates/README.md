# Asim 专家源码部署包 v1

本包提供CPU上的Asim周期模拟器、六类矩阵求逆/MMSE算子、通用数值重建及独立算法参考，以及Agent算子开发skill。无历史输入、结果、模型权重、账号、凭据、Git历史或服务器编译缓存。源码还保留Merge和HB等变体；默认验收六类入口见 docs/OPERATORS.md。

## 部署

目标：Linux x86_64，Windows可使用WSL2 Ubuntu。实际验证环境为Ubuntu24.04、GCC13、CMake3.28、Python3.12；WSL2未在本轮实机运行。建议8GB内存、4GB空闲磁盘，低内存机器串行构建。模拟24核NPU不要求主机24核，不需要昇腾驱动。

系统准备（需企业管理员允许的系统软件源）：

```bash
sudo apt-get install build-essential cmake python3 python3-dev python3-venv
python3 scripts/check_package.py
bash scripts/bootstrap.sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/smoke.py --operator ns --output work/ns-001
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/smoke.py --operator all --output work/six-001
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s verification/tests
```

C++依赖源码和Python3.12 x86_64的NumPy/pip wheel已随包提供，bootstrap不需要Conan或联网下载。默认单线程编译；可通过ASIM_BUILD_JOBS调整。其他Python版本/架构需要替换匹配wheel，不宣称已验证。不要从Windows原生Python运行Linux二进制。

每次output必须为新目录。smoke串行运行真实Asim→事件/调度检查→实际指令数值重放→同算法FP16位一致→FP64 MMSE误差门禁，batch96、Nr64/Nt16/RHS8、Rayleigh seed42、λ0.1。默认只保留汇总，逐例删除本次生成的原始数据；要研究中间过程加`--keep-raw`。汇总中的原始文件SHA指删除前的文件，不是包内存在的证据。单个样本集通过不代表全信道认证或性能晋级。

单元测试中SciPy独立置信区间检查、Torch/Sionna CDL-B生成测试在未安装可选依赖时明确SKIP。本包默认测试不依赖CDL-B，不将Rayleigh通过称CDL通过。完整通信验收和硬件上板独立进行。

## 专家关注点

- [Python算法经skill形成算子](docs/PYTHON_TO_OPERATOR.md)：skill约束Agent完成设计、映射、构建与审计，不是自动转译器。
- [算子数值重建的原理与边界](docs/RECONSTRUCTION.md)：真实指令重放、独立同算法精度参考和FP64任务参考三条路径。
- [六类入口、参数与代码](docs/OPERATORS.md)。原始规范见DOCS/OPERATOR_DEVELOPMENT_STANDARD_V3.md；skill位于.claude/skills/。

本包提供可运行的确定性构建/重建入口，不含跨天无人值守协调器。原样保留的skill可能提及历史Claude/UOBS脚本、旧registry或服务器封存结果；这些旧实验驱动/数据未随包提供，部署时使用本包bootstrap/smoke入口，不能将旧引用当作本包已实现的功能。

使用Claude/Codex等Agent时自行配置企业允许的工具和模型账户。不能只声明调用了不可用的slash command，应执行skill对应步骤并保留证据。

## 可追溯性

SOURCE_PROVENANCE.json记录源版本和打包适配；PACKING_MANIFEST.json记录源码包文件SHA-256，scripts/check_package.py检查完整性。打包仅适配CMake、依赖装配和独立测试夹具/可选依赖跳过，不修改算子算法。第三方许可证见THIRD_PARTY_NOTICES.md及各vendor/extern目录。本项目自身授权请向项目负责人确认。

本轮部署测试报告在原工程experiments/distribution/2026-11-10-v1/，不随本包携带实验结果。安装、构建和运行会在新机器生成build/.venv/work，这些目录不属于源码交付清单。
