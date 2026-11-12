# 便携源码构建与双客户端部署

目标：Linux x86_64 / WSL2 Ubuntu，建议Ubuntu24.04、GCC13、CMake3.28、CPython3.12。真实NPU/GPU驱动不是周期模拟器部署前置。先准备系统编译工具（build-essential、cmake、python3、python3-dev；venv使用--without-pip，pip wheel离线提供）。默认-j1，2-vCPU首次冷编译可能需数十分钟；建议至少4GiB空闲磁盘和8GiB内存。WSL2未在本轮实机测试。

## 首次联网部署

```bash
git clone --depth 1 --single-branch --branch codex/asim-claude-v1 https://github.com/nanajiang96300/Asim.git
cd Asim
python3 scripts/bootstrap.py --jobs 1
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/test_portable.py
.venv/bin/python -m unittest discover -s tests -p test_portable_bootstrap.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/portable_smoke.py --operator all --output results/first-six
python3 scripts/agent_entry.py doctor
claude
```

Codex环境改用codex/asim-codex-v1，后续构建/测试完全相同。API、模型和账户自行在用户环境配置，不复制别人的settings.local.json或认证。doctor仅查CLI，不发送API请求。S1已完成两个全新clone的冷构建及六类真实Asim/重建验收，见experiments/deployment/2026-11-11-s1-v1/README.md；WSL2仍待实测。

bootstrap下载dependencies/lock.json中固定commit/大小/SHA的三个制品，解包后逐文件校验，再创建本clone的.venv、编译本clone的build。无Conan、无服务器绝对路径、无CMake FetchContent、无全局pip安装。安装依赖阶段可以联网；configure/build及--offline模式不自动下载。不会改变旧runtime pin。源码依赖目录.deps、build与.venv不提交Git。

## 企业离线部署

在联网机器的同版本检出中：

```bash
python3 scripts/bootstrap.py --download-only --cache /path/to/media/asim-cache
```

将三个文件及Git源码检出送入企业机器（源码可使用git bundle或企业Git镜像；单带缓存不包含项目源码），然后：

```bash
python3 scripts/bootstrap.py --cache /path/to/media/asim-cache --offline --jobs 1
```

缓存内容必须与当前锁相同，缺失/大小或SHA不同直接拒绝，不静默回退或覆盖。依赖锁升级必须重新生成缓存。wheel只支持CPython3.12/Linux x86_64，不将换版本解释为已验证。

## 独立构建与门禁

`--deps-dir`、`--build-dir`支持独立路径；便捷smoke使用默认build；非默认build可通过agent_entry.py evaluate显式传binary/config/cwd/output运行原runner。同一可写目录不能并发安装/构建。预编译公共头ASIM_USE_PCH默认ON，仅减少主机编译时间；手动cmake配置可关闭，不表示NPU指令优化。

smoke先真实Asim，再指令/真实Core事件检查，再独立同算法与实际重放、FP64 MMSE；默认六类各batch96、Rayleigh seed42、Nr64/Nt16/RHS8、λ.1、FP16存储/FP32临时模型、K=16(NS)/192(显式BRI)。输出必须为新目录，默认逐例删本次原始大制品，仅保留带删除前SHA的摘要；需要原件用--keep-raw并预留磁盘。不是新增CDL/通信/硬件或性能晋级。

完整原Python测试由test_portable调用；缺少可选Torch/Sionna时CDL生成测试明确SKIP，SciPy缺失的独立区间测试也明确SKIP。默认依赖不包含这些可选研究工具。bootstrap额外4项负例检查缓存/锁、安装文件、危险归档和离线缺失。不宣称旧Simulator_test/GTest目标已迁移。

## 诊断与证据

直接运行cmake前应已安装且校验.deps；否则明确拒绝。改动依赖文件后再次配置会检查SHA。安装拒绝后提供正确制品或新目录，不用降门限/随机补文件。binary的运行库用相对$ORIGIN/../lib定位，允许整体移动build目录；源码/依赖/构建路径保存在CMakeCache中，移动源码后重新配置，不能借运行目录可搬迁推断缓存可搬迁。

每次实验记录源commit、依赖锁/制品SHA、binary/库SHA、工具链、完整任务/算法参数、事件/数值门禁及周期。旧daily pin是其历史环境证据，新平台应重新冻结运行时，不修改旧清单以绕过校验。协调器/候选生成闭环按DOCS/research/AGENT_WORKFLOW.md的S2继续开发，持久无人值守后置。
