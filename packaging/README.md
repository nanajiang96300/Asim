# 专家部署包生成工具

交付目录固定为`expert_delivery/asim-expert-v1/`，交付压缩包为同目录旁的`.tar.gz`，生成物不提交Git。源码/脚本/适配模板在本目录跟踪，验证摘要固定在`experiments/distribution/2026-11-10-v1/`。

当前assembler是原工程服务器上的显式依赖收集工具，需要代码中声明的本机第三方源码缓存及匹配ONNX生成源码；这些采集路径不会成为交付包的安装或运行依赖。它不是用于目标企业电脑上的安装脚本。目标机只运行交付包的bootstrap。

重建交付包（目标目录必须不存在，旧版本先另存；wheel目录位于目标目录外）：

```bash
.venv/bin/python packaging/assemble.py --wheel-dir /path/to/pinned-wheels
bash expert_delivery/asim-expert-v1/scripts/bootstrap.sh
# 在包内执行单元测试和scripts/smoke.py，先把摘要复制到包外
python3 packaging/finalize.py --seal
python3 expert_delivery/asim-expert-v1/scripts/check_package.py
tar -czf expert_delivery/asim-expert-v1.tar.gz -C expert_delivery asim-expert-v1
sha256sum expert_delivery/asim-expert-v1.tar.gz > expert_delivery/asim-expert-v1.tar.gz.sha256
```

wheel为官方pip26.1.2通用wheel、NumPy2.5.0 CPython3.12 manylinux x86_64 wheel。安装离线，不将本机Conan缓存、Git元数据、账号、虚拟环境、二进制/build缓存、研究输入/结果带入包。`finalize --seal`仅删除本工具生成包内的build/.venv/validation-work/relocation-build和Python缓存，并生成文件清单；不得对原工程results执行这一清理。

打包适配范围：独立CMake，源代码形式的依赖/ONNX protobuf，Ramulator依赖配置，历史contract夹具转包内小型测试fixture，CDL可选依赖测试显式跳过，删除旧Timeloop演示输出。算子src必须逐文件SHA一致；验证器运行核心保持原样。
