# Asim 专家部署交付

2026-11-12最新NN源码交付：[asim-claude-guide-v2.tar.gz](asim-claude-guide-v2.tar.gz)（43,778,024字节），[SHA-256](asim-claude-guide-v2.tar.gz.sha256)、[独立使用手册](asim-claude-guide-v2-使用说明.md)、[交付复现回执](asim-claude-guide-v2-validation.json)。源码标签asim-claude-nn-v1-20261112，source commit c84c3ce0；SHA `a5d819f80937f2c9789aab71fd8e720154a8666f9232af2c68870ff05cda4ebb`。

含原六类、affine、新公共Softmax/GELU及MLP/Attention；无研究数据/结果/build/Git历史/账号。新解压目录标准离线冷构建27.67分钟，264项中262通过/2可选跳过、8便携测试、模型36场、六类与affine九场通过；周期/输出SHA与开发验收一致，另新建本地pin并快检通过。只声明同Linux主机新路径软件复现，WSL2/第二主机/训练与硬件认证未签发。论文固定入口[本轮快照](../DOCS/research/paper_archive/2026-11-12-tiny-nn-v1/README.md)。

旧版压缩包保持不变；旧asim-expert-v1展开目录及本次v2验证目录按SHA核对后回收，所以下文旧展开目录链接需要先从对应tar.gz恢复。只读说明可直接在压缩包内查看，不静默改写旧制品清单。


2026-11-12最新交付：`asim-claude-guide-v1.tar.gz`（约42MB），旁附`.sha256`和`asim-claude-guide-v1-使用说明.md`。源码/六类及变体/affine/可信验证/技能和离线依赖，不含研究数据/结果/build/凭据；包内先读使用说明.md。已在新空格路径完成解压SHA、离线依赖与CMake配置、254测试（2可选跳过）＋8框架测试、九场景affine与六类真实Asim/重建回归。此轮复用字节绑定的封存binary，未再次冷编译；OpenCode仅接入指南尚未实测。

固定源码标签`asim-claude-guide-v1-20261112`，SOURCE_VERSION.json对应675cd233；手册在精简分支DOCS/CLAUDE_CODE_USER_GUIDE.md，验收在experiments/distribution/2026-11-12-claude-guide-v1/。下列旧包保留为历史版本，不覆盖。


交付包：[asim-expert-v1.tar.gz](asim-expert-v1.tar.gz)（35.67 MiB），[SHA-256](asim-expert-v1.tar.gz.sha256)。

校验值：`1583732bcc831e6bb6d9bcf61403eacb1af8662ac15a486d336550558fab2387`。

[使用说明](asim-expert-v1/README.md) · [Python→skill→算子](asim-expert-v1/docs/PYTHON_TO_OPERATOR.md) · [数值重建原理](asim-expert-v1/docs/RECONSTRUCTION.md) · [六类入口](asim-expert-v1/docs/OPERATORS.md) · [验收报告](../experiments/distribution/2026-11-10-v1/README.md)。

Linux x86_64／WSL2 Ubuntu源码部署包，不含研究数据/结果或构建缓存。实际Linux源码构建、六类×96batch Asim/重建、含空格目录迁移NS与完整性门禁已通过；250测试中248通过、2可选依赖跳过。WSL2/另一主机未实测。

解压后先按包内README准备系统编译工具和Python3.12，然后执行bootstrap.sh和smoke.py。打包构造工具/模板在packaging/跟踪；生成目录/压缩包不提交Git。
