# Asim 专家部署交付

2026-11-12最新交付：`asim-claude-guide-v1.tar.gz`（约42MB），旁附`.sha256`和`asim-claude-guide-v1-使用说明.md`。源码/六类及变体/affine/可信验证/技能和离线依赖，不含研究数据/结果/build/凭据；包内先读使用说明.md。已在新空格路径完成解压SHA、离线依赖与CMake配置、254测试（2可选跳过）＋8框架测试、九场景affine与六类真实Asim/重建回归。此轮复用字节绑定的封存binary，未再次冷编译；OpenCode仅接入指南尚未实测。

固定源码标签`asim-claude-guide-v1-20261112`，SOURCE_VERSION.json对应675cd233；手册在精简分支DOCS/CLAUDE_CODE_USER_GUIDE.md，验收在experiments/distribution/2026-11-12-claude-guide-v1/。下列旧包保留为历史版本，不覆盖。


交付包：[asim-expert-v1.tar.gz](asim-expert-v1.tar.gz)（35.67 MiB），[SHA-256](asim-expert-v1.tar.gz.sha256)。

校验值：`1583732bcc831e6bb6d9bcf61403eacb1af8662ac15a486d336550558fab2387`。

[使用说明](asim-expert-v1/README.md) · [Python→skill→算子](asim-expert-v1/docs/PYTHON_TO_OPERATOR.md) · [数值重建原理](asim-expert-v1/docs/RECONSTRUCTION.md) · [六类入口](asim-expert-v1/docs/OPERATORS.md) · [验收报告](../experiments/distribution/2026-11-10-v1/README.md)。

Linux x86_64／WSL2 Ubuntu源码部署包，不含研究数据/结果或构建缓存。实际Linux源码构建、六类×96batch Asim/重建、含空格目录迁移NS与完整性门禁已通过；250测试中248通过、2可选依赖跳过。WSL2/另一主机未实测。

解压后先按包内README准备系统编译工具和Python3.12，然后执行bootstrap.sh和smoke.py。打包构造工具/模板在packaging/跟踪；生成目录/压缩包不提交Git。
