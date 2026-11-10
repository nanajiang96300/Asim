# 专家部署包

构建：bash scripts/bootstrap.sh；验证：.venv/bin/python scripts/smoke.py --operator all --output work/new-run。

采用离线源码依赖，不使用原服务器Conan路径。开发流程skill位于.claude/skills；具体操作应遵守AGENTS.md、V3规范与docs中的三路径验证。研究基线/失败证据须自行按版本保存，本包不包含原实验结果。
