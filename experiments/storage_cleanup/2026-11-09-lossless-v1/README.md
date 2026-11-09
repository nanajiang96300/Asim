# 旧trace无损整理与空间回收

2026-11-09。负责人明确授权清理无用数据，要求保留最新结果、关键中间数据及结论。执行前先冻结plan及SHA；未触及最新results/bri_implicit/2026-11-09-v1、QPSK全链路域、输入/输出/参考数组、日志、源码、二进制或当前82文件运行器。

29份较早的大型trace/指令/流水制品转为同路径后缀.gz（gzip level1）。每份先流式生成压缩副本，再完整解压并核对原大小/SHA，再记录持久台账，最后删除冗余未压缩副本；保留全部科学原字节。约41.31s，逻辑节省5,608,837,988B，可用空间从842,448,896B增至6,450,892,800B。不是删除5.61GB独有研究证据。

[plan.json](plan.json)列220个候选，本轮只执行其中29个达到6GiB空闲目标；未执行项继续原位保留。[execution.json](execution.json)是实际清理/归档映射，不能把候选清单当成全删除清单。[restore_test.json](restore_test.json)记录一份134,218,140B原件真实恢复、SHA核对及拒绝覆盖已有目的文件，测试临时恢复副本随后清理。

旧manifest保持原字节不回写；直接核对旧未压缩路径会拒绝缺文件，这不是数值失败。需要原件时显式恢复，不能关闭hash门禁：

```bash
.venv/bin/python scripts/research_compress.py restore \
  --ledger experiments/storage_cleanup/2026-11-09-lossless-v1/execution.json \
  --sha256 <execution.json的SHA256> --path <台账中的原路径>
# 可用 --destination <新目录中的文件> 先恢复到独立位置；目的父目录须存在。
```

恢复验证压缩文件SHA、展开字节数和原SHA；不足空间、未知原件或目标已存在均拒绝。恢复后再用对应历史Git/工具链版本执行原校验器。不修改旧报告或将已压缩制品伪称当前可直接读取。
