# 旧诊断无损压缩 v4

2026-11-10，用户授权保留最新结果/关键中间数据及结论后清理空间。本轮沿用受保护runtime排除与六个旧根白名单，先冻结plan，再逐文件gzip、完整解压SHA校验、写台账后删除冗余未压缩副本。实际64个文件，释放926970294字节，见execution.json。原始输入/数值数组/最新BRI基线与候选/凭据及用户未跟踪资料不选择。

历史manifest不重写。复现旧trace需用scripts/research_compress.py restore，指定execution.json及其SHA和台账原路径；不静默将.gz当旧路径通过。
