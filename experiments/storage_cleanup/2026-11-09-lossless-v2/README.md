# 第二轮无损空间回收

用户2026-11-09授权保留最新结果、关键中间数据和结论后清理无用数据。第一轮释放5,608,837,988字节；新高K原生HB检查需要额外空间，本轮沿用同一已审计脚本、同一六个旧结果根和受保护最新/运行时排除规则。先冻结plan（commit c0168eab），再逐文件压缩、完整解压SHA-256校验、写台账后删除冗余未压缩副本，约25.61秒。

实际文件数与3,467,128,578字节回收见 [execution.json](execution.json)。两轮合计约9.08GB（十进制），未丢弃原始字节；最新BRI、QPSK、数值数组、日志、源码、二进制、输入及用户未跟踪资料均不删除。新固定HB失败v1也完整保留。

原历史manifest不覆盖，直接访问退役未压缩路径会拒绝；需要复现时显式restore。本轮真实恢复试验见 [restore_test.json](restore_test.json)。命令：`python scripts/research_compress.py restore --ledger <execution.json> --sha256 <台账SHA> --path <原始路径>`；先核对空间。不得将.gz当成原路径静默通过旧验收。
