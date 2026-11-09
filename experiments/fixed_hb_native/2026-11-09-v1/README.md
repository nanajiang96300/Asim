# 原生固定HB首次执行：中断记录

冻结契约见 [contract.json](contract.json)，源1754ea63、根封存62b72af1，SHA891e2b0a35e046008cef707878bcf65c784063350e86b22302aaa95cad7cd97c。C++/参数/独立参考与后续v2相同。

λ1/K16与λ.1/K53已通过真实Asim、保守事件检查和逐FP16位重建，周期18409/47121。λ.01/K174的Asim实际完成141017周期，schedule通过，独立references落盘，但进程收到SIGTERM（退出143），没有outputs/result及完整阶段index；该档和本次campaign不签发通过。

观察到大量换页，进程接近工具会话10分钟生命周期时结束；没有证据将信号归因于算法错误，也不将其确诊为OOM。详见原始 `results/fixed_hb_native/2026-11-09-v1/interruption.json`。旧原始文件不覆盖。尚未生成H59001/59002，因此没有消费新CDL留出。

后续 [v2](../2026-11-09-v2/README.md)单独冻结源码/契约：只在数值阶段前释放已检查的trace/events/statistics对象，使用分阶段前台受监督进程；不变更C++、参考、K、阈值或H种子。245单测重新通过。不得将v1中断后补写成一次无中断通过。
