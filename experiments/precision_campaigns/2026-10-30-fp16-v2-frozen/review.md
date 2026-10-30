# 精度与调度基础审查

Agent legacy_semantic_review只读检查后，数值模型限定范围通过。实现补齐输入量化边界、逐FP16位模式门禁、signed zero、逐指令写回、非有限归档、GEMM内维检查。P2存储/P4 Vector计费一致，独立参考不导入engine。未校准转换成本不出性能分。

FW4合成检查按字节范围与core/buffer识别RAW/WAR/WAW，缺真实或未绑定读完成/写可见事件时UNSUPPORTED；合成模型不冒充G4。审查建议的partial覆盖/跨core-buffer及未绑定文件回归已补。主Agent最终执行103项单元测试；Agent没有重跑。
