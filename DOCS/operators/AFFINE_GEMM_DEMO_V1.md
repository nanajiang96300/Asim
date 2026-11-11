# AFFINE GEMM DEMO V1 设计文档

## 1. 冻结公式与舍入
目标为独立批次的实矩阵 Z = A @ B + C，每个 batch b 独立计算：
Z[b, i, j] = half(half(sum_{k=0}^{K-1} A[b, i, k] * B[b, k, j]) + C[b, i, j])

冻结 Python 参考的舍入语义如下：
- A/B/C 为 FP16 存储；GEMM 内部把 A/B 元素转为 FP32 后逐 k 计算外积。
- 外积 A[:, k:k+1].astype(fp32) * B[k:k+1, :].astype(fp32) 在 FP32 中进行。
- acc（m x n）是 FP32，按 k=0,1,...,K-1 升序执行 acc = acc + product。每次 FP32 加法保留 FP32 结果。
- GEMM 完成后先把整个 FP32 acc 一次性 cast 为 FP16 得到 temporary，这个 GEMM 中间结果要保留为独立缓冲区，不能与 ADD 融合。
- ADD 阶段把 temporary cast 回 FP32，把 bias C cast 回 FP32，执行 FP32 加法，最后 cast 为 FP16 得到 Z。
因此指令级表达是：GEMM（FP16 A/B -> FP16 temporary，内部 FP32 累加），随后 ADD（FP16 temporary + FP16 C -> FP16 Z），两个算术指令不融合。

## 2. Python 到 Asim 指令映射
每个 batch b 作为一个 Tile，core_id = b % 24。
1. MOVIN A：input slot 100+0，shape[B,M,K]，batch 偏移 batch*M*K*2 字节。
2. MOVIN B：input slot 100+1，shape[B,K,N]，batch 偏移 batch*K*N*2 字节。
3. MOVIN C：input slot 100+2，shape[B,M,N]，batch 偏移 batch*M*N*2 字节。
4. GEMM_PRELOAD：src views A(M,K)、B(K,N) 不转置；dst 为 GemmTmpFp16(M,N)；compute_size=N，tile_m=M，tile_k=K，tile_n=N。该指令语义为 k 升序的乘加，累加保留为 FP32，输出舍入到 FP16。
5. ADD：src views GemmTmpFp16(M,N)、C(M,N)；dst 为 Z_batch(M,N)；compute_size=M*N*4，tile_m=M，tile_k=N，tile_n=N。输出舍入到 FP16。
6. MOVOUT Z：output slot 200，shape[B,M,N]，batch 偏移 batch*M*N*2 字节。
每个 MOVIN、GEMM_PRELOAD、ADD 后各插入一个 PIPE_BARRIER，保证依赖顺序。MOVOUT 是 tile 的 last_inst。

## 3. 地址分配与 DMA
所有 DMA 使用 base_addr=0、request_size=64、size=ceil(rows*cols*2/64)。
局部 SPAD 分配顺序（全部互不重叠，按元素数向上对齐到 64 字节）：
- A_batch: M*K*2 字节
- B_batch: K*N*2 字节
- C_batch: M*N*2 字节
- GemmTmpFp16: M*N*2 字节
- Z_batch: M*N*2 字节
- sync: 32768 字节，供 PIPE_BARRIER 使用
不重排或分块，每个 batch 完整加载 A/B/C，GEMM 完整输出 G，ADD 完整输出 Z，MOVOUT 完整写出 Z。没有外部预计算或隐式初始化。

## 4. 模型与头部配置
Model 配置项：matrix_m、matrix_k、matrix_n、batch_size、task_contract、precision_policy。
默认 M=K=N=32、batch_size=96。合法条件：precision=2（FP16 存储）、dram_req_size=64、num_cores=24、batch 为正且 24 的倍数、M/K/N 为 16 的正倍数且 <=64；不合法时抛出异常。
Operation 绑定 numeric header：
- schema = source-views-program-v1
- profile = modeled_fp16_fp32_v1
- precision_policy 与 task_contract 原样取自 Model 配置
- algorithm 精确为 {"id":"affine_gemm_v1","formula":"Z=half(half(A@B)+C)","gemm_order":"k_ascending_multiply_then_add"}
实际 Instruction 通过 source_instruction 附上 typed views；没有另写公式日志。FormulaLogger 仅用于诊断 GEMM/ADD 步骤，不承担验证。

## 5. 未执行的验收范围
本文件由候选生成阶段交付，未执行构建、Asim 仿真、数值对标或评分，不声称任何测试通过。
以下事项由主协调器/审计/构建/评估框架执行，不在本候选内：
- 编译、链接、注册和候选路径写入
- 与冻结 Python 参考的逐位/容差数值检查
- Asim 调度、功耗/性能评分以及 source-views 审计
- 信道/MMSE 门禁：本任务是 AFFINE GEMM 演示，不是 MMSE，不声称通过信道或 MMSE 门禁
- 资源预算的最终审计，包括 SPAD 容量和 PIPE_BARRIER 空间

## 6. 边界说明
本实现没有迭代矩阵求逆和预处理；k 维的升序乘加完全体现在单一 GEMM_PRELOAD 指令的 gemm_order 中。
