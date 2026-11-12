# NN primitives v1：冻结语义与实现状态

2026-11-12, DESIGN then implementation. Isolated codex/tiny-nn-primitives-v1. No legacy operator rewrite. Reuse SOFTMAX(8)/GELU(22); new kernel_contract=nn-vector-v1, no new opaque opcode. Public C++ constructors under operations/nn emit actual source views and derive cost metadata together. Generic NNGraphModel handles topology, DMA and barriers, not model-specific numerics.

Softmax: complete M×N matrix, axis=-1, FP32 row max/sub/exp, ascending-column FP32 sum/div, one FP16 output writeback. compute_size=N*4 (row compute bytes), tile_m=M/tile_n=N, full allocation size=ceil(M*N*2/64). Versioned latency=(2*row_tree_iterations*add_tree_latency + row_vector_iterations*(add_latency+exp_latency+div_latency))*M. Old SOFTMAX branch unchanged. GELU: tanh approximation .5*x*(1+tanh(sqrt(2/pi)*(x+.044715*x^3))), explicit FP32 operations, FP16 writeback; compute_size=M*N*4; existing gelu_latency vector model. Unsupported axis/formula/profile rejected.

Memory: inputs occupy distinct contiguous SPAD regions; intermediates occupy distinct ACCUM regions, each allocated complete shape; 64B alignment. Last region 32KiB barrier scratch. All source/weight/bias/scale tensors real MOVIN, request64/base0; one complete output MOVOUT. Every LOAD/compute followed by explicit barrier. Shared 2D weights/bias loaded per sample for initial cold-DMA baseline. Bias broadcast row_rhs explicit, same col count; scale full matrix declared constant. Matrix multiply transpose expressed in actual view. Graph nodes are frozen task metadata, interpreted in C++ into actual instructions; replay only handles opcodes.

Prototype shapes M=16/32 N=32/64 for primitive cases; batch96,24cores. NN model: B96/L32/D32/F64, formulas in TINY_ATTENTION_MODEL_PLAN. Generic primitive tests and MLP/attention use identical units, no separate replay. Input roles/shape, nodes, outputs, numeric and timing policies bind header. Max shapes128, input/node count budgets checked before allocation; capacity checked by common replay/schedule before accepting.

Independent FP16 algorithm reference does not import numerical interpreter; FP64 task reference uses mathematical operations. Initial 1% task tolerance is a target requiring calibration before candidate acceptance, not inherited MMSE2%; exact FP16 model equality mandatory. Intermediate captures and wrong-axis/formula/cost/read-source/omitted-node negative gates required. No trained-task quality or hardware certification claimed. All raw runs new directory, no score until applicable acceptance; old six/affine regressions retained.

Independent audit found old Softmax only scales the tree term by row count; versioned full-row path fixes this without rewriting legacy costs. C++ unary full-view/graph preflight guards have been strengthened; unknown node attributes are rejected. Calibration seed90001 max FP64 errors Softmax0.0225%, GELU0.0316%, MLP0.0753%, Attention0.0401%; heldout seeds91001/2/3 are not calibration. Pure Agent thinking time null. These are calibration values; acceptance uses the independent frozen holdout below.

Capacity audit: historical quiet config supplies262144 to fields treated as KiB (Common.cc and KB macro); this is not256KiB. Do not change/relabel old experiments. New NN config ascend_910b_nn_v1.json explicitly sets256KiB total per SPAD/ACCUM, split into128KiB buffers. NN graph inputs/weights and sync occupySPAD, intermediates ACCUM; both banks prechecked on every core. Largest declared acceptance shape L32/D64/F128 uses105344 SPAD bytes incl sync and83968 ACCUM bytes, under128KiB. This is software configured-capacity validation, not board certification.

Cost finalization before any Asim acceptance: new stable Softmax explicitly uses FP32 division, so new row cost includes div_latency (not legacy reciprocal/mul estimate). GELU remains one opaque modeled Vector primitive with configuredgelu_latency; no fictitious tanh/microinstruction trace attribution. Old kernel branches unchanged.

## 2026-11-12 实际验收更新

已实现公共VectorUnits、NNGraphModel、统一重放与独立参考；suite-v2中4类任务×3形状×3种子共36组/3456batch通过真实Asim→事件→数值门禁，全输出与捕获阶段同FP16参考位一致。最终源码16c28f7a，二进制SHA-256 f8578ac30c4278511dcd7efd120429f2e27ab80dd31134f1bf06fbd607773d94。默认B96/L32/D32/F64的Softmax/GELU/MLP/Attention分别1534/346/1802/4975周期；最大FP64相对误差0.075541%，冻结门限1%。

真实错误K转置分支仍4975周期、事件通过，但所有96batch的logits开始偏离，重放拒绝；3个batch最终FP16输出偶合也不能隐藏阶段错误。详见最终[验收报告](../../experiments/tiny_nn/2026-11-12-v1/README.md)，其归档版本、最终复现记录优先于本节中间快照。

模型构造时设置确定的外部root producer；输入先注册Model再add_input；输出producer绑定真实Operation id。审计发现历史Model(json,config)未初始化root id的风险，当前仅修NN入口；旧家族回归通过不能据此证明历史所有模型均安全。

软件宏指令周期、配置容量和保守事件检查不代表910B上板认证。既有LanguageModel/ONNX入口不自动获得本次验收；首版支持有版本的受限图及上述四类任务，不宣称任意模型编译器。
