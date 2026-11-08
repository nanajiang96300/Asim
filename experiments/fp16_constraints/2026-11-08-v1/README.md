# FP16约束下的误差拆分与路线修正

状态：REFERENCE_DIAGNOSTIC_COMPLETE。重用上一轮192个CDL-B的冻结输入，做低成本误差来源诊断，不运行新Asim或候选，不声称FP16算子通过。代码：verification/studies/fp16_constraints.py；代码提交6ad8d786。

20dB下，目标逆矩阵仅存FP16的最大相对逆误差0.0243%；H/λ量化为FP16后由FP64参考解算，逆误差0.0536%、MMSE输出误差0.1571%；显式FP16 Gram参考逆误差4.9739%。因此4.97%不能作为所有FP16算法的不可突破下界。所有FP64 solve均只作参考诊断，不是候选，也未计成免费硬件能力。

[result.json](result.json)绑定原result、每份CIR输入、payload与源码SHA、Python/NumPy版本，candidate_verified=false、score=null。原件仍在results/bri_principles/2026-11-08-v4/；本轮独立结果在results/fp16_constraints/2026-11-08-v1/。不修改旧归档、算子、参考或固定运行器。

当前设计与具体任务顺序：[FP16_FIRST_PLAN](../../../DOCS/research/FP16_FIRST_PLAN.md)。混合精度从默认下一步改为后备扩展，优先验证现有精度下的正确公式、包谱界、块布局、隐式Gram与显式缩放；是否达标仍待候选实现和留出验收。

复现：使用result.json中source_result.sha256作为批准摘要，新输出目录：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m verification.studies.fp16_constraints \
  --study results/bri_principles/2026-11-08-v4/result.json \
  --study-sha256 <result.json中source_result.sha256> \
  --output results/fp16_constraints/<new-id>
```

这份诊断程序不启动原语候选，不构建、不评估周期；构造新候选仍按Asim先行、实际指令重建三路径验收。
