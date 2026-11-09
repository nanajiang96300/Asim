from pathlib import Path
import json
repo=Path('/home/nanajiang/Asim');exp=repo/'experiments/fixed_hb_native/2026-11-09-v2';s=json.loads((exp/'summary.json').read_text());assert s['status']=='FIXED_HB_CDLB_RECONSTRUCTION_CHECKED'
rows=[]
for r in s['cases'][:3]:
 q=next(g for g in s['cdlb'] if g['regularization']==r['regularization'])
 frames=[f for f in s['frames'] if f['regularization']==r['regularization']]
 rows.append(f"| {r['regularization']} | {r['iterations']} | {r['cycles']:,} | {100*q['max_task_error']:.6f}% | {sum(f['phase_timings_seconds']['independent_references'] for f in frames):.2f} | {sum(f['phase_timings_seconds']['instruction_replay_and_comparison'] for f in frames):.2f} |")
readme='''# 固定HB原生接入、实际重建与新CDL-B验收

状态：FIXED_HB_CDLB_RECONSTRUCTION_CHECKED，2026-11-09。新入口`bri_stable_hb_fp16_v1`在既定软件数值模型与输入域内通过，不替代旧β-only候选，也不是性能晋级或硬件认证。

## 先冻结算法，再验收实际指令

[算法筛选](../../fixed_hb/2026-11-09-v1/README.md)固定无oracle谱界、α/β联动和K16/53/174；校准H58001，算法留出H58002/3。旧失败H56002/b95只回归，新误差0.5261%，不参与选K。

原生设计提交3a6e293f先于C++40bcbf0e，封存1754ea63。首次[v1](../2026-11-09-v1/README.md)第三档数值进程中断未签发通过；本v2冻结bd1e520b，唯一评估器改动是在事件门禁后释放不再使用的大型诊断对象，算法/参考/门限/K不变。契约SHA42242c3e0e52a93023b12163332a265f10b436a8acefffe337d3d9393a30f959，根冻结commit c0168eab；本轮分阶段前台PTY运行，不宣称持久控制器已完成。

2vCPU/3.8GiBRAM+swap，串行构建/仿真。独立build复制旧缓存，准备时校验141份共享C++字节相同与382个对象SHA，删除旧Simulator并重新编译Asim本体/新入口，第三方依赖复用；不是冷启动干净构建。[缓存回执](build_cache_receipt.json)和原始configure/build日志保留。binary源码版本1754ea63的C++与bd1e520b逐字相同，后者只有Python内存释放和example。

## 验证结果

Nr64/Nt16/RHS8、batch96、24核、Cube16³；H/Y/λ→MMSE直接输出。FP16张量/系数/所有迭代状态，沿用FP32临时算术、固定k序GEMM、RNE/次正规保留；不是纯FP16 MAC，不新增混合精度指令。点预条件与系数归约、初始化、补偿、状态复制及输出恢复均在真实Instruction与周期内。

先完成三档真实Asim瑞利模板（共288batch）、trace/实际Core事件/代价绑定/逐指令重建，再生成此前未使用的Sionna CDL-B静态频域H59001/59002各96，载荷59101/59102。对每档实际指令原序列执行192batch，共576batch；与独立同算法FP16实现完整输出逐位一致，再与FP64 MMSE比较；没有未知语义/随机填补/参考代填/跳过操作。

| λ | 冻结K | Asim周期 | 新CDL最大任务误差 | 新CDL独立参考秒（192batch） | 新CDL实际重建秒（192batch） |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

三SNR相对BER增加≤.01、SER增加≤.02、原始EVM增加≤.01，按独立H聚合与多组置信门禁均通过。误差门限2%未改变，全部无失败。相同192个H复用三SNR，不能宣称576独立H；每H8个RHS。完整候选/参考BER、误差与置信区间见[summary.json](summary.json)。Asim为数据独立模板计时，CDL重放不产生新的周期。此处仅认证声明静态几何/输入域/三SNR，不覆盖任意信道、时变OFDM、FEC、总体SE/BLER或实机。

## 周期与详细流水

[cycles_and_replay.csv](cycles_and_replay.csv)包含同任务周期、统一comparison_identity和Asim/事件/参考/重建分阶段墙钟；[pipeline_summary.csv](pipeline_summary.csv)给各流水指令数、活动区间/重叠/空闲等，[core_pipeline_summary.csv](core_pipeline_summary.csv)给24核分项。逐指令流水CSV、完整trace/events在原始结果目录，不混入旧周期表。

同新binary/硬件配置/seed57001/λ.1的NS K16与旧隐式K33另行重跑，通过既有瑞利数值范围，comparison_identity与新HB K53一致；这只建立局部成本对照，旧入口未在本新CDL集合验收，不能按此表宣布性能胜者。新HB保守谱界与更多K增加成本，下一轮需要先守住质量再降周期。score=null；冷启动跨环境复现与完整晋级尚未完成。

## 可信度与泛式

[controls.json](controls.json)记录四种离线真实指令变异拒绝、正/负零RHS逐位一致、同输入真实Asim重复五类制品字节一致、真实性能拒绝后没有数值制品，以及错误SHA在Asim前拒绝。离线变异不冒充故障二进制。[静态审计](static_audit.md)、[动态审计](dynamic_audit.md)明确独立抽检范围；245单测通过，未把参考自一致当真实事件认证。

新入口只新增C++ typed views、可信参数adapter和独立同算法reference，复用engine/source_views/binding/runtime/QPSK，无求解器名字分支。首末系数/谱界/P只读见证也验证布局未被迭代覆盖。域外或未知操作明确拒绝；其他形状/精度不自动继承本次验收。

## 归档与复现

完整原件`results/fixed_hb_native/2026-11-09-v2/`，[run_manifest.json](run_manifest.json)给路径/大小/SHA。命令见[run_stages.sh](tools/run_stages.sh)，使用冻结契约、新输出目录，阶段为asim→dataset→replay，随后controls/regression。当前日常固定runtime见orchestrator/codex_runtime_stable_hb_fp16_v1.json，白名单不代表每入口已通过CDL。纯Agent思考时间null，实际工具阶段及独立审查墙钟分别记录。

论文固定入口[paper_archive](../../../DOCS/research/paper_archive/README.md)。两轮旧制品无损回收9,075,966,566字节，78份原始trace完整保留gzip与恢复SHA台账；最新结果、旧失败、输入/关键数值和结论未删。[回收台账](../../storage_cleanup/2026-11-09-lossless-v2/README.md)。下一优先任务是通用重建器流式读取/及时释放大对象，随后新校准研究更紧谱界和指令复用；禁止在H590上继续选K。
'''
(exp/'README.md').write_text(readme)
intro='2026-11-09最新：固定HB隐式FP16新入口已完成三档真实Asim→实际指令重建→另192新CDL-B/576batch与完整未编码QPSK软件门禁；K16/53/174，逐FP16位一致，2%任务门限及相对质量门禁全部通过。保守谱界/联动系数修复旧β-only稳定域问题；旧失败原件不升级。周期18,409/47,121/141,017，未宣称性能最优、硬件认证或完整晋级。详见[原生报告](../../experiments/fixed_hb_native/2026-11-09-v2/README.md)、[选K与误差图](../../experiments/fixed_hb/2026-11-09-v1/README.md)。\n\n'
p=repo/'DOCS/research/README.md';txt=p.read_text();txt=txt.replace('\n\n','\n\n'+intro,1);p.write_text(txt)
notes={
 'FP16_FIRST_PLAN.md':'F16-5固定HB原生阶段已通过声明域软件门禁；不新增混合精度指令，K16/53/174。下一步先做通用日志/内存优化，再在新校准集研究更紧谱界和内循环降周期。',
 'ROADMAP.md':'F16-5完成独立算法选K、新留出、固定HB原生映射与三SNR新CDL-B/QPSK软件验收；旧β-only失败保持。依赖任务：流式指令/流水统计以降低高K内存→新校准更紧谱界/复用/流水→相同质量与输入域成本比较→冷启动复现/完整晋级。不得在H590调参。',
 'BRI_OPTIMIZATION_PLAN.md':'固定HB在无oracle谱界与原有FP16原语上完成新Asim入口和实际重建/CDL-B验收，替代上一阶段“仅设计”的后续状态，不升级旧入口。保守界导致K174和141017周期，后续优化需保持硬质量门禁。',
 'BRI_PRINCIPLES.md':'固定系数HB联动α/β与可实施保守谱界已通过新原生软件阶段；解析证明仅保证底层定系数线性系统，实际有限精度仍用独立三路径与新CDL/QPSK检验。旧变系数β-only反例仍有效。',
 'VERIFICATION_DESIGN.md':'固定HB再次验证通用重建范式：新增typed实际指令/adapter/独立reference而不改engine。288真实瑞利batch和576新CDLbatch逐位一致；高K首次进程中断保留，新版只释放已验证诊断对象后独立重跑，不把中断补写成通过。',
 'QUALITY_DRIVEN_WORKFLOW.md':'固定HB先独立校准选后缀K16/53/174，再算法新留出，冻结原生候选，Asim先行，随后全新H590 CDL重建和QPSK门禁通过。门限不变，score仍null，未做硬件认证/完整性能晋级。',
 'RUNTIME_STABILITY.md':'新固定runtime为orchestrator/codex_runtime_stable_hb_fp16_v1.json；旧implicit_v1固定清单不回写。高K事件/流水诊断造成显著内存压力，门禁后释放trace/stats/events；通用流式解析仍待做。默认日常入口仍瑞利。',
}
for name,note in notes.items():
 p=repo/'DOCS/research'/name
 if p.exists():
  with p.open('a') as f:f.write('\n\n2026-11-09固定HB更新：'+note+' 证据：[固定HB原生报告](../../experiments/fixed_hb_native/2026-11-09-v2/README.md)。\n')
p=repo/'DOCS/research/F16_STABLE_HB_DESIGN.md';txt=p.read_text().replace('状态：设计/待数值校准；','状态：独立校准与原生Asim/CDL-B软件验收完成，证据见experiments/fixed_hb_native/2026-11-09-v2/README.md；');p.write_text(txt)
p=repo/'AGENTS.md';p.write_text(p.read_text()+'\n固定HB隐式新入口bri_stable_hb_fp16_v1已在声明FP16存储/FP32临时/保留次正规模型下完成K16/53/174与三SNR、192新独立CDL-H/576batch实际重建及未编码QPSK门禁；见experiments/fixed_hb_native/2026-11-09-v2/README.md。旧β-only失败不升级；新HB非性能晋级/硬件认证，冷启动跨环境复现未完成。日常新pin codex_runtime_stable_hb_fp16_v1.json；旧82文件pin保留，当前live pin以新清单为准。下一步先解决高K通用重建内存，再新校准研究更紧谱界/指令优化，不在H590留出调K。两轮78旧trace无损gzip回收9.08GB，按storage_cleanup台账显式restore，禁止静默改旧manifest。\n')
p=repo/'DOCS/operators/bri_stable_hb_fp16_v1.md';p.write_text(p.read_text()+'\n\n验收更新2026-11-09：本新入口三档实际Asim/共同重建及新CDL-B/QPSK软件门禁通过；K16/53/174、周期18409/47121/141017，域与精度不扩张。首次v1中断保留，v2仅诊断对象生命周期修复。完整证据见experiments/fixed_hb_native/2026-11-09-v2/README.md；未冷启动跨环境/性能晋级/硬件认证。\n')
print('current docs updated from checked results')
