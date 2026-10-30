"""Generate an evidence-backed paper report. Final reports require every gate.

Current version intentionally emits an INTERIM report: incomplete runtime evidence
cannot be replaced by synthetic tests or a manually supplied PASS flag.
"""
import argparse,csv,hashlib,json,os,subprocess
from pathlib import Path
from verification.contracts.core import Invalid

REQUIRED=('actual_instruction_replay','independent_algorithm','high_precision_task','declared_precision_model',
          'hardware_precision_cost_calibration','actual_async_G4','formal_communication','CDL_B','all_legacy_operators','clean_reproduction')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_verified(manifest):
    manifest=Path(manifest);m=json.loads(manifest.read_text());repo=Path(__file__).resolve().parents[2]
    for key in ('artifacts','supporting_artifacts','small_artifacts'):
        for e in m[key]:
            p=repo/e['path']
            if p.stat().st_size!=e['bytes'] or sha(p)!=e['sha256']:raise Invalid('changed report evidence: '+str(p))
    return m


def readiness(structure,precision,legacy):
    structural=structure.get('status')=='GATES_PASSED' and len([c for c in structure['cases'] if c.get('actual')=='STRUCTURE_CHECKED' and c.get('passed')])>=3
    modeled=precision.get('status')=='GATES_PASSED' and len([c for c in precision['cases'] if c.get('status')=='MODELED_NUMERIC_CHECKED' and c.get('bit_equal_batches')==96 and c.get('passed')])==3
    states={k:'PENDING' for k in REQUIRED}
    states.update(actual_instruction_replay='PASS' if structural else 'FAILED',independent_algorithm='PASS' if structural and modeled else 'FAILED',
        high_precision_task='PASS' if structural else 'FAILED',declared_precision_model='PASS' if modeled else 'FAILED',
        actual_async_G4='UNSUPPORTED_MISSING_RUNTIME_RW_EVENTS',CDL_B='UNSUPPORTED',
        all_legacy_operators='PASS' if legacy.get('all_numerics_passed') else 'FAILED_OR_UNSUPPORTED')
    return {'schema':'paper-readiness-v1','status':'FINAL_READY' if all(s=='PASS' for s in states.values()) else 'INTERIM',
            'final_eligible':all(s=='PASS' for s in states.values()),'gates':states,'score':None}


def require_final(state):
    if set(state.get('gates',{}))!=set(REQUIRED) or not all(s=='PASS' for s in state['gates'].values()):
        raise Invalid('FINAL_REPORT_BLOCKED: '+','.join(k for k in REQUIRED if state.get('gates',{}).get(k)!='PASS'))


def generate(structure_manifest,precision_manifest,legacy_manifest,output,final=False):
    out=Path(output).resolve()
    if out.exists():raise Invalid('report snapshot must use new directory')
    sources=[Path(p).resolve() for p in (structure_manifest,precision_manifest,legacy_manifest)]
    manifests=[load_verified(p) for p in sources]
    summaries=[json.loads((Path(m['raw_root'])/'summary.json').read_text()) for m in manifests]
    s,p,l=summaries;state=readiness(s,p,l)
    if final:require_final(state)
    out.mkdir(parents=True)
    evidence=[{'manifest':str(path),'manifest_sha256':sha(path),'raw_root':m['raw_root'],'runner_commit':m['runner_commit'],'binary_source_commit':m['binary_source_commit']} for path,m in zip(sources,manifests)]
    repo=Path(__file__).resolve().parents[2];generator_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    (out/'readiness.json').write_text(json.dumps(state,indent=2))
    (out/'evidence_index.json').write_text(json.dumps({'generator_commit':generator_commit,'evidence':evidence},indent=2))
    rows=[]
    for c in s['cases']:
        if c['case'].startswith('seed'):rows.append({'profile':'structure_f64','case':c['case'],'cycle':c['cycle'],'batches':96,'exact_half_bits':None,'max_task_error':c['max_task_error']})
    for c in p['cases']:
        if c['case'].startswith('seed'):rows.append({'profile':'modeled_fp16_fp32','case':c['case'],'cycle':c['cycle'],'batches':c['batches'],'exact_half_bits':c['bit_equal_batches'],'max_task_error':c['max_task_error']})
    with (out/'numerical_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    links=[f'[{path.parent.parent.name}/{path.parent.name}]({os.path.relpath(path,out)})' for path in sources]
    text=['# Asim数值重建流程验证报告（论文资料阶段稿）','',
      '日期：2026-10-30（Asia/Hong_Kong）；状态：'+state['status']+'。当前不具备生成“全部验证完成”最终报告的条件。', '',
      '## 研究问题与方法','',
      '目标是在通信质量约束下减少NPU通信算子周期。验证对象是实际生成的Instruction，而不是按算法名调用标准求解，也不是仅重放FormulaLogger声明。共同任务输入H/Y/λ，输出MMSE均衡结果；完整计时包括Gram、初始化、迭代、输出恢复和DMA。', '',
      '三路径分别为实际指令重建、同初值/参数的独立算法参考、独立高精度solve任务参考。Asim成功、成本/事件绑定与性能预筛之后才执行数值；不竞争或异常加速保持NOT_RUN。', '',
      '## 与早期流程的区别','',
      '| 环节 | 早期证据边界 | 当前新增约束 |','|---|---|---|',
      '| 数值来源 | 公式日志或算法名不能证明实际Instruction | 源构造点视图、opcode/地址/shape/DMA/成本绑定 |',
      '| 多核与输入 | 单样例、batch/输出绑定可能混用 | 24核、batch96、每batch独立输入与输出检查 |',
      '| 数学真值 | DAG自一致不等于MMSE正确 | 独立同算法与高精度任务两种参考 |',
      '| 精度 | FP64结构通过不等于FP16 | 逐指令FP16写回、FP32逐k累加模型与完整位比较 |',
      '| 周期 | 不同输入、范围或估算可能混排 | task/input/config/scope绑定；profile不同不算优化收益 |',
      '| 报告 | PASS陈述可能缺原件 | commit/输入/工具链/验证器/制品hash与最终报告硬门禁 |', '',
      '## 实验设置','',
      '共同配置：瑞利H(64×16)、Y(64×8)、λ=0.1，batch96，24核，Cube16³；种子42/123/456。采用显式实块编码，物理H128×32、输出32×16。NS K16，X0=I/trace(R(A))；Gram、trace初始化、负号MUL与最终X@b均实际计时。', '',
      '精度模型自定义为：FP16存储与每指令写回，FP32标量/Vector运算，GEMM按k升序非融合FP32乘后加。输入编码后RNE量化一次；subnormal保留、非有限拒绝。独立参考自行实现同操作顺序，任务oracle使用原始H/Y/λ。Vector工作量按FP32宽度计，存储按FP16宽度计。', '',
      '这不是原生910B精度/转换成本校准。2%任务误差目标是本轮工程目标，不是硬件误差界或正式通信质量阈值。', '',
      '## 数值结果','', '| profile | seed | batch | 周期 | FP16逐位相同batch数 | 最大任务相对误差 |','|---|---|---:|---:|---:|---:|']
    text += [f"| {r['profile']} | {r['case']} | {r['batches']} | {r['cycle']} | {r['exact_half_bits'] if r['exact_half_bits'] is not None else '不适用'} | {r['max_task_error']:.6g} |" for r in rows]
    text += ['', 'FP64与FP16模型的周期属于不同profile，不能把差值写成同精度优化收益。旧NS4,647周期与新完整任务14,298周期也不可直接比较。', '',
      '## 拒绝错误与剩余问题','',
      '低K候选能与算法参考一致而任务质量失败；漏LOAD、错误残差、最终平方、跨batch读取、工作量低报和不匹配基线均有拒绝证据。旧七入口仍有五类语义歧义入口、BRI越界及NS输入/数值失败，不可声称全部算法正确。', '',
      'RAW/WAR/WAW检查器已测试部分写、初始化、core/buffer别名及合法重叠；这些是合成事件模型测试。实际Asim缺少一一绑定的读完成/写可见事件，pipeline寿命不能替代G4。', '',
      '## 最终报告门禁','', '| 门禁 | 当前状态 |','|---|---|']
    text += [f'| {k} | {v} |' for k,v in state['gates'].items()]
    text += ['', '只有全部必需门禁PASS才允许--final。当前可用于论文方法、实验设置与阶段性数值结果；不得写成完整框架、全部算子、硬件精度、CDL-B、通信质量或实机性能均已验证。', '',
      '## 证据与复现','',*['- '+x for x in links],'',
      'readiness.json为机器可读完成状态，evidence_index.json绑定原清单hash与提交；numerical_results.csv为可用于论文作图的原始小表。大trace、输入、参考、输出和二进制保留原目录，旧报告不覆盖。']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n');return state


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('structure_manifest','precision_manifest','legacy_manifest','output'):ap.add_argument('--'+n.replace('_','-'),type=Path,required=True)
    ap.add_argument('--final',action='store_true');a=ap.parse_args()
    state=generate(a.structure_manifest,a.precision_manifest,a.legacy_manifest,a.output,a.final);print(json.dumps(state))

if __name__=='__main__':main()
