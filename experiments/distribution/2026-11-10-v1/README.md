# 专家源码部署包验收 v1

日期：2026-11-10（香港）。状态：**六类部署门禁通过、目录迁移通过、数据清理与压缩完成**。本轮不修改算子算法，不进行性能晋级，不切换BRI默认基线。

## 交付

- 项目目录：`expert_delivery/asim-expert-v1/`。
- 压缩包：`expert_delivery/asim-expert-v1.tar.gz`，37,399,692 bytes（约35.67 MiB）。
- SHA-256：`1583732bcc831e6bb6d9bcf61403eacb1af8662ac15a486d336550558fab2387`；同目录提供.sha256。
- 包内README给出Linux x86_64/WSL2 Ubuntu安装和离线构建命令；docs/PYTHON_TO_OPERATOR.md、docs/RECONSTRUCTION.md分别说明两个专家关注点；docs/OPERATORS.md列出入口及冻结参数。
- 包含Asim完整src、六家族及既有变体源码、配置、当前重建核心/独立参考/测试、三项原始开发skill、第三方源码及许可证、Python3.12 x86_64离线wheel。默认验收使用六个修正版入口；旧源码存在不等于通过验证。
- 不包含原实验数据/结果、模型权重、账号凭据、Git历史、Conan账户缓存、服务器二进制、build/.venv或本轮测试数据。

## 验证结果

统一Rayleigh seed42、Nr64/Nt16/RHS8、batch96、λ0.1、24核/16³ Cube成本模型；完整H/Y→MMSE计时边界。采用FP16存储/FP32临时算术、逐原语写回、保留次正规模型，不宣称硬件级混合精度认证。

| 家族 | K（直接法为占位1） | 周期 | 最大MMSE相对误差 | 96batch逐FP16位一致 |
|---|---:|---:|---:|---|
| cholesky | 1 | 28,006 | 0.254045% | PASS |
| ldl | 1 | 18,697 | 0.254293% | PASS |
| cholesky-block | 1 | 21,338 | 0.218327% | PASS |
| ldl-block | 1 | 21,665 | 0.193923% | PASS |
| ns | 16 | 7,950 | 0.105726% | PASS |
| bri | 192 | 23,047 | 0.149710% | PASS |

六类都为MODELED_NUMERIC_CHECKED、MODEL_RW_CHECKED_CONSERVATIVE；无失败batch，全部低于2%任务门限。同一96个H在六类间复用，是576次算子batch验证，**不是576个独立信道**。六类串行门禁总墙钟287.32秒。周期仅用于本固定部署任务的回归记录，不与旧报告拼成排行榜；score=null，首次baseline_creation不等于性能优化晋级。

1. 新交付目录从依赖源码构建，不复用原Asim二进制/对象文件。配置缺失文件、ONNX导出宏、bundled fmt链接问题均修复在打包层；详情build_notes.json。最终bootstrap离线安装/配置/增量构建复跑通过。
2. 包内独立venv/NumPy2.5.0执行250项测试：248通过、2显式跳过（可选SciPy独立区间检查、Torch/Sionna CDL生成）。最终日志results/distribution-unit-tests-final.log。
3. 实际逐类Asim→Instruction/trace/真实Core事件检查→同算法参考→通用重放→FP64 MMSE均通过，详见six_case_summary.json及CSV。Python原型示例也已执行；它本身不证明C++正确。
4. 将完整包移动到含空格的新目录，清除LD_LIBRARY_PATH后ldd找到新目录中的Ramulator，无Conan或旧build动态库路径。新空build配置成功；同NS实际Asim/重建再次通过（7,950周期、误差不变、逐位一致），见relocation_ns_summary.json。该重跑不是新独立信道。
5. 对照原工程151个src文件SHA全部一致。verification对照113个文件，仅两处预期的部署测试适配（独立contract夹具与可选CDL跳过）；重建核心/参考不变，见verification_source_check.json。
6. 包内生成的build/.venv/validation-work/relocation-build与Python缓存已清理。归档包含16,541个源码/配置/依赖文件的清单；重新解压后全部SHA通过，修改README的负例被拒绝；无数据/结果/build/虚拟环境或符号链接，见archive_check.json。

原始测试数据按本轮要求删除，仅在包外保留小型摘要、版本/文件SHA、环境及关键日志索引。summary中原始制品SHA记录的是删除前文件，不宣称原件仍在。新机器可用--keep-raw重新生成证据。

## 证据与边界

- environment.json：源版本/打包版本、工具链、binary/库SHA、资源；wheel_inventory.json：离线wheel版本/大小/SHA与许可证文件计数。
- SOURCE_PROVENANCE/PACKING_MANIFEST在包内；打包工具及模板见packaging/。source_commit=0aa874e1，打包修复commit=c138ffec；归档SHA固定本交付版本。
- 原始构建/运行日志在results/distribution-*.log，未随包分发；摘要集中本目录，不混入旧周期结果。初次配置/链接失败保留说明，没有隐藏或改算法放行。
- 真实目标为Linux x86_64；本轮实际执行环境Ubuntu24.04/GCC13/CMake3.28/Python3.12、2vCPU。WSL2与另一主机尚未实机执行。换目录检查不等于跨主机验收。
- 本轮只验规定Rayleigh样本、六个修正入口；不升级旧CDL-B失败、不认证全部旧变体/可选HB，不宣称完整通信、硬件时序或性能晋级。
- skill是Agent的工程约束流程，不是自动转译编译器。typed views补齐数值语义并与真实指令绑定，但软件模型不证明未来NPU微码切片/硬件时序；详见包内两份原理说明。

## 空间管理

为完成打包，按先提交的dedup_plan.json将40个重复且SHA相同的历史诊断文件转为只读硬链接，保留原路径与字节，共回收1,761,948,368 bytes。执行台账dedup_execution.json；未删除历史研究输入、输出或报告。只清理本包自己生成的验证数据和构建缓存，未动用户原有未跟踪资料。

## 企业机下一步

校验压缩包SHA→解压→python3 scripts/check_package.py→bash scripts/bootstrap.sh→scripts/smoke.py --operator all --output work/new-run。在企业实际环境执行这一步并保留--keep-raw结果后，才补充跨主机验收记录。BRI后续优化仍沿原内循环台账推进，不以本部署验收替代新校准/留出或晋级。
