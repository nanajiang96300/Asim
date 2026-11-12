# Claude Code使用手册与无数据源码包

状态PASS，2026-11-12（香港时间）。源码版本675cd233328a8e2a120e408293f6f3e40bb16794，Claude profile；压缩包43,713,213字节（约41.69MiB），SHA见archive.json。交付文件位于原研究工程expert_delivery/asim-claude-guide-v1.tar.gz，旁附.sha256和独立中文说明。生成工具packaging/build_claude_release.py，只读已提交源树及固定缓存，不收集个人设置或未跟踪文件。

手册DOCS/CLAUDE_CODE_USER_GUIDE.md包含离线部署/Git导入、Claude启动与任务示例、六类入口、新算子Python→C++→实际指令重建、内循环基线/分支/预算/台账、OpenCode/其他平台适配边界、常见拒绝状态。包内使用说明.md为同字节别名。OpenCode指南依据官方CLI/Instructions/Providers文档，本轮未安装或实测其平台适配。

包中4570项逐文件清单，包括三份锁定依赖制品及许可证；无Git历史、experiments/results、输入/输出数组、trace、Asim二进制/build/venv或账号。源码包需要自行git init并生成本地导入commit，不能把它误认上游commit。仅一处测试历史契约路径替换为example-only合成fixture；实际C++、Asim模型及数值重放/独立参考核心不改变。

实际验证：新含空格路径安全解压、全部文件大小/SHA通过；无数据/认证路径检查通过；与前次真实构建的156项src/CMake/lock逐字节一致；离线依赖准备与NumPy安装通过，CMake Release配置通过；254项数值测试252通过/2可选跳过，另8项框架测试通过。新解压目录使用自己的Python/源码/配置完成九场景affine与重复评分、六类MMSE回归，均逐FP16位一致、任务门限通过；六类周期28006/18697/21338/21665/7950/23047，与之前相同。文件被改字节时包校验明确拒绝，恢复后再次完整校验通过。

本轮没有再冷编译Asim，运行复用前次已验收封存binary205dbd3d5db70b9b68462750eee6afa1d6d84be849becd11847ad7f8ddada055；其源码与本包编译输入已逐字节绑定。CMake配置/离线依赖/解压Python及真实Asim与重建是本轮新运行；原双新clone冷构建证据在experiments/deployment/2026-11-11-s1-v1/，不冒称本轮另做冷构建或WSL实测。

完整阶段时长见validation.json；九组与六类详见两个summary，日志在logs/。仅清理本次生成的源码暂存目录/验证解压目录/依赖缓存及临时数值输出，清理记录保留；不修改原研究输入/报告或旧manifest。压缩包在验证前已封存，未将测试结果或临时Git/依赖安装目录重新打入。
