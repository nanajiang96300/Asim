# S1 共享源码便携构建验收

状态：PASS（Linux x86_64 Ubuntu24.04/GCC13/CMake3.28/Python3.12）。WSL2/跨主机未实测。两个客户端共享一套源码与数值门禁，本轮不是模型生成算子的验收；该项另见S2报告。

两个全新浅clone分别检出Codex和Claude配置，独立依赖目录、venv、build。Codex联网按固定commit下载；Claude复制三个已校验缓存后`bootstrap --offline`，路径含空格。依赖来源、锁和许可见`dependencies/`与依赖分支`codex/asim-deps-v1`。未复制账号、API配置或旧二进制。

完整实测版本/工具输出/耗时见progress.json。两个冷源码构建各约27分钟、默认-j1；每个环境250原测试中248通过、2个可选依赖明确跳过，另4个bootstrap负例通过。两环境各执行六种真实Asim→实际Core事件→实际指令重建→独立同算法FP16逐位比较→FP64 MMSE误差门禁，全部通过。

| 算法入口 | 迭代 | 周期（两个环境一致） |
|---|---:|---:|
| Cholesky explicit | 1 | 28006 |
| LDL explicit | 1 | 18697 |
| Cholesky block explicit | 1 | 21338 |
| LDL block explicit | 1 | 21665 |
| NS corrected | 16 | 7950 |
| BRI explicit | 192 | 23047 |

统一条件：24core，batch96，Cube16×16×16；Rayleigh seed42，同一96个H跨算法复用；Nr64/Nt16/RHS8，λ0.1；FP16存储/FP32临时模型，2%相对任务误差门限。误差和所有生成制品删除前SHA见两份six.json，不能把六组当576独立H，也不是统一性能晋级/CDL或通信认证。源src和原verification在S1测试快照中保持原研究内容；之后S2通用任务扩展单独验收。

动态库仅自有ramulator和系统库，RUNPATH为相对$ORIGIN/../lib；无服务器依赖绝对路径。测试后仅清理本次生成PCH和Codex展开依赖，原件恢复方法与字节数见cleanup清单；Codex可从其缓存`bootstrap --prepare-only --offline`恢复。Claude测试完才作为新算子独立执行目录，S1二进制/库已先冻结保存。原始研究资料未清理。

原始运行日志根目录及所有SHA见artifacts.json；Git只存小型摘要，raw测试制品按脚本默认在确认摘要后删除。尚未解决事项见[S2依赖与边界](S2_DEPENDENCIES.md)。
