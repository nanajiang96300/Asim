# 桌面 Codex → SSH → Ubuntu 工程：操作指南

日期：2026-10-28。官方命令已核对；本文命令供用户部署。当前只完成文档归档，尚未执行安装或用户认证。

目标主机：`nanajiang@192.168.31.110:22`；工程：`/home/nanajiang/Asim/`。

## 1. Windows：添加 SSH 主机别名

在本机 PowerShell 执行：

```powershell
New-Item -ItemType Directory -Force "$HOME\.ssh" | Out-Null
notepad "$HOME\.ssh\config"
```

在编辑器末尾加入下面配置；若已有同名 Host，应编辑原有段落，避免重复：

```sshconfig
Host asim-ubuntu
    HostName 192.168.31.110
    User nanajiang
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

保存后测试：

```powershell
ssh asim-ubuntu
```

输入密钥口令，成功后得到 Ubuntu shell。密钥口令与 Ubuntu 用户密码不同。

可选：减少重复输入密钥口令。在**管理员 PowerShell**执行以下两行：

```powershell
Set-Service -Name ssh-agent -StartupType Automatic
Start-Service ssh-agent
```

随后在普通 PowerShell 执行，并在提示时输入密钥口令：

```powershell
ssh-add "$HOME\.ssh\id_ed25519"
ssh-add -l
```

只在提示中输入口令，不把口令写入 SSH config 或脚本。

## 2. Ubuntu：安装 Codex

以下在刚才登录的 **Ubuntu shell** 中执行。服务器已有 curl，采用官方独立安装器，无需先安装 Node/npm，也不需要 sudo：

```bash
cd /home/nanajiang/Asim
curl -fsSL https://chatgpt.com/codex/install.sh -o /tmp/asim-codex-install.sh
sh /tmp/asim-codex-install.sh
```

安装器交互完成后，在当前 shell 加入默认安装路径：

```bash
export PATH="$HOME/.local/bin:$PATH"
command -v codex
codex --version
```

默认可见命令路径是 `~/.local/bin`。如果安装器提示了其他目录，以其实际输出为准。

让新的登录 shell 也能找到它；当前服务器检查到有 `~/.profile`，无 `~/.bash_profile`：

```bash
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.profile" || \
  printf '\n%s\n' 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.profile"
bash -lc 'command -v codex && codex --version'
```

若 curl 下载失败，先保留完整错误，检查 Ubuntu 访问官方服务的网络连通性。Windows 能联网不代表 Ubuntu 能联网；不要用修改 TLS 校验的方式解决下载问题。

## 3. Ubuntu：登录认证

```bash
codex login --device-auth
```

按终端提示在 Windows 浏览器打开网址、输入一次性验证码，并用自己的 ChatGPT 账户完成授权。设备码登录可能需要在个人安全设置或工作区权限中启用。

完成后执行：

```bash
codex login status
```

如果设备码登录不可用，先退出当前 SSH shell，在 **Windows PowerShell** 建立回调转发：

```powershell
ssh -L 1455:localhost:1455 asim-ubuntu
```

然后在该会话的 **Ubuntu shell** 执行：

```bash
export PATH="$HOME/.local/bin:$PATH"
codex login
```

在 Windows 浏览器打开它打印的网址完成登录。若本机 1455 已被占用，先关闭此前的登录会话再重试。此方案无需把认证缓存复制到聊天或工程。

## 4. Windows：验收远端非交互环境

在另一个 **Windows PowerShell** 中执行：

```powershell
ssh asim-ubuntu 'command -v codex && codex --version && codex login status'
```

这条命令用于验证桌面端启动远端进程时也能找到 Codex。若只在交互终端能找到，检查默认登录 shell、PATH 和启动文件，不继续配置桌面直到此项通过。

## 5. 桌面端：选择 SSH 工程

这一步需要图形界面，不能仅用一条 shell 命令替代：

1. 打开桌面 Codex，进入 **Settings → Connections → SSH**（名称随客户端语言可能不同）。
2. 添加或启用 `asim-ubuntu` 主机。
3. 选择远程项目目录 `/home/nanajiang/Asim/`。
4. 在这个远程项目中新建聊天，确认运行位置指向 Ubuntu。

连接后由桌面端通过 SSH 启动远端 app server，不需要手动开放 WebSocket 端口。当前本地 projectless 聊天不会仅因服务器安装了 Codex 自动转成远程聊天。

第一条验证提示建议：

> 请读取 AGENTS.md 和 DOCS/research/README.md，执行 pwd、hostname、git branch --show-current 与 git status --short，确认你直接位于 /home/nanajiang/Asim。先只检查环境并汇报，不修改代码。

预期工作目录为 `/home/nanajiang/Asim`，主机名为 `nanajiangserver`。当前文档整理分支为 `research/codex-transition-20261028`。

## 6. 启动主 Agent／子 Agent

桌面远程项目中直接输入：

> 请使用两个子 Agent 并行开展只读检查：一个梳理 NS 指令、初值和最终输出，另一个梳理现有数值验证脚本。主 Agent 等待两者完成，汇总最小数值影子执行方案与需要补齐的语义字段，先不修改工程。

这是首个子 Agent 连通性与分工验收，不是自动算法搜索。原生子 Agent 不会因为被创建就自动获得独立工作区。

如果临时不用桌面端，在 **Ubuntu shell** 启动 CLI：

```bash
cd /home/nanajiang/Asim
codex -C /home/nanajiang/Asim
```

在 CLI 对话中发送同样提示，使用 `/subagents` 查看子 Agent。当前官方版本支持项目级 `.codex/agents/*.toml` 自定义角色；初次连通测试不需要先配置角色文件。需要固定角色时再将配置加入 Git。

## 7. 候选工作区与构建

先完成只读环境检查。需要开始第一个真实实现任务时，由控制器或主 Agent 在服务器创建独立候选工作区。下面是明确的新分支／目录示例，仅执行一次；同名已存在时先检查，不能强制覆盖：

```bash
cd /home/nanajiang/Asim
mkdir -p /home/nanajiang/Asim-worktrees
git worktree add -b infra/ns-shadow-pilot \
  /home/nanajiang/Asim-worktrees/ns-shadow-pilot HEAD
git -C /home/nanajiang/Asim-worktrees/ns-shadow-pilot status --short
```

将写代码子 Agent 的任务明确指向该路径。注意 Git worktree 默认不复制忽略的 build 和 .venv，子模块与 Conan 依赖也需按工程实际构建方式初始化。不要假定在新 worktree 中执行 build 就能成功，更不能为省事让不同候选共用可写 build。

现有主工程如需单独验收既有 Simulator 构建，可在其现有 build 目录中执行以下命令；它会实际构建，不代表新验证模块已经存在：

```bash
cd /home/nanajiang/Asim
cmake --build build --target Simulator -j1
```

完整自动闭环的数值影子执行入口和新评分入口尚未实现，因此现在不能给出一个真实可用的“启动全自动闭环”命令。下一开发任务是 NS 影子执行 pilot，通过后再增加统一 runner 命令与运行清单。

## 8. 部署记录

部署后将以下输出中的非敏感信息记录到研究日志：日期、`codex --version`、`command -v codex`、Ubuntu 版本、运行主机、工程 commit、设备码／回调认证方式、桌面连接成功与子 Agent 测试结果。不记录验证码、token 或 auth.json 内容。

CLI 升级应作为环境版本变更记录，不在一轮搜索中自动更新。构建／仿真先设为单任务，避免 2 vCPU 上过度并发。

## 官方依据（2026-10-28 核对）

- [Codex CLI 安装](https://learn.chatgpt.com/docs/codex/cli)
- [安装器路径与环境变量](https://learn.chatgpt.com/docs/config-file/environment-variables)
- [远程 SSH 工程](https://learn.chatgpt.com/docs/remote-connections)
- [远端认证与设备码／SSH 回调](https://learn.chatgpt.com/docs/auth)
- [原生子 Agent](https://learn.chatgpt.com/docs/agent-configuration/subagents)
