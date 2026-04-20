# AutoDL SSH 配置教程（给 Claude 用）

> 每次开新 AutoDL 实例后，按这个教程把新的 SSH 信息配好，Claude 就能直接连过去跑实验。

## 前置：从 AutoDL 获取连接信息

1. 登录 [AutoDL 控制台](https://www.autodl.com/console/instance/list)
2. 找到你的实例，点「快捷工具」→「JupyterLab」或查看 SSH 信息
3. 记下这三个东西：

| 项目 | 示例 | 在哪找 |
|------|------|--------|
| **登录地址** | `connect.bjb1.seetacloud.com` | SSH 信息里 Host 字段 |
| **端口** | `43348` | SSH 信息里 Port 字段 |
| **密码** | `9Iwv2c0zLhJs` | SSH 信息里 Password 字段，或控制台设置 |

---

## 一键配置（在 Claude 的终端里跑）

### Step 1：更新 SSH config

**注意**：把下面的 `HostName`、`Port` 替换成你实际的值。

在 Claude 的聊天框输入：

```
! cat > ~/.ssh/config << 'EOF'
Host autodl
    HostName connect.bjb1.seetacloud.com
    Port 43348
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 30
    ServerAliveCountMax 3
EOF
```

> `StrictHostKeyChecking no` 和 `UserKnownHostsFile /dev/null` 是为了避免每次换实例都要手动确认 host key。

### Step 2：更新密码文件

**注意**：把 `你的新密码` 替换成实际密码。

```
! echo '你的新密码' > ~/.ssh/autodl_pass && chmod 600 ~/.ssh/autodl_pass
```

### Step 3：告诉 Claude 信息已更新

在聊天里直接跟我说：

```
autodl 已经配好了，请继续跑实验
```

或者更具体：

```
新 AutoDL 实例已配置，连接地址 connect.bjb1.seetacloud.com:43348，
你可以 sshpass -f ~/.ssh/autodl_pass ssh autodl 连过去
```

---

## Claude 内部使用的命令（参考）

Claude 会用这些命令连到 AutoDL：

```bash
# 测试连接
sshpass -f ~/.ssh/autodl_pass ssh autodl "echo ALIVE && nvidia-smi -L"

# 跑命令
sshpass -f ~/.ssh/autodl_pass ssh autodl "cd /root/autodl-tmp/VSR-Place && <command>"

# 传文件
sshpass -f ~/.ssh/autodl_pass scp local_file.py autodl:/root/autodl-tmp/VSR-Place/local_file.py
```

如果这些命令报错，通常是：
- `Connection refused` → 实例停机了，去 AutoDL 控制台开机
- `Permission denied` → 密码错了，更新 `~/.ssh/autodl_pass`
- `Host key verification failed` → SSH config 里没加 `StrictHostKeyChecking no`

---

## 新实例首次使用（第一次开实例时）

如果是全新 AutoDL 账号或完全新建的实例，还需要：

### 1. 选镜像

推荐：`PyTorch 2.6+ / CUDA 12.8 / Python 3.10`（或任意 PyTorch 2.3+ 的镜像）

### 2. 连接测试

```
! sshpass -f ~/.ssh/autodl_pass ssh autodl "nvidia-smi"
```

应该看到 GPU 信息（RTX 4090 / 3090 / 5090 等）。

### 3. 如果实例是全新的，需要完整部署

跟 Claude 说：

```
新 AutoDL 实例，需要完整部署。按 docs/autodl_guide.md 流程走：
clone 仓库 → 装依赖 → 下 checkpoint → 下数据
```

### 4. 如果实例继承了之前的数据盘（推荐做法）

数据盘 `/root/autodl-tmp/` 是持久化的，关机不丢。所以：

```
! sshpass -f ~/.ssh/autodl_pass ssh autodl "ls /root/autodl-tmp/VSR-Place/"
```

如果能看到 `README.md`, `src/`, `checkpoints/large-v2/` 等，说明之前的环境还在。
只需要告诉 Claude：

```
AutoDL 已配置好，VSR-Place 目录在 /root/autodl-tmp/VSR-Place，
请 git pull 同步最新代码然后继续跑实验
```

Claude 会：
1. SSH 进去
2. `git pull` 更新代码
3. 继续上次中断的实验

---

## 快速参考卡

| 情况 | 你要做的 | 跟 Claude 说 |
|------|---------|-------------|
| 同一实例，只是重启了 | 不用做什么 | "AutoDL 已恢复，继续跑" |
| 换了新实例（同账号） | 更新 `~/.ssh/config`（Host/Port）+ `~/.ssh/autodl_pass` | "AutoDL 换了新实例，已配好新的 SSH 信息" |
| 全新账号 | 开实例 → 配 SSH → 告诉 Claude 需要完整部署 | "新 AutoDL 账号，全新实例，请完整部署" |

---

## 故障排查

### 问题 1：`ssh: connect to host ... Connection refused`

**原因**：实例没开机
**解决**：去 AutoDL 控制台「开机」，等 30 秒再试

### 问题 2：`ssh: Permission denied`

**原因**：密码错了（换实例后密码会变）
**解决**：从 AutoDL 控制台复制最新密码，重新写入 `~/.ssh/autodl_pass`

### 问题 3：实例关机后数据还在吗？

**数据盘 `/root/autodl-tmp/`**：**持久化**，关机不丢（包括 checkpoint、数据、代码）
**系统盘 `/root/` 其他位置**：**每次开机重置**，装的 pip 包可能需要重装

建议把所有项目文件放 `/root/autodl-tmp/`，pip 包也可以装到这里（`pip install --target=/root/autodl-tmp/pypkgs`）。

### 问题 4：密码泄露了怎么办？

- AutoDL 控制台可以重置 SSH 密码
- 重置后记得更新 `~/.ssh/autodl_pass`
