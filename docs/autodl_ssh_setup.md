# AutoDL SSH 配置教程（给 Claude 用）

> 每次开新 AutoDL 实例，你只需要给我两样东西：
> 1. **登录指令**（AutoDL 控制台复制的那行，形如 `ssh -p 12345 root@connect.xxx.seetacloud.com`）
> 2. **密码**（形如 `9Iwv2c0zLhJs`）
>
> 你有两种用法：**手动**（你在聊天框执行一个命令）或 **托管**（你直接把两样信息告诉 Claude，Claude 自己配）。

---

## 用法 A：手动配置（最简单，推荐）

把下面这段命令复制到 Claude 聊天框，**替换里面的两个值**然后发送：

```
! SSH_CMD='ssh -p 12345 root@connect.bjb1.seetacloud.com' && \
  PASSWORD='你的密码' && \
  PORT=$(echo "$SSH_CMD" | grep -oP '(?<=-p )\d+') && \
  HOST=$(echo "$SSH_CMD" | grep -oP 'root@\K[^ ]+') && \
  mkdir -p ~/.ssh && \
  cat > ~/.ssh/config << EOF
Host autodl
    HostName $HOST
    Port $PORT
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 30
EOF
  echo "$PASSWORD" > ~/.ssh/autodl_pass && chmod 600 ~/.ssh/autodl_pass && \
  sshpass -f ~/.ssh/autodl_pass ssh autodl "echo 'SSH OK' && nvidia-smi -L"
```

**替换规则**：
- `ssh -p 12345 root@connect.bjb1.seetacloud.com` → AutoDL 给你的那行**登录指令原样粘贴**
- `你的密码` → AutoDL 给你的**密码原样粘贴**

执行成功会看到：
```
SSH OK
GPU 0: NVIDIA GeForce RTX 4090 (UUID: ...)
```

然后告诉 Claude：

```
autodl 已配好，继续跑实验
```

---

## 用法 B：托管给 Claude 配（懒人版）

直接把两样信息发给 Claude：

```
帮我配 autodl，登录指令: ssh -p 12345 root@connect.bjb1.seetacloud.com
密码: 你的密码
```

Claude 会自动解析并执行上面的配置命令。

> ⚠️ **安全提醒**：密码会出现在聊天记录里。如果介意可以用「用法 A」自己执行。

---

## 验证配置成功

配置后 Claude 会跑这个确认：

```bash
sshpass -f ~/.ssh/autodl_pass ssh autodl "nvidia-smi -L && ls /root/autodl-tmp/VSR-Place 2>/dev/null"
```

如果显示 GPU 信息 + `README.md src/ ...`，说明：
1. SSH 连通 ✅
2. 之前的项目还在（数据盘持久化）✅

---

## 常见情况

### 换了新实例（最常见）

AutoDL 控制台「释放实例」再「新建」→ 登录指令和密码都会变。
按上面「用法 A」或「用法 B」**重跑一次**配置即可。

### 同一实例只是重启

SSH 信息不变，啥都不用做。直接告诉 Claude：
```
autodl 已恢复，继续跑
```

### 全新账号（第一次用）

配好 SSH 后告诉 Claude：
```
新 AutoDL，需要从头部署，请按 docs/autodl_guide.md 流程走
```

---

## 故障排查

| 报错 | 原因 | 解决 |
|------|------|------|
| `Connection refused` | 实例没开机 | AutoDL 控制台「开机」 |
| `Permission denied` | 密码错了 | 重跑配置命令，检查密码复制对了没 |
| `grep: -P: 不支持` | 系统没 PCRE grep | 改用 `ssh -p 12345 root@host.com` 手动填到 `~/.ssh/config` |
| `sshpass: command not found` | 没装 sshpass | `! apt-get install -y sshpass` |

---

## 快速参考：Claude 内部用的命令

```bash
# 测连接
sshpass -f ~/.ssh/autodl_pass ssh autodl "echo ALIVE"

# 跑远程命令
sshpass -f ~/.ssh/autodl_pass ssh autodl "cd /root/autodl-tmp/VSR-Place && <cmd>"

# 传文件上去
sshpass -f ~/.ssh/autodl_pass scp file.py autodl:/root/autodl-tmp/VSR-Place/

# 从远程拉文件
sshpass -f ~/.ssh/autodl_pass scp autodl:/root/autodl-tmp/VSR-Place/results.csv .
```
