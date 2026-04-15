# AutoDL RTX 5090 部署指南

## Step 1：租实例

1. 登录 [AutoDL](https://www.autodl.com/)
2. 点击「容器实例」→「创建实例」
3. 选择配置：

| 项目 | 选择 |
|------|------|
| GPU | RTX 5090 (32GB) |
| 镜像 | PyTorch 2.6.x / CUDA 12.8 / Python 3.10 |
| 数据盘 | ≥50GB |

4. 点击「创建」，等待实例启动

---

## Step 2：进入终端

- 实例启动后，点击「JupyterLab」或「终端」进入命令行
- 或用 SSH 连接（实例详情页有 SSH 地址和密码）

---

## Step 3：验证 GPU

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

确认输出包含 `True` 和 `RTX 5090`。

---

## Step 4：克隆仓库

```bash
cd /root/autodl-tmp
git clone --recursive https://github.com/moyunxiang/VSR-Place.git
cd VSR-Place
```

> `--recursive` 会一并拉取 ChipDiffusion 子模块。如果子模块拉取失败：
> ```bash
> git submodule update --init --recursive
> ```

---

## Step 5：安装依赖

```bash
pip install -r requirements.txt
pip install -e .
```

验证安装：

```bash
python -c "from vsr_place.verifier.verifier import Verifier; print('OK')"
pytest tests/ -v
```

应看到 49 个测试全部通过。

---

## Step 6：下载预训练权重

### 方法 A：gdown（需要能访问 Google Drive）

```bash
pip install gdown
bash scripts/download_checkpoints.sh
```

### 方法 B：手动上传

如果 Google Drive 被墙：

1. 在本地电脑用浏览器打开：https://drive.google.com/drive/folders/16b8RkVwMqcrlV_55JKwgprv-DevZOX8v
2. 下载 `large-v2` 文件夹（里面有 `.ckpt` 文件）
3. 通过 AutoDL 的文件管理器上传到 `/root/autodl-tmp/VSR-Place/checkpoints/large-v2/`

### 方法 C：用学术网络代理

```bash
# 如果有代理
export http_proxy=http://xxx:port
export https_proxy=http://xxx:port
bash scripts/download_checkpoints.sh
```

验证：

```bash
ls -lh checkpoints/large-v2/
```

应看到 `.ckpt` 文件（约 1-2GB）。

---

## Step 7：准备数据集

### 合成数据（快速验证用）

```bash
cd third_party/chipdiffusion
PYTHONPATH=. python data-gen/generate_parallel.py versions@_global_=v1
cd ../..
```

### IBM 基准（论文正式实验用）

```bash
cd third_party/chipdiffusion

# 1. 下载 IBM benchmarks（DEF/LEF 格式）到 benchmarks/ibm/
mkdir -p benchmarks/ibm
# 需要从 https://vlsicad.ucsd.edu/GSRC/bookshelf/Slots/Mixed-Size/ 手动下载
# 或者问导师/实验室是否已有数据

# 2. 下载 hmetis（聚类工具）
# 从 http://glaros.dtc.umn.edu/gkhome/metis/hmetis/overview 下载
# 放到 chipdiffusion 根目录

# 3. 运行解析 + 聚类
PYTHONPATH=. python parsing/cluster.py
PYTHONPATH=. python parsing/cluster.py num_clusters=0

# 4. 配置数据集
cp datasets/graph/config.yaml datasets/clustered/

cd ../..
```

### ISPD 2005 基准

```bash
cd third_party/chipdiffusion

# 1. 下载 ISPD2005 benchmarks（bookshelf 格式）到 benchmarks/ispd2005/
mkdir -p benchmarks/ispd2005
# 从 http://www.ispd.cc/contests/05/ispd2005_contest.html 下载

# 2. 使用 notebook 解析
# 打开 JupyterLab，运行 notebooks/parse_bookshelf.ipynb

cd ../..
```

---

## Step 8：快速验证（先跑通再说）

找到 checkpoint 文件名：

```bash
CKPT=$(ls checkpoints/large-v2/*.ckpt | head -1)
echo "Using checkpoint: $CKPT"
```

跑一个最小实验确认全链路通畅：

```bash
# 单样本、单种子、合成数据
python scripts/run_vsr.py \
    --checkpoint "$CKPT" \
    --task v1.61 \
    --seed 42
```

如果看到类似输出说明成功：

```
Loading checkpoint: checkpoints/large-v2/step_250000.ckpt
Successfully loaded checkpoint...
Loaded 18 validation samples from task 'v1.61'
Canvas size: 2.0 x 2.0
Running vsr_place on 18 samples (seed=42)...
  Sample 1/18... violations=3 (12.5s)
  Sample 2/18... LEGAL (8.2s)
  ...
Pass rate: 14/18 (77.8%)
```

---

## Step 9：跑全部实验

```bash
CKPT=$(ls checkpoints/large-v2/*.ckpt | head -1)

# 一键跑完所有基线 + VSR + 消融（耗时约 3-8 小时）
nohup bash scripts/run_all.sh "$CKPT" v1.61 42 > experiment.log 2>&1 &

# 查看进度
tail -f experiment.log
```

如果只想跑部分实验：

```bash
# 只跑 VSR-Place
python scripts/run_vsr.py --checkpoint "$CKPT" --task v1.61 --seed 42

# 只跑基线
python scripts/run_vsr.py --checkpoint "$CKPT" --task v1.61 --seed 42 --no-vsr

# 只跑部分消融
python scripts/run_ablations.py --checkpoint "$CKPT" --task v1.61 --seed 42 \
    --ablations scope_global scope_threshold_0.0 strength_fixed_0.3

# 查看可用消融列表
python scripts/run_ablations.py --list
```

---

## Step 10：查看结果

```bash
# 汇总结果表
python scripts/aggregate_results.py

# 生成可视化
python scripts/visualize.py

# 结果位置
ls results/runs/       # 每次实验的详细输出
ls results/tables/     # 汇总表格（CSV + LaTeX）
ls results/figures/    # 图表
```

---

## Step 11：拉取结果到本地

在本地机器执行：

```bash
# 用 AutoDL 实例详情页的 SSH 信息
scp -P <端口> -r root@<地址>:/root/autodl-tmp/VSR-Place/results/ ./results/
```

或在 AutoDL JupyterLab 中打包下载：

```bash
cd /root/autodl-tmp/VSR-Place
tar czf /root/autodl-tmp/vsr_results.tar.gz results/
# 然后在 JupyterLab 文件管理器中下载 vsr_results.tar.gz
```

---

## Step 12：同步结果回 GitHub

实验跑完后，在 AutoDL 实例上把结果提交并推送：

```bash
cd /root/autodl-tmp/VSR-Place

# 1. 配置 git（如果还没配）
git config user.name "moyunxiang"
git config user.email "2556377578@qq.com"

# 2. 配置 SSH key（如果还没配）
ssh-keygen -t ed25519 -C "2556377578@qq.com" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
# 把输出的公钥添加到 GitHub: https://github.com/settings/keys

# 3. 测试连接
ssh -T git@github.com
# 应显示: Hi moyunxiang! You've successfully authenticated...

# 4. 更新 .gitignore，允许结果文件被提交
#    results/runs/ 和 results/tables/ 默认被 gitignore 了
#    如果想提交结果，先取消忽略：
echo '!results/tables/' >> .gitignore
echo '!results/tables/**' >> .gitignore

# 5. 添加结果文件
git add results/tables/ results/figures/ log.md
git add results/ablations/ 2>/dev/null  # 如果有消融结果

# 6. 提交
git commit -m "Add experiment results: <简要说明跑了什么>"

# 7. 推送
git push origin main
```

> **注意**：不要提交 `results/runs/` 里的原始 placement 数据（太大），只提交汇总表格和图表。

### 回到本地机器同步

```bash
# 在你的本地机器上
cd /home/dev/workspace/vsr_place
git pull origin main
```

这样本地和 GitHub 就保持同步了。

---

## 省钱技巧

| 技巧 | 说明 |
|------|------|
| 无卡模式装环境 | Step 3-7 不需要 GPU，用无卡模式 ¥0.1/h |
| nohup 后台跑 | Step 9 用 nohup 挂后台，跑完自动停 |
| 数据盘持久化 | checkpoint 和数据放 `/root/autodl-tmp/`，关机不丢失 |
| 分批跑 | 先跑核心实验验证方向，再跑消融 |

---

## 常见问题

### Q: `ModuleNotFoundError: No module named 'torch_geometric'`

```bash
pip install torch-geometric
```

### Q: `CUDA out of memory`

减少 batch size 或 num_samples：

```bash
python scripts/run_vsr.py --checkpoint "$CKPT" --task v1.61 --seed 42
# 默认 num_samples=1，不太可能 OOM
# 如果还 OOM，可能是模型太大，换更大显存的 GPU
```

### Q: Google Drive 下载失败

用方法 B（手动上传）或方法 C（代理）。

### Q: `successfully loaded model`（不是 `successfully loaded state dict`）

说明 checkpoint 参数不匹配。检查：
- 是否下载了正确的 `large-v2` checkpoint
- model config 的 backbone_params 是否和训练时一致

### Q: ChipDiffusion 子模块为空

```bash
git submodule update --init --recursive
```
