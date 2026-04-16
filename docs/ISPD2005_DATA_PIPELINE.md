# ISPD2005 Benchmark — 数据拉取、处理与解析文档

> Date: 2026-04-16
> 用途: 记录 ISPD2005 benchmark 从下载到可用于 ChipDiffusion eval 的完整流程

---

## 1. 数据来源

| 项目 | 值 |
|------|---|
| 名称 | ISPD 2005 Placement Contest Benchmarks |
| 格式 | Bookshelf (.nodes, .nets, .pl, .scl, .aux, .wts) |
| 电路数 | 8 (adaptec1-4, bigblue1-4) |
| 下载源 | UTexas CERC Mirror |
| URL | `http://www.cerc.utexas.edu/~zixuan/ispd2005dp.tar.xz` |
| 文件大小 | 104MB (压缩), ~500MB (解压) |

### 为什么不用 IBM ICCAD04

ChipDiffusion 论文同时报告了 IBM ICCAD04 和 ISPD2005 的结果。我们尝试了 ICCAD04 但无法获取数据:

- **官方源** `vlsicad.eecs.umich.edu` 服务器离线，原始 DEF/LEF 文件无法下载
- **TILOS MacroPlacement** 提供了 ICCAD04 的 protobuf 格式，但这是 pre-clustered 数据（标准 cell 已被聚类成 soft macro），与 ChipDiffusion 需要的原始 DEF/LEF（含完整标准 cell）不在同一抽象层级
- TILOS 的 `ProtobufToLEFDEF.py` 转换器因为 protobuf 中缺少 `stdcell` 类型节点而报错

因此使用 ISPD2005 Bookshelf 格式作为替代，ChipDiffusion 完整支持此格式。

---

## 2. 下载

```bash
mkdir -p benchmarks/ispd2005
wget -q 'http://www.cerc.utexas.edu/~zixuan/ispd2005dp.tar.xz' -O /tmp/ispd2005dp.tar.xz

# 解压 (需要 pyunpack + patool)
pip install pyunpack patool
python -c "from pyunpack import Archive; Archive('/tmp/ispd2005dp.tar.xz').extractall('benchmarks/ispd2005/')"
```

解压后目录结构:

```
benchmarks/ispd2005/ispd2005/
├── adaptec1/
│   ├── adaptec1.aux
│   ├── adaptec1.nodes      # 元件名称 + 尺寸
│   ├── adaptec1.nets       # 网表连接关系
│   ├── adaptec1.pl         # 元件放置坐标
│   ├── adaptec1.scl        # 行信息 (row structure)
│   ├── adaptec1.wts        # 权重 (unused)
│   ├── adaptec1.dp.aux
│   ├── adaptec1.eplace.aux
│   ├── adaptec1.eplace-ip.pl
│   └── adaptec1.lg.pl
├── adaptec2/
├── adaptec3/
├── adaptec4/
├── bigblue1/
├── bigblue2/
├── bigblue3/
└── bigblue4/
```

---

## 3. Bookshelf 格式说明

### .nodes — 元件定义

```
UCLA nodes 1.0
NumNodes : 211447
NumTerminals : 543

o0  72  72
o1  72  72
...
p0  0  0  terminal    # terminal = macro/fixed block
```

每行: `<name> <width> <height> [terminal]`

### .nets — 网表

```
UCLA nets 1.0
NumNets : 221142
NumPins : 644785

NetDegree : 2 n0
  o100  I : 0.0000 0.0000
  o200  O : 36.0000 0.0000
```

每个 net: `NetDegree : <pin_count> <net_name>`，后跟 pin 列表，每个 pin 有 `<obj_name> <I/O> : <x_offset> <y_offset>`

### .pl — 放置坐标

```
UCLA pl 1.0
o0  459  5279  : N
o1  459  5351  : N
...
p0  0  5340  : N /FIXED
```

每行: `<name> <x> <y> : <orientation> [/FIXED]`

---

## 4. 解析 (Bookshelf → PyG Pickle)

ChipDiffusion 需要的格式是 PyTorch Geometric 的 pickle 文件。解析函数来自 ChipDiffusion 的 `notebooks/parse_bookshelf.ipynb`。

### 4.1 芯片尺寸 (硬编码)

ISPD2005 的芯片边界不在 Bookshelf 文件中，需要手动指定:

```python
ISPD_CHIP_SIZES = {
    0: [0.459, 0.459, 0.459 + 10692/1000, 0.459 + 12*890/1000],  # adaptec1
    1: [0.609, 0.616, 0.609 + 14054/1000, 0.616 + 12*1170/1000], # adaptec2
    2: [0.036, 0.058, 0.036 + 23190/1000, 23386/1000],           # adaptec3
    3: [0.036, 0.058, 0.036 + 23190/1000, 23386/1000],           # adaptec4
    4: [0.459, 0.459, 0.459 + 10692/1000, 11139/1000],           # bigblue1
    5: [0.036, 0.076, 0.036 + 18690/1000, 18868/1000],           # bigblue2
    6: [0.036, 0.076, 0.036 + 27690/1000, 27868/1000],           # bigblue3
    7: [0.036, 0.058, 0.036 + 32190/1000, 32386/1000],           # bigblue4
}
```

格式: `[x_start, y_start, x_end, y_end]`，单位已除以 1000 (Bookshelf 坐标单位为纳米级，除 1000 转微米级)。

### 4.2 解析流程

`parse_bookshelf(nodes_path, nets_path, pl_path)` 做以下事情:

1. **读 .pl** — 提取每个元件的 (x, y) 坐标和 /FIXED 标记
2. **读 .nodes** — 提取每个元件的 (width, height) 和 terminal 标记
3. **读 .nets** — 提取网表连接，每个 net 有一个 output pin 和多个 input pin，记录 pin offset
4. **过滤** — 删除 degree < 2 的 net (单 pin net 无意义)
5. **构建边** — 每个 net 的 output→input 连接转为有向边，edge_attr 记录 pin 偏移量
6. **双向化** — `flip_stack` 将单向边 (u→v) 复制为双向 (u→v, v→u)
7. **缩放** — 所有坐标除以 SCALING_UNITS=1000

### 4.3 输出格式

每个电路生成两个 pickle 文件:

**graph{idx}.pickle** — PyG `Data` 对象:
```
Data(
    x:                  (V, 2)  float   # 元件尺寸 (width, height)，已缩放
    edge_index:         (2, 2E) int64   # 双向边索引
    edge_attr:          (2E, 4) float   # pin 偏移 (src_dx, src_dy, dst_dx, dst_dy)，已缩放
    edge_pin_id:        (2E, 2) int64   # pin 唯一 ID
    is_ports:           (V,)    bool    # ISPD 全为 False
    is_macros:          (V,)    bool    # terminal 标记
    name_index_mapping: dict            # 元件名 → 索引
    chip_size:          list[4]         # [x_start, y_start, x_end, y_end]
)
```

**output{idx}.pickle** — placement 坐标:
```
Tensor (V, 2) float   # 每个元件的 (x, y) 放置坐标，已缩放
```

### 4.4 完整解析代码

```python
import os, pickle, re, torch
from torch_geometric.data import Data

# (parse_bookshelf 函数定义见 VSR_Place_Final_v2.ipynb Cell 4)

benchmark_path = 'benchmarks/ispd2005/ispd2005'
output_dir = 'datasets/graph/ispd2005'
os.makedirs(output_dir, exist_ok=True)

ispd_names = [f'adaptec{i+1}' for i in range(4)] + [f'bigblue{i+1}' for i in range(4)]
for idx, name in enumerate(ispd_names):
    x, cond = parse_bookshelf(
        os.path.join(benchmark_path, name, f'{name}.nodes'),
        os.path.join(benchmark_path, name, f'{name}.nets'),
        os.path.join(benchmark_path, name, f'{name}.pl'),
    )
    cond.chip_size = ISPD_CHIP_SIZES[idx]
    with open(os.path.join(output_dir, f'graph{idx}.pickle'), 'wb') as f:
        pickle.dump(cond, f)
    with open(os.path.join(output_dir, f'output{idx}.pickle'), 'wb') as f:
        pickle.dump(x, f)
```

### 4.5 Config 文件

解析完成后需要在输出目录放一个 `config.yaml`:

```yaml
train_samples: 0
val_samples: 8
scale: 1
```

---

## 5. 电路规模统计

| 电路 | 总元件 | Macro/Terminal | 标准 Cell | 网表 | 边 (双向) |
|------|--------|---------------|-----------|------|----------|
| adaptec1 | 211,447 | 543 | 210,904 | ~221K | ~14K (macro) |
| adaptec2 | 255,023 | 566 | 254,457 | ~266K | ~19K (macro) |
| adaptec3 | 451,650 | 723 | 450,927 | ~469K | ~23K (macro) |
| adaptec4 | 496,045 | 1,329 | 494,716 | ~516K | ~29K (macro) |
| bigblue1 | 278,164 | 560 | 277,604 | ~284K | ~6K (macro) |
| bigblue2 | 557,866 | 23,084 | 534,782 | ~577K | ~144K (macro) |
| bigblue3 | 1,096,812 | 1,298 | 1,095,514 | ~1.1M | ~33K (macro) |
| bigblue4 | 2,177,353 | 8,170 | 2,169,183 | ~2.2M | ~223K (macro) |

注: `eval_macro_only` 模式下只处理 macro/terminal 元件，标准 cell 被忽略。边数为双向后的 macro-only 子图边数。

---

## 6. ChipDiffusion Eval 命令

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python diffusion/eval.py \
    method=eval_macro_only \
    task=ispd2005 \
    from_checkpoint=large-v2/large-v2.ckpt \
    "legalizer@_global_=opt-adam" \
    "guidance@_global_=opt" \
    num_output_samples=8 \
    model.grad_descent_steps=20 \
    model.hpwl_guidance_weight=0.0016 \
    legalization.alpha_lr=0.008 \
    legalization.hpwl_weight=0.00012 \
    legalization.legality_potential_target=0 \
    legalization.grad_descent_steps=20000 \
    macros_only=True \
    logger.wandb=False
```

参数说明:
- `method=eval_macro_only`: 只评估 macro 放置
- `task=ispd2005`: 从 `datasets/graph/ispd2005/` 加载数据
- `from_checkpoint=large-v2/large-v2.ckpt`: 预训练权重 (相对于 `logs/`)
- `guidance@_global_=opt`: 使用优化引导采样
- `legalizer@_global_=opt-adam`: 使用 Adam 优化器做合法化
- `legalization.grad_descent_steps=20000`: 合法化梯度下降步数
- `num_output_samples=8`: 8 个电路各跑一次
- `macros_only=True`: 只处理 macro，跳过标准 cell

---

## 7. 踩坑记录

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `parse_bookshelf` lambda 报错 | notebook 中 `lambda x: 0` 只接受 1 个参数，但 print 传多个 | 改为 `lambda *a, **kw: None` |
| Hydra 解析 `16e-4` 报错 | Hydra CLI 不支持科学记号 | 改为小数 `0.0016` |
| `performer-pytorch` 装不上 | 不兼容 Python 3.12 | 跳过，eval 不需要 |
| `scikit-learn` 在 AutoDL 装不上 | 阿里云 PyPI 镜像缺包 | 换 pypi.org 源或用 Colab |
| IBM ICCAD04 不可用 | 官方源离线 + TILOS 数据层级不匹配 | 使用 ISPD2005 替代 |
| Colab 断连丢数据 | 免费版 session 超时 | nohup 后台跑 + 日志写 Google Drive |
| ChipDiffusion checkpoint 路径嵌套 | gdown 下载目录嵌套一层 | 用 `gdown.download()` 指定精确路径 |
