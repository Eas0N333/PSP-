# CHAOS 数据准备与复现结果

> 本文档记录用 CHAOS 单数据集跑通 PSP 全流程的过程。
> **重要更正**:`DATASETS.md` 早期版本写的 `image*.npy` 是错的,代码实际读**整卷 `.nii.gz`**。

---

## 一、代码真正要求的磁盘格式

`dataloader/datasets.py:68` 的排序键是 `int(x.split('_')[-1].split('.nii.gz')[0])`,
`datasets.py:95` 用 `sitk.ReadImage` 读图,所以:

```
data_dir/                                  # config.py path[dataset]['data_dir']
├── chaos_MR_T2_normalized/
│   ├── image_1.nii.gz ... image_20.nii.gz   # 整卷 3D,不是 2D 切片
│   └── label_1.nii.gz ... label_20.nii.gz   # 与 image 一一对应,文件名后缀数字一致
└── supervoxels_5000/                        # 注意:与 normalized 目录并列,不在其内部
    └── superpix-MIDDLE_1.nii.gz ...
```

- 文件名必须形如 `*_<整数>.nii.gz`(排序靠 `int()` 解析,否则 ValueError)
- `n_sv` 决定目录名 `supervoxels_<n_sv>`;`config.py` 默认 MR 用 5000
- 尺寸必须已经是 256×256(`TestDataset.resize_array` 是空实现),本仓库预处理已保证
- 训练时整卷 z-score 归一化在 `datasets.py:369` 内部完成,所以磁盘上存**原始强度**即可

---

## 二、预处理脚本

```bash
python scripts/prepare_chaos.py --src data/raw/Train_Sets/MR --dry-run   # 先看
python scripts/prepare_chaos.py --src data/raw/Train_Sets/MR            # 全量(约 1 分钟)
python scripts/prepare_chaos.py --src data/raw/Train_Sets/MR --sv-only  # 只重算超体素
```

做三件事:

1. **DICOM → 整卷 nii.gz**:`SimpleITK.ImageSeriesReader` 按序列读入,逐层 `cv2.resize` 到 256×256
   (图像双线性、标注最近邻),`sitk.WriteImage` 落盘。
2. **标签重映射**:CHAOS 原始像素值 `63/126/189/252` → `1/2/3/4`
   (LIVER / RIGHT_KIDNEY / LEFT_KIDNEY / SPLEEN,与 `get_label_names('ABDOMEN_MR')` 一致)。
3. **超体素**:见下节。

产出两个域(各 20 例):

| 目录 | 序列 | 用途 |
| --- | --- | --- |
| `data/ABD/ABDOMEN_MR/chaos_MR_T2_normalized` | T2-SPIR | 论文的腹部 MR 域(源域) |
| `data/ABD/ABDOMEN_MR_T1/chaos_MR_T1_normalized` | T1-DUAL InPhase | 我额外加的第二域,做跨序列迁移 |

---

## 三、超体素的替代实现(与论文的差异)

官方/ADNet 用 `supervoxels/felzenszwalb_3d.pyx`(Cython)生成超体素,**需要 C 编译器**。
本机没有 MSVC/cython,无法编译,因此改用 `skimage.segmentation.slic` 的 3D 版本,并严格复刻
ADNet `generate_supervoxels.py` 的后处理约定:

1. 强度归一化到 0-255
2. 逐层前景掩码:强度 > 10 → 取最大连通域 → `binary_fill_holes`
3. `seg[seg == 0] = seg.max() + 1`(避免 0 冲突)
4. `seg[~fg] = 0` —— 背景统一为 0

第 4 步是必须的:训练代码 `datasets.py:387` 有 `unique.remove(0)`,要求存在 0 类。

参数经实测选定:`n_segments = clip(size//8000, 120, 300)`、`compactness=0.1`、`sigma=1`。
compactness 必须小:默认的 10 会把区域切成小块,单层内最多 64 像素,
撑不到 `min_size=500`,采样循环会**死循环**(这是调试时踩到的坑)。
当前设置下每例可用类 33–67 个,单层最大 700–1750 像素,采样正常。

> 若日后装了 Visual Studio Build Tools,可以编译 `supervoxels/` 下的 cython 版还原官方实现:
> `cd PSP-main/supervoxels && python setup.py build_ext --inplace`

---

## 四、代码改动清单

| 文件 | 改动 |
| --- | --- |
| `dataloader/image_transforms.py:6` | `from collections import Sequence` → `collections.abc`(Python 3.10+) |
| `dataloader/datasets.py` | 两处 if 链各加 `ABDOMEN_MR_T1` 分支 → `chaos_MR_T1_normalized` |
| `dataloader/dataset_specifics.py` | `get_label_names` / `get_folds` 各加 `ABDOMEN_MR_T1`(划分与 ABDOMEN_MR 一致) |
| `config.py` | `path` 表加 `ABDOMEN_MR_T1`;新增 `start_iter` / `iter_budget`(默认 0,关闭) |
| `train.py` | 支持断点续训:读 `reload_model_path`(可为 `resume_*.pth`,含模型+优化器+步数)、按 `start_iter` 恢复 lr、每块末尾补存一份续训快照。默认参数下行为与官方一致 |
| `supervoxels/` | 从 ADNet 补齐(原仓库缺失) |

> 续训补丁的动机:单次会话对单条命令有时长上限,而完整训练需 2 小时以上。
> 分块训练用 `scripts/run_chunk.sh` 驱动,每块 1700 步。
> 注意:分块训练**不是**逐 bit 复现(数据采样 RNG 未跨块保存),只保证优化轨迹连续。

---

## 五、已跑通的结果

### 冒烟测试

- `models/fewshot.py` 前向:train + test 模式均通过,63.8M 参数,峰值显存 2.4 GB
- 训练 200 步:loss 9.46 → 0.18(伪标签来自超体素,收敛很快是正常的)

### 短程实验(1500 步,约 6.5 分钟)

源域 T2-SPIR `fold0` 训练 → 目标域 T1-DUAL 评估(单折,`supp_idx=2`,`n_part=3`):

| 器官 | Dice |
| --- | --- |
| LIVER | 0.5991 |
| RIGHT_KIDNEY | 0.5334 |
| LEFT_KIDNEY | 0.5235 |
| SPLEEN | 0.3638 |
| **Mean** | **0.5050** |

只训练 1500 步(论文是 30001 步)就到 50.5%,说明链路是对的、方法在跨序列场景确实有效。
**这不是论文表格里可比的数字**(数据集/域对/训练量都不同),只作链路验证。

### 正式训练(fold 0,30630 步,约 2.4 小时)

源域 T2-SPIR `fold0`(`fold0` 的 5 例做测试集),按论文设置 SGD lr=1e-3 / gamma=0.98 / 每 1000 步存快照,
实际跑到 30630 步(分块驱动器最后一块略超 30001)。

**跨序列:T2-SPIR → T1-DUAL**(单折,`supp_idx=2`,`n_part=3`)

| 步数 | LIVER | RK | LK | SPLEEN | Mean |
| --- | --- | --- | --- | --- | --- |
| 5 000 | — | — | — | — | 0.5284 |
| 8 530 | 0.5767 | 0.5566 | 0.5752 | 0.4256 | 0.5335 |
| 15 000 | — | — | — | — | 0.5169 |
| 20 000 | — | — | — | — | 0.4917 |
| 25 000 | — | — | — | — | 0.5083 |
| 30 000 | 0.3728 | 0.4131 | 0.4649 | 0.4381 | 0.4222 |

**同域参照:MR(T2-SPIR)→ MR**,同一份 30000 权重:

| LIVER | RK | LK | SPLEEN | Mean |
| --- | --- | --- | --- | --- |
| 0.4147 | 0.7938 | 0.7606 | 0.6861 | **0.6638** |

几点解读:

1. 同域 0.664 vs 跨序列 ~0.50,跨域确实有掉点,符合论文的核心论点(域差异导致性能下降)。
2. 跨序列在 5k–25k 步稳定在 0.49–0.53,**30k 掉到 0.42**,像是往源域过度拟合。
   由于只跑了 1 折、`supp_idx` 固定为 2,单点波动也可能很大,不能据此下结论。
3. 与论文表格**不可直接比较**:论文是 Abd-CT↔Abd-MR(跨模态,真实数据 BTCV+CHAOS)、5 折平均;
   这里是 CHAOS 内部跨序列、单折、超体素用 SLIC 替代。仅作方法有效性验证。

### 正式实验脚本

```bash
bash scripts/run_train_abd_mr.sh        # 官方设置,5 折 × 30001 步(每折 ~2.4h)
bash scripts/run_test_abd_mr2t1.sh      # 跨序列评估 + 5 折汇总
bash scripts/run_chunk.sh 0 1700        # 分块续训(单次调用有时长上限时用)
```

---

## 六、还没做的

- **BTCV(腹部 CT)** 与 **MS-CMRSeg(心脏)** 仍缺,论文的 Abd-CT↔MR、Cardiac 两组实验无法复现
- 5 折正式训练未跑完(需长时间占用 GPU)
- 超体素是 SLIC 替代版,与官方 felzenszwalb_3d 存在实现差异,可能影响绝对数值
