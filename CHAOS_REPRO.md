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
| `config.py` | `path` 表加 `ABDOMEN_MR_T1` |
| `supervoxels/` | 从 ADNet 补齐(原仓库缺失) |

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

### 正式实验

```bash
bash scripts/run_train_abd_mr.sh 0        # 单折,约 2.5 小时
bash scripts/run_train_abd_mr.sh          # 5 折,约 13 小时
bash scripts/run_test_abd_mr2t1.sh        # 评估 + 5 折汇总
```

耗时依据:实测约 0.27 s/iter(256×256、batch=1、2080 Ti),30000 iter ≈ 2.3 小时。

---

## 六、还没做的

- **BTCV(腹部 CT)** 与 **MS-CMRSeg(心脏)** 仍缺,论文的 Abd-CT↔MR、Cardiac 两组实验无法复现
- 5 折正式训练未跑完(需长时间占用 GPU)
- 超体素是 SLIC 替代版,与官方 felzenszwalb_3d 存在实现差异,可能影响绝对数值
