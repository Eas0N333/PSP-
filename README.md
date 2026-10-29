# PSP 复现工程

PSP: Harnessing Position and Shape Priors for Cross-Domain Few-Shot Medical Image Segmentation(MICCAI 2026)。

本仓库目录:

| 路径 | 内容 |
| --- | --- |
| `PSP-main/` | 论文官方代码(`xubin471/PSP`)+ 本仓库的修改;完整介绍见其内部 README |
| `PSP-main/supervoxels/` | 从 ADNet 移植的 supervoxel 生成代码,官方仓库未包含但训练需要 |
| `scripts/dl_thread.py` | 多线程分块下载器(Zenodo 等慢速源用,支持断点续传) |
| `DATASETS.md` | 数据集清单、预处理目录结构、环境说明、运行方式 |
| `requirements.txt` | Python 依赖 |

## 相对官方代码的改动

1. `dataloader/image_transforms.py`:`from collections import Sequence` → `collections.abc`(适配 Python 3.10+)
2. 新增 `supervoxels/`:补齐 `supervoxels_5000` / `supervoxels_1000` 生成流程
3. 新增 `scripts/dl_thread.py`、`DATASETS.md`、`requirements.txt`

## 环境

- Python 3.12 + torch 2.4.1+cu121 + torchvision 0.19.1+cu121(RTX 2080 Ti 验证通过)
- 其余依赖见 `requirements.txt`
- 预训练权重 `deeplabv3_resnet50_coco-cd0a2569.pth` 放到 `PSP-main/checkpoint/`(已 gitignore)

## 未包含在本仓库中的内容

- 原始数据集(CHAOS / BTCV / MS-CMRSeg),按各官方渠道单独获取
- 预训练权重(GitHub 单文件上限,自行从 PyTorch 官方下载)
- 训练日志与}hooks

## 许可与致谢

`PSP-main/` 下的代码版权归原论文作者(Bin Xu 等,南京理工大学)所有,本仓库仅用于复现实验与学习。
若用于发表请引用原文。`supervoxels/` 来自 ADNet(Hansen 等,MIA 2022)。
> 注:官方仓库未附带 LICENSE,若需公开引用或二次分发,请先联系原作者获取授权。
