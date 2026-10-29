# PSP 复现:代码 + 数据集清单

论文:*PSP: Harnessing Position and Shape Priors for Cross-Domain Few-Shot Medical Image Segmentation*(MICCAI 2026)
官方代码:https://github.com/xubin471/PSP

---

## 一、代码(已扒下来)

| 项目 | 位置 | 说明 |
| --- | --- | --- |
| PSP 主仓库 | `E:/workplace/psp/PSP-main/` | `main` 分支完整源码(35 个文件) |
| supervoxel 生成 | `E:/workplace/psp/PSP-main/supervoxels/` | 主仓库**没有**这块,从 ADNet 补齐(生成 `supervoxels_5000`) |
| 多线程下载器 | `E:/workplace/psp/scripts/dl_thread.py` | 自写,Zenodo 单线程只有 ~20KB/s,48 线程可到 2MB/s |

核心模块对应关系:

- `models/PCE.py` → Position Coordinate Embedding(极坐标位置编码 + SA + 动态残差 β)
- `models/SPM.py` → Shape Prototype Modulation(SDM 几何统计 + 傅里叶描述子, `Nspec=15`)
- `models/fewshot.py` → Hybrid-Prototype Prediction(coarse-to-fine + cross-attention + 自适应阈值 τ)
- `models/ca.py` → Cross-Attention
- `train.py` / `test.py` + `scripts/*.sh` → 6 个跨域任务的训练/推理脚本
- 主干:ResNet-50(需 `checkpoint/deeplabv3_resnet50_coco-cd0a2569.pth`)

---

## 二、需要下载的数据集

论文用 **2 组跨域 benchmark(共 4 个数据源)**,代码还额外支持前列腺(论文未报告)。

| # | 数据集 | 角色 | 内容 | 大小 | 来源 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **CHAOS**(T2-SPIR) | 腹部 MRI 域 | 20 例 3D T2-SPIR MRI + 标注 | 890MB(zip)/1.7G(解压) | [Zenodo 3431873](https://zenodo.org/records/3431873)(开放) | ✅ **已下载并解压** → `data/raw/Train_Sets/MR/` |
| 2 | **BTCV / MICCAI 2015 Multi-Atlas** | 腹部 CT 域 | 30 例腹部 CT(用 20 例),13 器官标注 | ~1.1GB | [Synapse syn3193805](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292) | ❌ **需注册账号**(见下) |
| 3 | **MS-CMRSeg 2019** | 心脏 LGE 域 | 45 例 LGE MRI | ~5GB | [zmiclab](https://zmiclab.github.io/zxh/0/mscmrseg19/index.html) 表单 → Google Drive | ❌ **需填表注册 + GDrive 不通** |
| 4 | **MS-CMRSeg 2019** | 心脏 b-SSFP 域 | 45 例 b-SSFP cine MRI | 同上 | 同上 | ❌ 同上 |
| 5 | Prostate UCLH / NCI *(可选)* | 前列腺 | 论文未使用,代码有脚本 | 3.3GB | [Zenodo 7013610](https://zenodo.org/records/7013610)(开放) | ⏸ 未下(非必需) |

分割目标:
- 腹部(CT↔MR):Liver、Spleen、Right Kidney、Left Kidney
- 心脏(LGE↔b-SSFP):LV-BP、LV-MYO、RV

### 为什么 2 和 3/4 我没法直接下

- **BTCV**:Synapse 必须登录账号并接受数据使用协议才能拿 `RawData.zip`,无匿名直链,我不能替你注册。
  → 你注册后:`https://www.synapse.org/#!Synapse:syn3193805/files/` 下载 `Abdomen/RawData.zip`;
  或给我一个 Synapse Personal Access Token,我可以用 `synapseclient` 自动拉。
- **MS-CMRSeg**:官方流程是填 Google 表单 → 拿 Google Drive 链接。当前环境 **Google Drive / HuggingFace 域名不可达**,
  即使拿到链接也下不了。→ 需要你在本地浏览器下载后放进 `data/raw/`。

---

## 三、数据目录结构与预处理

代码不吃原始数据,要的是**预处理后的 2D npy 切片**,放在 `PSP-main/data/` 下(见 `config.py` 的 `path` 字段):

```
PSP-main/data/
├── ABD/
│   ├── ABDOMEN_CT/
│   │   ├── sabs_CT_normalized/{image*.npy, label*.npy}   ← BTCV
│   │   └── supervoxels_5000/super*.npy
│   └── ABDOMEN_MR/
│       ├── chaos_MR_T2_normalized/{image*.npy, label*.npy} ← CHAOS
│       └── supervoxels_5000/super*.npy
├── Cardiac/
│   ├── LGE/cmr_LGE_normalized/ + supervoxels_1000/
│   └── bSSFP/cmr_bssFP_normalized/ + supervoxels_1000/
└── Prostate/{NCI,UCLH}/...
```

预处理流程遵循 Ouyang et al. (SSL-ALPNet),README 指定的参考实现:
1. DICOM → nii(`dcm2nii` / SimpleITK)
2. 逐病例归一化、重采样、转 2D 切片存 `image_*.npy` / `label_*.npy`
3. 用 ADNet 的 supervoxel 代码生成 `supervoxels_<n_sv>`(`sabs/chaos` 用 5000,Cardiac 用 1000):
   ```bash
   cd PSP-main/supervoxels && python setup.py build_ext --inplace   # cython 编译
   python generate_supervoxels.py                                    # 改路径后运行
   ```

已验证的 CHAOS MR 标注像素值(供写预处理映射用):
`63=LIVER, 126=RIGHT_KIDNEY, 189=LEFT_KIDNEY, 252=SPLEEN` → 需重映射到 PSP 的 `1=LIVER, 2=RK, 3=LK, 4=SPLEEN`。

---

## 四、环境(已配好 ✅)

| 项 | 值 |
| --- | --- |
| 解释器 | `C:\Users\Easonpc\psp-python312\python.exe`(便携版 Python 3.12.9) |
| torch | 2.4.1+cu121 ✅ GPU 可用 |
| torchvision | 0.19.1+cu121 |
| GPU | RTX 2080 Ti,23.6GB(驱动 591.86 / CUDA 13.1) |
| 其他 | numpy 1.26.4、matplotlib 3.7.5、scipy、SimpleITK 2.5.6、opencv 4.11、scikit-image、nibabel、sacred 0.8.7、yacs |
| 预训练权重 | `PSP-main/checkpoint/deeplabv3_resnet50_coco-cd0a2569.pth` ✅ 已下 |
| 依赖清单 | `requirements.txt` |

冒烟测试已过:`models/fewshot.py` 前向(train + test 模式)在 GPU 上跑通,63.8M 参数,峰值显存 2.52GB。

> 注:便携 Python 不自带 `distutils` 时代的自动 PATH,`PSP-main` 下跑脚本时要保证 cwd 在 sys.path。
> torch CUDA 轮子 PyPI 只给 CPU 版,需从 PyTorch 官方 / 上海交大镜像取 `+cu121` 本地轮子安装。
> 兼容性修复:`dataloader/image_transforms.py` 的 `from collections import Sequence` 已改为 `collections.abc`。

网络:`github.com` 网页端、**Google Drive、HuggingFace 不可达**;`api.github.com`、`codeload.github.com`、
`zenodo.org`(多线程)、`pypi.tuna.tsinghua.edu.cn`、`mirror.sjtu.edu.cn`(~79MB/s)可用。

---

## 五、下一步

1. 补齐 BTCV 与 MS-CMRSeg(需你注册;给 token 或本地下载都可以)
2. ~~搭 Python 环境~~ ✅ 已完成(详见第四节)
3. ~~下载 `deeplabv3_resnet50_coco-cd0a2569.pth` 到 `PSP-main/checkpoint/`~~ ✅ 已完成
4. 写/跑预处理脚本,产出 `*_normalized` 与 `supervoxels_5000`(**唯一还缺的一步**)
5. `./scripts/train_on_ABDOMEN_MR.sh` → `./scripts/test_on_ABDOMEN_MR2CT.sh`(论文 Abd-MR→CT: 68.25 Dice)

> 临时替代方案:只用 CHAOS 也能先自洽跑「跨序列」(MR T2-SPIR ↔ T1-DUAL InPhase,两者都带 4 器官标注),
> 不需额外数据。论文的 Abd CT↔MR 则必须补 BTCV(CHAOS 的 CT 只有肝脏标注)。

## 运行方式

便携 python 没有自动 PATH,直接这样跑(在 `PSP-main/` 下):

```bash
C:/Users/Easonpc/psp-python312/python.exe train.py with mode='train' dataset='ABDOMEN_MR' ...
```

或参考 `scripts/train_on_ABDOMEN_MR.sh` 里的完整参数列表。
