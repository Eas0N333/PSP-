# -*- coding: utf-8 -*-
"""把 CHAOS 原始数据转成 PSP 需要的整卷 .nii.gz 格式。

输入(Train_Sets 目录,来自 CHAOS_Train_Sets.zip):
    MR/<case>/T2SPIR/DICOM_anon/*.dcm  +  MR/<case>/T2SPIR/Ground/*.png
    MR/<case>/T1DUAL/DICOM_anon/InPhase/*.dcm + MR/<case>/T1DUAL/Ground/*.png

输出(每个域一个目录):
    <out>/chaos_MR_T2_normalized/image_1.nii.gz ... image_20.nii.gz
    <out>/chaos_MR_T2_normalized/label_1.nii.gz ...
    <out>/chaos_MR_T2_normalized/supervoxels_5000/superpix-MIDDLE_1.nii.gz ...

标签映射(CHAOS 原始像素值 → PSP 的 ABDOMEN_MR 连续标签):
    63 -> 1 (LIVER)   126 -> 2 (RIGHT_KIDNEY)
    189 -> 3 (LEFT_KIDNEY)   252 -> 4 (SPLEEN)

用法:
    python scripts/prepare_chaos.py --src data/raw/Train_Sets/MR --dry-run
    python scripts/prepare_chaos.py --src data/raw/Train_Sets/MR
"""
import argparse
import glob
import os
import re
import sys

import cv2
import numpy as np
import SimpleITK as sitk

LABEL_MAP = {63: 1, 126: 2, 189: 3, 252: 4}
SIZE = 256
N_SV = 5000


def read_dicom_volume(dicom_dir):
    """按 DICOM 序列读成 (z, y, x)。"""
    names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_dir)
    if not names:
        raise RuntimeError("no dicom in " + dicom_dir)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(names)
    vol = sitk.GetArrayFromImage(reader.Execute())  # (z, y, x)
    return vol, len(names)


def read_mask_volume(ground_dir, n_slices):
    """读 CHAOS 的逐切片 PNG 标注,按文件名排序后堆叠成 (z, y, x)。"""
    files = sorted(glob.glob(os.path.join(ground_dir, "*.png")))
    if len(files) != n_slices:
        raise RuntimeError(
            "切片数不一致: dicom={} png={} ({})".format(n_slices, len(files), ground_dir)
        )
    masks = []
    for f in files:
        m = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if m is None:
            raise RuntimeError("读不到 " + f)
        if m.ndim == 3:
            m = m[..., 0]
        masks.append(m)
    return np.stack(masks, axis=0)


def remap_labels(mask):
    out = np.zeros(mask.shape, dtype=np.uint8)
    for raw, new in LABEL_MAP.items():
        out[mask == raw] = new
    return out


def resize_volume(vol, is_label):
    """逐切片重采样到 SIZE×SIZE。"""
    interp = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
    out = np.empty((vol.shape[0], SIZE, SIZE), dtype=vol.dtype)
    for z in range(vol.shape[0]):
        out[z] = cv2.resize(vol[z], (SIZE, SIZE), interpolation=interp)
    return out


def make_supervoxels(img_vol, n_segments=N_SV):
    """生成超体素(替代 ADNet 的 felzenszwalb_3d,本机无 C 编译器)。

    照抄 ADNet generate_supervoxels.py 的后处理约定:
      1. 强度归一化到 0-255 后做分割
      2. 逐层取前景掩码(强度 > 10),取最大连通域并填洞
      3. 背景区域统一置 0 —— 训练代码里 `unique.remove(0)` 要求存在 0 类
    """
    from scipy.ndimage import binary_fill_holes
    from skimage.measure import label as sklabel
    from skimage.segmentation import slic

    v = img_vol.astype(np.float32)
    v = 255.0 * (v - v.min()) / max(float(v.ptp()), 1e-6)

    # 段数按"每段约 8000 体素"折算;compactness 取很小值,让区域跟随强度
    # (ADNet 的 felzenszwalb_3d 同样是强度驱动的合并,compactness 大反而切得太碎)
    n_seg = int(np.clip(img_vol.size // 8000, 120, 300))
    seg = slic(
        v,
        n_segments=n_seg,
        compactness=0.1,
        channel_axis=None,
        start_label=1,
        enforce_connectivity=True,
        sigma=1,
    )

    fg = np.zeros(seg.shape, dtype=bool)
    for z in range(v.shape[0]):
        m = v[z] > 10
        if m.max() < 0.999:  # 整层无前景
            fg[z] = m
            continue
        lb = sklabel(m)
        if lb.max() == 0:
            fg[z] = m
            continue
        counts = np.bincount(lb.ravel())[1:]
        largest = lb == (int(np.argmax(counts)) + 1)
        fg[z] = binary_fill_holes(largest)

    seg = np.int32(seg)
    seg[seg == 0] = seg.max() + 1
    seg[~fg] = 0
    return seg


def save_nii(arr, path):
    img = sitk.GetImageFromArray(arr)
    sitk.WriteImage(img, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/Train_Sets/MR")
    ap.add_argument("--out", default="data/ABD")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sv-only", action="store_true", help="只重算 supervoxels(图像/标签已存在时)")
    args = ap.parse_args()

    cases = sorted(
        [d for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d))],
        key=lambda s: int(s),
    )
    print("发现 {} 个病例: {}".format(len(cases), cases), flush=True)

    # 两个域:T2SPIR(论文的 Abdomen MR) + T1DUAL InPhase(额外的第二域,做跨序列)
    doms = [
        ("ABDOMEN_MR", "chaos_MR_T2_normalized", "T2SPIR", "DICOM_anon", "Ground"),
        ("ABDOMEN_MR_T1", "chaos_MR_T1_normalized", "T1DUAL", "DICOM_anon/InPhase", "Ground"),
    ]

    for dom_name, sub, seq, dcm_rel, gnd_rel in doms:
        root = os.path.join(args.out, dom_name, sub)
        # 注意:supervoxels_<n_sv> 必须与 *_normalized 目录并列(代码在 data_dir 下 glob)
        sv_dir = os.path.join(args.out, dom_name, "supervoxels_{}".format(N_SV))
        print("\n=== {} -> {} ===".format(seq, root), flush=True)
        if not args.dry_run:
            os.makedirs(root, exist_ok=True)
            os.makedirs(sv_dir, exist_ok=True)

        ok = 0
        for idx, case in enumerate(cases, start=1):
            cdir = os.path.join(args.src, case, seq)
            dcm_dir = os.path.join(cdir, dcm_rel)
            gnd_dir = os.path.join(cdir, gnd_rel)
            if not os.path.isdir(dcm_dir):
                print("  [skip] {} 缺 {}".format(case, dcm_rel), flush=True)
                continue
            if args.sv_only:
                img_path = os.path.join(root, "image_{}.nii.gz".format(idx))
                if not os.path.exists(img_path):
                    print("  [skip] 缺 {}".format(img_path), flush=True)
                    continue
                vol = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)
                sv = make_supervoxels(vol)
                save_nii(sv, os.path.join(sv_dir, "superpix-MIDDLE_{}.nii.gz".format(idx)))
                ok += 1
                print("  case {:>2} -> supervoxels={} 背景占比 {:.3f}".format(
                    case, int(sv.max()), float((sv == 0).mean())), flush=True)
                continue
            try:
                vol, n = read_dicom_volume(dcm_dir)
                msk = read_mask_volume(gnd_dir, n)
            except Exception as e:  # noqa: BLE001
                print("  [fail] case {}: {}".format(case, e), flush=True)
                continue

            if vol.shape != msk.shape:
                print("  [warn] case {} 形状不一致 {} vs {}".format(case, vol.shape, msk.shape), flush=True)
                continue

            vol = resize_volume(vol.astype(np.float32), is_label=False)
            msk = resize_volume(msk, is_label=True)
            lbl = remap_labels(msk)

            organs = np.unique(lbl)
            organs = organs[organs > 0]
            if args.dry_run:
                print(
                    "  case {:>2} -> image_{}.nii.gz  vol{}  organs{}".format(
                        case, idx, vol.shape, organs.tolist()
                    ),
                    flush=True,
                )
                continue

            save_nii(vol, os.path.join(root, "image_{}.nii.gz".format(idx)))
            save_nii(lbl, os.path.join(root, "label_{}.nii.gz".format(idx)))

            sv = make_supervoxels(vol)
            save_nii(
                sv, os.path.join(sv_dir, "superpix-MIDDLE_{}.nii.gz".format(idx))
            )
            ok += 1
            print(
                "  case {:>2} -> {} 器官{} supervoxels={}".format(
                    case, idx, organs.tolist(), int(sv.max())
                ),
                flush=True,
            )
        print("  完成 {} 例".format(ok), flush=True)

    print("\n全部完成", flush=True)


if __name__ == "__main__":
    main()
