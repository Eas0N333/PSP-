#!/bin/bash
# 快速验证:1500 步(约 7 分钟),用来确认代码/数据链路正常,不是论文设置的正式实验
# 用法: bash scripts/run_quick_sanity.sh
set -e
PY="C:/Users/Easonpc/psp-python312/python.exe"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/PSP-main"

"$PY" -u train.py with \
  mode='train' dataset='ABDOMEN_MR' num_workers=0 \
  n_steps=1501 max_iters_per_load=500 eval_fold=0 \
  test_label=[1,2,3,4] exclude_label=None use_gt=False seed=2026 \
  save_snapshot_every=1500 print_interval=100 lr_step_gamma=0.98 \
  path.log_dir=./exps_quick \
  "path.ABDOMEN_MR.data_dir=../data/ABD/ABDOMEN_MR"

echo "================ 跨序列评估(T2-SPIR -> T1-DUAL) ================"
"$PY" -u test.py with \
  mode='test' dataset='ABDOMEN_MR_T1' num_workers=0 \
  n_steps=50000 max_iters_per_load=5000 eval_fold=0 supp_idx=2 \
  test_label=[1,2,3,4] seed=2026 n_part=3 \
  reload_model_path='./exps_quick/CDFS_train_ABDOMEN_MR_cv0/1/snapshots/1500.pth' \
  save_snapshot_every=1000 lr_step_gamma=0.98 \
  path.log_dir=./result_quick \
  "path.ABDOMEN_MR_T1.data_dir=../data/ABD/ABDOMEN_MR_T1"
