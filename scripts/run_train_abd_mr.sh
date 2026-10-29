#!/bin/bash
# 正式训练:源域 CHAOS T2-SPIR(ABDOMEN_MR),5 折交叉验证
# 论文设置:n_steps=30001, max_iters_per_load=5000, SGD lr=1e-3, gamma=0.98, 每 1000 步存快照
# 单折约 2.5 小时(2080 Ti),5 折约 13 小时
#
# 用法(任意目录):
#   bash scripts/run_train_abd_mr.sh          # 全部 5 折
#   bash scripts/run_train_abd_mr.sh 0        # 只跑第 0 折

set -e
PY="C:/Users/Easonpc/psp-python312/python.exe"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/PSP-main"

FOLDS=${1:-"0 1 2 3 4"}
for EVAL_FOLD in $FOLDS; do
  echo "================ fold $EVAL_FOLD ================"
  "$PY" -u train.py with \
    mode='train' \
    dataset='ABDOMEN_MR' \
    num_workers=0 \
    n_steps=30001 \
    max_iters_per_load=5000 \
    eval_fold=$EVAL_FOLD \
    test_label=[1,2,3,4] \
    exclude_label=None \
    use_gt=False \
    seed=2026 \
    save_snapshot_every=1000 \
    print_interval=100 \
    lr_step_gamma=0.98 \
    path.log_dir=./exps_on_ABDOMEN_MR \
    "path.ABDOMEN_MR.data_dir=../data/ABD/ABDOMEN_MR"
done
