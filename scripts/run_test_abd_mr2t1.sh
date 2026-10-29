#!/bin/bash
# 跨序列评估:用 T2-SPIR 上训练的权重,在 T1-DUAL InPhase 上测试
# 对应论文里的"跨域"设定(此处域差异来自序列而非模态,因为只用了 CHAOS)
#
# 用法:
#   bash scripts/run_test_abd_mr2t1.sh        # 全部 5 折
#   bash scripts/run_test_abd_mr2t1.sh 0      # 只测第 0 折

set -e
PY="C:/Users/Easonpc/psp-python312/python.exe"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/PSP-main"

FOLDS=${1:-"0 1 2 3 4"}
for EVAL_FOLD in $FOLDS; do
  echo "================ fold $EVAL_FOLD ================"
  CKPT="./exps_on_ABDOMEN_MR/CDFS_train_ABDOMEN_MR_cv${EVAL_FOLD}/1/snapshots/30000.pth"
  if [ ! -f "$CKPT" ]; then
    echo "缺少权重 $CKPT,先跑 run_train_abd_mr.sh"
    exit 1
  fi
  "$PY" -u test.py with \
    mode='test' \
    dataset='ABDOMEN_MR_T1' \
    num_workers=0 \
    n_steps=50000 \
    max_iters_per_load=5000 \
    eval_fold=$EVAL_FOLD \
    supp_idx=2 \
    test_label=[1,2,3,4] \
    seed=2026 \
    n_part=3 \
    reload_model_path="$CKPT" \
    save_snapshot_every=1000 \
    lr_step_gamma=0.98 \
    path.log_dir=./result_MR_2_MR_T1 \
    "path.ABDOMEN_MR_T1.data_dir=../data/ABD/ABDOMEN_MR_T1"
done

echo "================ 5 折汇总 ================"
awk 'NF {sum+=$1; n++} END {printf "mean_dice: %.4f (n=%d folds)\n", sum/n, n}' results.txt
