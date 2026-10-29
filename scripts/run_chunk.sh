#!/bin/bash
# 分块续训驱动器:跑一段固定步数后退出(单次调用有时长上限时用)
# 用法: bash scripts/run_chunk.sh <fold> [每块步数]
#
# sacred 每次运行都新建 run 目录(1/、2/、3/…),快照散落各处,
# 所以这里跨所有 run 扫描 resume_*.pth,取全局最大步数作为起点。

set -e
PY="C:/Users/Easonpc/psp-python312/python.exe"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/PSP-main"

FOLD=${1:-0}
BUDGET=${2:-1700}
TARGET=30001
LOGDIR=./exps_on_ABDOMEN_MR

START=0
RESUME=""
LATEST=$(ls $LOGDIR/CDFS_train_ABDOMEN_MR_cv${FOLD}/*/snapshots/resume_*.pth 2>/dev/null \
         | sed 's/.*resume_\([0-9]*\)\.pth/\1/' | sort -n | tail -1 || true)
if [ -n "$LATEST" ]; then
  START=$LATEST
  RESUME=$(ls $LOGDIR/CDFS_train_ABDOMEN_MR_cv${FOLD}/*/snapshots/resume_${LATEST}.pth | head -1)
fi

if [ "$START" -ge "$TARGET" ]; then
  echo "fold $FOLD 已达 $START 步(目标 $TARGET),无需继续"
  exit 0
fi

echo "===== fold $FOLD: 起点 $START 步,本块 $BUDGET 步,目标 $TARGET ====="

COMMON=(mode=train dataset=ABDOMEN_MR num_workers=0 n_steps=$TARGET
        max_iters_per_load=5000 eval_fold=$FOLD "test_label=[1,2,3,4]"
        exclude_label=None use_gt=False seed=2026 save_snapshot_every=1000
        print_interval=100 lr_step_gamma=0.98 start_iter=$START
        iter_budget=$BUDGET path.log_dir=$LOGDIR
        path.ABDOMEN_MR.data_dir=../data/ABD/ABDOMEN_MR)

if [ -n "$RESUME" ]; then
  "$PY" -u train.py with "${COMMON[@]}" reload_model_path="$RESUME" 2>&1 \
    | grep -E "Resumed|step [0-9]+:|tail snapshot|Completed|Error|Traceback" | tail -5
else
  "$PY" -u train.py with "${COMMON[@]}" 2>&1 \
    | grep -E "Resumed|step [0-9]+:|tail snapshot|Completed|Error|Traceback" | tail -5
fi
