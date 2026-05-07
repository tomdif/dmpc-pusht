#!/usr/bin/env bash
# Stage M chain: pretrain (30K) -> reward fine-tune (10K) -> BC retrain (8K).
# Chained sequentially; second step resumes from first's final ckpt.
# Run from /workspace/cser-jepa-v2/ with venv active.

set -euo pipefail

ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_stage_M
CKPTROOT=$ROOT/ckpts/stage_M

mkdir -p $LOGDIR $CKPTROOT/pretrain $CKPTROOT/reward

echo "=== [chain] Step 1/3: 30K pretrain ===" | tee -a $LOGDIR/chain.log
date | tee -a $LOGDIR/chain.log
python -u scripts/stage1_pretrain.py \
    --config configs/stage1_pusht_M.yaml \
    --device cuda --steps 30000 \
    --log-every 200 --diag-every 2000 \
    --num-workers 4 \
    --ckpt-every 5000 \
    --ckpt-dir $CKPTROOT/pretrain \
    > $LOGDIR/pretrain.log 2>&1
PRETRAIN_CKPT=$CKPTROOT/pretrain/ckpt_step30000.pt
echo "[chain] pretrain done at $(date), ckpt=$PRETRAIN_CKPT" | tee -a $LOGDIR/chain.log

echo "=== [chain] Step 2/3: 10K reward fine-tune ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1_pretrain.py \
    --config configs/stage1b_reward_M.yaml \
    --device cuda --steps 10000 \
    --log-every 100 --diag-every 1000 \
    --num-workers 4 \
    --ckpt-every 2500 \
    --ckpt-dir $CKPTROOT/reward \
    --resume $PRETRAIN_CKPT \
    > $LOGDIR/reward.log 2>&1
REWARD_CKPT=$CKPTROOT/reward/ckpt_step10000.pt
echo "[chain] reward fine-tune done at $(date), ckpt=$REWARD_CKPT" | tee -a $LOGDIR/chain.log

echo "=== [chain] Step 3/3: 8K BC retrain ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1c_bc_train.py \
    --config configs/stage1b_reward_M.yaml \
    --ckpt $REWARD_CKPT \
    --out $CKPTROOT/bc_policy_M.pt \
    --device cuda \
    --steps 8000 \
    --log-every 200 \
    --num-workers 4 \
    > $LOGDIR/bc.log 2>&1
echo "[chain] BC retrain done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [chain] ALL DONE — ready for online eval ===" | tee -a $LOGDIR/chain.log
echo "  reward ckpt: $REWARD_CKPT" | tee -a $LOGDIR/chain.log
echo "  bc ckpt    : $CKPTROOT/bc_policy_M.pt" | tee -a $LOGDIR/chain.log
