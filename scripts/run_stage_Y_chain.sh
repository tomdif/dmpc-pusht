#!/usr/bin/env bash
# Stage Y chain — per-step reward head retrain on X-pretrain ckpt:
#   reward (5K) -> BC (8K) -> eval

set -euo pipefail

ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_stage_Y
CKPTROOT=$ROOT/ckpts/stage_Y

X_PRETRAIN_CKPT=$ROOT/ckpts/stage_X/pretrain/ckpt_step16000.pt

mkdir -p $LOGDIR $CKPTROOT/reward

echo "=== [Y] Step 1/3: 5K reward fine-tune (per-step reward head) ===" | tee -a $LOGDIR/chain.log
date | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1x_reward.yaml \
    --device cuda --steps 5000 \
    --log-every 100 --diag-every 1000 \
    --num-workers 4 \
    --ckpt-every 2500 \
    --ckpt-dir $CKPTROOT/reward \
    --resume $X_PRETRAIN_CKPT \
    > $LOGDIR/reward.log 2>&1
REWARD_CKPT=$CKPTROOT/reward/ckpt_step5000.pt
echo "[Y] reward done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [Y] Step 2/3: 8K BC retrain ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1c_bc_train.py \
    --config configs/stage1x_reward.yaml \
    --ckpt $REWARD_CKPT \
    --out $CKPTROOT/bc_policy_Y.pt \
    --device cuda \
    --steps 8000 \
    --log-every 200 \
    --num-workers 4 \
    > $LOGDIR/bc.log 2>&1
echo "[Y] BC done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [Y] Step 3/3: online eval (T2 hyperparams) ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1x_reward.yaml \
    --ckpt $REWARD_CKPT \
    --bc-ckpt $CKPTROOT/bc_policy_Y.pt \
    --device cuda \
    --n-episodes 10 \
    --max-steps 200 \
    --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 --cem-init-std 0.3 \
    --replan-every 2 \
    > $LOGDIR/eval.log 2>&1
echo "[Y] eval done at $(date)" | tee -a $LOGDIR/chain.log
echo "=== ALL DONE ===" | tee -a $LOGDIR/chain.log
