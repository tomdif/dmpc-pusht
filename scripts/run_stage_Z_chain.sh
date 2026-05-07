#!/usr/bin/env bash
# Stage Z chain — value-head bootstrap retrain on Y's reward ckpt:
#   reward+value training (5K) -> BC (8K) -> eval with use_value

set -euo pipefail

ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_stage_Z
CKPTROOT=$ROOT/ckpts/stage_Z

Y_REWARD_CKPT=$ROOT/ckpts/stage_Y/reward/ckpt_step5000.pt

mkdir -p $LOGDIR $CKPTROOT/reward

echo "=== [Z] Step 1/3: 5K reward+value fine-tune (resume Y) ===" | tee -a $LOGDIR/chain.log
date | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1z_reward.yaml \
    --device cuda --steps 5000 \
    --log-every 100 --diag-every 1000 \
    --num-workers 4 \
    --ckpt-every 2500 \
    --ckpt-dir $CKPTROOT/reward \
    --resume $Y_REWARD_CKPT \
    > $LOGDIR/reward.log 2>&1
REWARD_CKPT=$CKPTROOT/reward/ckpt_step5000.pt
echo "[Z] reward+value done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [Z] Step 2/3: 8K BC retrain ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1c_bc_train.py \
    --config configs/stage1z_reward.yaml \
    --ckpt $REWARD_CKPT \
    --out $CKPTROOT/bc_policy_Z.pt \
    --device cuda \
    --steps 8000 \
    --log-every 200 \
    --num-workers 4 \
    > $LOGDIR/bc.log 2>&1
echo "[Z] BC done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [Z] Step 3/3: online eval with use_value ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1z_reward.yaml \
    --ckpt $REWARD_CKPT \
    --bc-ckpt $CKPTROOT/bc_policy_Z.pt \
    --device cuda \
    --n-episodes 10 \
    --max-steps 200 \
    --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 --cem-init-std 0.3 \
    --replan-every 2 \
    --use-value --value-weight 1.0 \
    > $LOGDIR/eval.log 2>&1
echo "[Z] eval done at $(date)" | tee -a $LOGDIR/chain.log
echo "=== ALL DONE ===" | tee -a $LOGDIR/chain.log
