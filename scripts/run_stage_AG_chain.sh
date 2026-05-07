#!/usr/bin/env bash
# Stage AG chain — object-aware pixel masking (light C-JEPA):
#   resume Z reward ckpt with object_mask_prob=0.5 in synthetic data
#   reward+value (5K) -> eval

set -euo pipefail

ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_stage_AG
CKPTROOT=$ROOT/ckpts/stage_AG

Z_REWARD_CKPT=$ROOT/ckpts/stage_Z/reward/ckpt_step5000.pt
Z_BC_CKPT=$ROOT/ckpts/stage_Z/bc_policy_Z.pt

mkdir -p $LOGDIR $CKPTROOT/reward

echo "=== [AG] Step 1/3: 5K reward+value fine-tune with object masking ===" | tee -a $LOGDIR/chain.log
date | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1ag_reward.yaml \
    --device cuda --steps 5000 \
    --log-every 100 --diag-every 1000 \
    --num-workers 4 \
    --ckpt-every 2500 \
    --ckpt-dir $CKPTROOT/reward \
    --resume $Z_REWARD_CKPT \
    > $LOGDIR/reward.log 2>&1
REWARD_CKPT=$CKPTROOT/reward/ckpt_step5000.pt
echo "[AG] reward+value done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [AG] Step 2/3: online eval (training seeds 0-9) ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1ag_reward.yaml \
    --ckpt $REWARD_CKPT \
    --bc-ckpt $Z_BC_CKPT \
    --device cuda \
    --n-episodes 10 \
    --max-steps 200 \
    --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 --cem-init-std 0.3 \
    --replan-every 2 \
    --use-value --value-weight 1.0 \
    --seed 0 \
    > $LOGDIR/eval_train.log 2>&1
echo "[AG] eval (training) done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [AG] Step 3/3: online eval (held-out 100-109) ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1ag_reward.yaml \
    --ckpt $REWARD_CKPT \
    --bc-ckpt $Z_BC_CKPT \
    --device cuda \
    --n-episodes 10 \
    --max-steps 200 \
    --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 --cem-init-std 0.3 \
    --replan-every 2 \
    --use-value --value-weight 1.0 \
    --seed 100 \
    > $LOGDIR/eval_holdout.log 2>&1
echo "[AG] eval (held-out) done at $(date)" | tee -a $LOGDIR/chain.log
echo "=== ALL DONE ===" | tee -a $LOGDIR/chain.log
