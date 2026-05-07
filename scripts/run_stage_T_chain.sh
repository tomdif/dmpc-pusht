#!/usr/bin/env bash
# Stage T chain — 80M params on V's 2-way combined data:
#   pretrain (12K) -> reward (5K) -> BC (8K) -> eval (10 ep, same seeds 0-9)

set -euo pipefail

ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_stage_T
CKPTROOT=$ROOT/ckpts/stage_T

mkdir -p $LOGDIR $CKPTROOT/pretrain $CKPTROOT/reward

echo "=== [T] Step 1/4: 12K combined-data pretrain (80M) ===" | tee -a $LOGDIR/chain.log
date | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1t_combined.yaml \
    --device cuda --steps 12000 \
    --log-every 200 --diag-every 1500 \
    --num-workers 4 \
    --ckpt-every 3000 \
    --ckpt-dir $CKPTROOT/pretrain \
    > $LOGDIR/pretrain.log 2>&1
PRETRAIN_CKPT=$CKPTROOT/pretrain/ckpt_step12000.pt
echo "[T] pretrain done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [T] Step 2/4: 5K reward fine-tune ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1t_reward.yaml \
    --device cuda --steps 5000 \
    --log-every 100 --diag-every 1000 \
    --num-workers 4 \
    --ckpt-every 2500 \
    --ckpt-dir $CKPTROOT/reward \
    --resume $PRETRAIN_CKPT \
    > $LOGDIR/reward.log 2>&1
REWARD_CKPT=$CKPTROOT/reward/ckpt_step5000.pt
echo "[T] reward done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [T] Step 3/4: 8K BC retrain ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1c_bc_train.py \
    --config configs/stage1t_reward.yaml \
    --ckpt $REWARD_CKPT \
    --out $CKPTROOT/bc_policy_T.pt \
    --device cuda \
    --steps 8000 \
    --log-every 200 \
    --num-workers 4 \
    > $LOGDIR/bc.log 2>&1
echo "[T] BC done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [T] Step 4/4: online eval (same seeds 0-9) ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1t_reward.yaml \
    --ckpt $REWARD_CKPT \
    --bc-ckpt $CKPTROOT/bc_policy_T.pt \
    --device cuda \
    --n-episodes 10 \
    --max-steps 200 \
    --horizon 8 --cem-samples 256 --cem-elite 32 --cem-iters 4 --cem-init-std 0.5 \
    --replan-every 2 \
    > $LOGDIR/eval.log 2>&1
echo "[T] eval done at $(date)" | tee -a $LOGDIR/chain.log
echo "=== ALL DONE ===" | tee -a $LOGDIR/chain.log
