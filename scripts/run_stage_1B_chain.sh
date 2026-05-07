#!/usr/bin/env bash
# Stage 1B-param chain — 1B-param ViT-Giant on 5-way self-play data:
#   pretrain (12K) -> reward (4K) -> BC (8K) -> eval train + held-out

set -euo pipefail

ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_stage_1B
CKPTROOT=$ROOT/ckpts/stage_1B

mkdir -p $LOGDIR $CKPTROOT/pretrain $CKPTROOT/reward

echo "=== [1B] Step 1/4: 12K pretrain (1B params, 5-way data) ===" | tee -a $LOGDIR/chain.log
date | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1_1b_pretrain.yaml \
    --device cuda --steps 12000 \
    --log-every 200 --diag-every 1500 \
    --num-workers 4 \
    --ckpt-every 3000 \
    --ckpt-dir $CKPTROOT/pretrain \
    > $LOGDIR/pretrain.log 2>&1
PRETRAIN_CKPT=$CKPTROOT/pretrain/ckpt_step12000.pt
echo "[1B] pretrain done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [1B] Step 2/4: 4K reward+value fine-tune ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1_1b_reward.yaml \
    --device cuda --steps 4000 \
    --log-every 100 --diag-every 1000 \
    --num-workers 4 \
    --ckpt-every 2000 \
    --ckpt-dir $CKPTROOT/reward \
    --resume $PRETRAIN_CKPT \
    > $LOGDIR/reward.log 2>&1
REWARD_CKPT=$CKPTROOT/reward/ckpt_step4000.pt
echo "[1B] reward done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [1B] Step 3/4: 8K BC retrain ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1c_bc_train.py \
    --config configs/stage1_1b_reward.yaml \
    --ckpt $REWARD_CKPT \
    --out $CKPTROOT/bc_policy_1B.pt \
    --device cuda \
    --steps 8000 \
    --log-every 200 \
    --num-workers 4 \
    > $LOGDIR/bc.log 2>&1
echo "[1B] BC done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [1B] Step 4/4a: online eval (training seeds 0-9) ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1_1b_reward.yaml \
    --ckpt $REWARD_CKPT \
    --bc-ckpt $CKPTROOT/bc_policy_1B.pt \
    --device cuda \
    --n-episodes 10 \
    --max-steps 200 \
    --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 --cem-init-std 0.3 \
    --replan-every 2 \
    --use-value --value-weight 1.0 \
    --seed 0 \
    > $LOGDIR/eval_train.log 2>&1
echo "[1B] eval (training) done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [1B] Step 4/4b: online eval (held-out 100-109) ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1_1b_reward.yaml \
    --ckpt $REWARD_CKPT \
    --bc-ckpt $CKPTROOT/bc_policy_1B.pt \
    --device cuda \
    --n-episodes 10 \
    --max-steps 200 \
    --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 --cem-init-std 0.3 \
    --replan-every 2 \
    --use-value --value-weight 1.0 \
    --seed 100 \
    > $LOGDIR/eval_holdout.log 2>&1
echo "[1B] eval (held-out) done at $(date)" | tee -a $LOGDIR/chain.log
echo "=== ALL DONE ===" | tee -a $LOGDIR/chain.log
