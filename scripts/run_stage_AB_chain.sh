#!/usr/bin/env bash
# Stage AB chain — self-play with Z's policy + value retrain:
#   collect 700 episodes with Z (MPC+V, --use-value)
#   retrain on 5-way data (LeRobot + V/T/U/AB self-play)
#   reward+value (5K) -> BC (8K) -> eval

set -euo pipefail

ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_stage_AB
CKPTROOT=$ROOT/ckpts/stage_AB
ROLLDIR_NEW=$ROOT/rollouts/self_play_z

Z_REWARD_CKPT=$ROOT/ckpts/stage_Z/reward/ckpt_step5000.pt
Z_BC_CKPT=$ROOT/ckpts/stage_Z/bc_policy_Z.pt

mkdir -p $LOGDIR $CKPTROOT/pretrain $CKPTROOT/reward $ROLLDIR_NEW

echo "=== [AB] Step 1/5: collect 700 self-play with Z policy (use-value) ===" | tee -a $LOGDIR/chain.log
date | tee -a $LOGDIR/chain.log
python -u scripts/collect_self_play.py \
    --config configs/stage1z_reward.yaml \
    --ckpt $Z_REWARD_CKPT \
    --bc-ckpt $Z_BC_CKPT \
    --out $ROLLDIR_NEW \
    --device cuda \
    --n-episodes 700 \
    --max-steps 200 \
    --noise-std 0.2 \
    --random-frac 0.05 \
    --min-max-coverage 0.02 \
    --seed-base 200000 \
    > $LOGDIR/collect.log 2>&1
echo "[AB] collect done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [AB] Step 2/5: 16K combined-data pretrain (5-way, 173M, value) ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1ab_pretrain.yaml \
    --device cuda --steps 16000 \
    --log-every 200 --diag-every 2000 \
    --num-workers 4 \
    --ckpt-every 4000 \
    --ckpt-dir $CKPTROOT/pretrain \
    > $LOGDIR/pretrain.log 2>&1
PRETRAIN_CKPT=$CKPTROOT/pretrain/ckpt_step16000.pt
echo "[AB] pretrain done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [AB] Step 3/5: 5K reward+value fine-tune ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1ab_reward.yaml \
    --device cuda --steps 5000 \
    --log-every 100 --diag-every 1000 \
    --num-workers 4 \
    --ckpt-every 2500 \
    --ckpt-dir $CKPTROOT/reward \
    --resume $PRETRAIN_CKPT \
    > $LOGDIR/reward.log 2>&1
REWARD_CKPT=$CKPTROOT/reward/ckpt_step5000.pt
echo "[AB] reward+value done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [AB] Step 4/5: 8K BC retrain ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1c_bc_train.py \
    --config configs/stage1ab_reward.yaml \
    --ckpt $REWARD_CKPT \
    --out $CKPTROOT/bc_policy_AB.pt \
    --device cuda \
    --steps 8000 \
    --log-every 200 \
    --num-workers 4 \
    > $LOGDIR/bc.log 2>&1
echo "[AB] BC done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [AB] Step 5/5: online eval with use-value ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1ab_reward.yaml \
    --ckpt $REWARD_CKPT \
    --bc-ckpt $CKPTROOT/bc_policy_AB.pt \
    --device cuda \
    --n-episodes 10 \
    --max-steps 200 \
    --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 --cem-init-std 0.3 \
    --replan-every 2 \
    --use-value --value-weight 1.0 \
    > $LOGDIR/eval.log 2>&1
echo "[AB] eval done at $(date)" | tee -a $LOGDIR/chain.log
echo "=== ALL DONE ===" | tee -a $LOGDIR/chain.log
