#!/usr/bin/env bash
# Stage AF chain — dense proximity reward via state decoder:
#   collect 300 episodes with state labels (Z policy)
#   retrain Z with state decoder + proximity-aware CEM
#   eval

set -euo pipefail

ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_stage_AF
CKPTROOT=$ROOT/ckpts/stage_AF
ROLLDIR_NEW=$ROOT/rollouts/self_play_af

Z_REWARD_CKPT=$ROOT/ckpts/stage_Z/reward/ckpt_step5000.pt
Z_BC_CKPT=$ROOT/ckpts/stage_Z/bc_policy_Z.pt

mkdir -p $LOGDIR $CKPTROOT/reward $ROLLDIR_NEW

echo "=== [AF] Step 1/3: collect 300 self-play with state labels ===" | tee -a $LOGDIR/chain.log
date | tee -a $LOGDIR/chain.log
python -u scripts/collect_self_play.py \
    --config configs/stage1z_reward.yaml \
    --ckpt $Z_REWARD_CKPT \
    --bc-ckpt $Z_BC_CKPT \
    --out $ROLLDIR_NEW \
    --device cuda \
    --n-episodes 300 \
    --max-steps 200 \
    --noise-std 0.25 \
    --random-frac 0.05 \
    --min-max-coverage 0.02 \
    --seed-base 300000 \
    > $LOGDIR/collect.log 2>&1
echo "[AF] collect done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [AF] Step 2/3: 5K reward+value+state fine-tune ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1af_reward.yaml \
    --device cuda --steps 5000 \
    --log-every 100 --diag-every 1000 \
    --num-workers 4 \
    --ckpt-every 2500 \
    --ckpt-dir $CKPTROOT/reward \
    --resume $Z_REWARD_CKPT \
    > $LOGDIR/reward.log 2>&1
REWARD_CKPT=$CKPTROOT/reward/ckpt_step5000.pt
echo "[AF] reward+value+state done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [AF] Step 3/3: online eval with use-value AND use-proximity ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1af_reward.yaml \
    --ckpt $REWARD_CKPT \
    --bc-ckpt $Z_BC_CKPT \
    --device cuda \
    --n-episodes 10 \
    --max-steps 200 \
    --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 --cem-init-std 0.3 \
    --replan-every 2 \
    --use-value --value-weight 1.0 \
    --use-proximity --proximity-weight 0.01 \
    > $LOGDIR/eval_train.log 2>&1
echo "[AF] eval (training seeds) done at $(date)" | tee -a $LOGDIR/chain.log

# Held-out eval
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1af_reward.yaml \
    --ckpt $REWARD_CKPT \
    --bc-ckpt $Z_BC_CKPT \
    --device cuda \
    --n-episodes 10 \
    --max-steps 200 \
    --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 --cem-init-std 0.3 \
    --replan-every 2 \
    --use-value --value-weight 1.0 \
    --use-proximity --proximity-weight 0.01 \
    --seed 100 \
    > $LOGDIR/eval_holdout.log 2>&1
echo "[AF] eval (held-out) done at $(date)" | tee -a $LOGDIR/chain.log
echo "=== ALL DONE ===" | tee -a $LOGDIR/chain.log
