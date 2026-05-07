#!/usr/bin/env bash
# Stage V' chain — second iteration of self-play loop:
#   collect with V's policy (stronger than M's) -> retrain on union of
#   original + V-self-play + V'-self-play -> eval.

set -euo pipefail

ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_stage_Vp
CKPTROOT=$ROOT/ckpts/stage_Vp
ROLLDIR_NEW=$ROOT/rollouts/self_play_vp
ROLLDIR_PREV=$ROOT/rollouts/self_play

V_REWARD_CKPT=$ROOT/ckpts/stage_V/reward/ckpt_step5000.pt
V_BC_CKPT=$ROOT/ckpts/stage_V/bc_policy_V.pt

mkdir -p $LOGDIR $CKPTROOT/pretrain $CKPTROOT/reward $ROLLDIR_NEW

echo "=== [Vp] Step 1/5: collect 600 self-play rollouts with V policy ===" | tee -a $LOGDIR/chain.log
date | tee -a $LOGDIR/chain.log
python -u scripts/collect_self_play.py \
    --config configs/stage1b_reward_M.yaml \
    --ckpt $V_REWARD_CKPT \
    --bc-ckpt $V_BC_CKPT \
    --out $ROLLDIR_NEW \
    --device cuda \
    --n-episodes 600 \
    --max-steps 200 \
    --noise-std 0.3 \
    --random-frac 0.1 \
    --min-max-coverage 0.02 \
    --seed-base 50000 \
    > $LOGDIR/collect.log 2>&1
echo "[Vp] collect done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [Vp] Step 2/5: 12K combined-data pretrain (3-way) ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1vp_combined.yaml \
    --device cuda --steps 12000 \
    --log-every 200 --diag-every 1500 \
    --num-workers 4 \
    --ckpt-every 3000 \
    --ckpt-dir $CKPTROOT/pretrain \
    > $LOGDIR/pretrain.log 2>&1
PRETRAIN_CKPT=$CKPTROOT/pretrain/ckpt_step12000.pt
echo "[Vp] pretrain done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [Vp] Step 3/5: 5K reward fine-tune ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1vp_reward.yaml \
    --device cuda --steps 5000 \
    --log-every 100 --diag-every 1000 \
    --num-workers 4 \
    --ckpt-every 2500 \
    --ckpt-dir $CKPTROOT/reward \
    --resume $PRETRAIN_CKPT \
    > $LOGDIR/reward.log 2>&1
REWARD_CKPT=$CKPTROOT/reward/ckpt_step5000.pt
echo "[Vp] reward done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [Vp] Step 4/5: 8K BC retrain ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1c_bc_train.py \
    --config configs/stage1b_reward_M.yaml \
    --ckpt $REWARD_CKPT \
    --out $CKPTROOT/bc_policy_Vp.pt \
    --device cuda \
    --steps 8000 \
    --log-every 200 \
    --num-workers 4 \
    > $LOGDIR/bc.log 2>&1
echo "[Vp] BC done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [Vp] Step 5/5: online eval (same seeds 0-9) ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1b_reward_M.yaml \
    --ckpt $REWARD_CKPT \
    --bc-ckpt $CKPTROOT/bc_policy_Vp.pt \
    --device cuda \
    --n-episodes 10 \
    --max-steps 200 \
    --horizon 8 --cem-samples 256 --cem-elite 32 --cem-iters 4 --cem-init-std 0.5 \
    --replan-every 2 \
    > $LOGDIR/eval.log 2>&1
echo "[Vp] eval done at $(date)" | tee -a $LOGDIR/chain.log
echo "=== ALL DONE ===" | tee -a $LOGDIR/chain.log
