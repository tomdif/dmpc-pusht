#!/usr/bin/env bash
# Run n=500 PushT eval (K=5 + discriminator) split across 6 B200 GPUs.
# Shards: each GPU runs 84 seeds, total 504 (rounded for nice slicing).
set -euo pipefail
ROOT=/workspace/cser-jepa-v2
LOGDIR=/root/step5/logs/n500
mkdir -p $LOGDIR
cd $ROOT
source .venv/bin/activate

CKPT=$ROOT/ckpts/stage_AB/reward/ckpt_step5000.pt
GOAL=$ROOT/goals/pusht_goal_lerobot.pt
DISC=/root/step5/ckpts/discriminator_round1.pt

run_shard() {
    local gpu=$1
    local seed_start=$2
    local n=$3
    CUDA_VISIBLE_DEVICES=$gpu python -u scripts/eval_dmpc.py \
        --world-config configs/stage1ab_reward.yaml \
        --world-ckpt $CKPT \
        --device cuda --n-episodes $n --max-steps 300 \
        --n-samples 64 --n-action-steps 8 --seed $seed_start --n-attempts 5 \
        --use-value --value-weight 1.0 \
        --use-goal --goal-weight 0.3 --goal-aggregate min --goal-multi \
        --goal-file $GOAL \
        --rerank-mode dmpc \
        --discriminator-ckpt $DISC --discriminator-weight 1.0 \
        > $LOGDIR/shard_gpu${gpu}.log 2>&1 &
    echo "[gpu$gpu] launched seeds $seed_start..$((seed_start+n-1))  PID=$!"
}

run_shard 0 0   84
run_shard 1 84  84
run_shard 2 168 84
run_shard 3 252 84
run_shard 4 336 84
run_shard 5 420 80
wait

echo "=== ALL SHARDS DONE ==="
# Aggregate.
python -c "
import re, glob
n_total = 0; n_succ = 0
for path in sorted(glob.glob('$LOGDIR/shard_gpu*.log')):
    with open(path) as f:
        for line in f:
            if line.startswith('  ep ') and 'success=' in line:
                n_total += 1
                if 'success=True' in line:
                    n_succ += 1
print(f'n={n_total}  successes={n_succ}  rate={n_succ/n_total:.2%}')
"
