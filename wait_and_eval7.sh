#!/bin/bash
LOG=/home/ec2-user/pipeline7.log
EVAL_LOG=/home/ec2-user/eval7.log

echo "[wait_and_eval7] Started at $(date)" > /home/ec2-user/wait_and_eval7.out

# Wait for Pipeline 7 to finish (all 6 epochs done)
echo "[wait_and_eval7] Waiting for RL Epoch 6 completion..." >> /home/ec2-user/wait_and_eval7.out
while true; do
  if grep -q '\[RL Epoch 06/6\]' "$LOG" 2>/dev/null; then
    echo "[wait_and_eval7] Found epoch 6 completion at $(date)" >> /home/ec2-user/wait_and_eval7.out
    break
  fi
  if grep -qE 'Traceback|Error|CUDA out of memory' "$LOG" 2>/dev/null; then
    echo "[wait_and_eval7] ERROR detected in pipeline7.log at $(date)" >> /home/ec2-user/wait_and_eval7.out
    tail -n 30 "$LOG" >> /home/ec2-user/wait_and_eval7.out
    exit 1
  fi
  sleep 120
done

# Copy best checkpoint
echo "[wait_and_eval7] Copying checkpoint..." >> /home/ec2-user/wait_and_eval7.out
sudo mkdir -p /data/checkpoints/rl7_eval
sudo cp /data/checkpoints/rl7/rl_epoch_06/actor.pt /data/checkpoints/rl7_eval/model.pt
echo "[wait_and_eval7] Checkpoint copied: $(ls -lh /data/checkpoints/rl7_eval/model.pt)" >> /home/ec2-user/wait_and_eval7.out

# Run evaluation
echo "[wait_and_eval7] Starting evaluation at $(date)" >> /home/ec2-user/wait_and_eval7.out
cd /home/ec2-user/modeling-affective-decision
sudo docker compose run --rm addan python evaluate.py \
  --data_dir /app/data \
  --ckpt /app/checkpoints/rl7_eval \
  --output_csv /app/checkpoints/eval7_results.csv \
  > "$EVAL_LOG" 2>&1
echo "[wait_and_eval7] Evaluation finished at $(date) with exit code $?" >> /home/ec2-user/wait_and_eval7.out

# Copy results CSV locally so we can inspect it
sudo cp /data/checkpoints/eval7_results.csv /home/ec2-user/eval7_results.csv 2>/dev/null
echo "[wait_and_eval7] Results saved to /home/ec2-user/eval7_results.csv" >> /home/ec2-user/wait_and_eval7.out
echo "[wait_and_eval7] DONE" >> /home/ec2-user/wait_and_eval7.out
