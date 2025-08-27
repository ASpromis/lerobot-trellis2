export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

# backbone now only support "dinov3-vit7b16" or "dinov3-convnext-tiny"
# Need to complete
backbone="dinov3-vit7b16"
task="wipe_board"
policy="act"


date=$(date +%m%d)

accelerate launch \
  --num_processes=1 \
  --mixed_precision=fp16 \
  -m lerobot.scripts.train_accelerate \
  --dataset.repo_id="imManjusaka/${task}" \
  --policy.type="${policy}" \
  --batch_size=128 \
  --output_dir="outputs/train/${policy}/${backbone}/${task}_${date}" \
  --job_name="${policy}-${task}-${backbone}-${date}" \
  --policy.device="cuda" \
  --wandb.enable=true \
  --policy.repo_id="lerobot/${policy}" \
  --policy.vision_backbone="${backbone}"