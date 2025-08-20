
export CUDA_VISIBLE_DEVICES = 3

python -m lerobot-train \
  --dataset.repo_id=imManjusaka/cube_placement \
  --policy.type=act \
  --batch_size=128 \
  --output_dir=outputs/train/act_trells_pp_test \
  --job_name=act_cube_placement_v21 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=lerobot/act