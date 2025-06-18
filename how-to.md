## How to run INF2 DPOJPI benchmark?
NOTE: the data (rejected resampled from llama 3.3 instruct) is available at `/mnt/vast/home/sanjana/dpo_data/dpojpi_chatml_no_names_llama33i_resample.jsonl`
```bash
screen -S reward_bench
sinfo -N -h -o "%N %t"
srun -N 1 --gres=gpu:8 --exclusive --time=0 --job-name=inf2_reward_bench --pty bash
cd ~/reward-bench
MODEL_PATH="/mnt/vast/home/jimmy/verl-private/examples/slurm/checkpoints/verl_helpsteer3_multinode/verl-env-no_structured_inference_Qwen3-14B_N_2025-06-06_21-42-28/global_step_1200/merged_hf_model"
uv run python scripts/run_generative_v1_inf2.py --model=${MODEL_PATH} --num_gpus=8 --eval_set=inf2_sets --disable_beaker_save --do_not_save
```