export CUDA_VISIBLE_DEVICES=0,1
MODEL_NAME=mistral-7b
mkdir -p /data1/zehuali/workspace/zli_work/DSAA6000Q-ass4/train_logs/$MODEL_NAME
FORCE_TORCHRUN=1 llamafactory-cli train /data/zli/workspace/LLaMA-Factory/examples/train_lora/Mistral-7B-Instruct-v0.2_lora_dpo.yaml \
    > /data1/zehuali/workspace/zli_work/DSAA6000Q-ass4/train_logs/$MODEL_NAME/log_preferencedpo_$(date +"%Y%m%d_%H%M%S").log 2>&1