#!/bin/bash
#SBATCH -p gpu_chen
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -o job_obs_rag.out

# MemoryBank-SiliconFriend 记忆摘要流水线
# 对标 hyperbolic_memory 的 scripts/main.sh 参数风格
#
# 抽取模型 & 生成模型 完全分离，各自独立选择 transformers / openai 后端:
#   --extraction-handler-type transformers → 本地 llama/qwen 做逐日摘要
#   --generation-handler-type  transformers → 本地模型做整体汇总
#

python main.py \
  --memory-dir memories/eng_memory_cases.json \
  --out-file memories/eng_memory_cases_enriched.json \
  --language en \
  --extraction-handler-type transformers \
  --extraction-model-path /share/home/leiyh5/models/Qwen2.5-7B-Instruct \
  --generation-handler-type transformers \
  --generation-model-path /share/home/leiyh5/models/Llama-3.2-3B-Instruct \
  --embedding-model sentence-transformers/all-mpnet-base-v2 \
  "$@"
