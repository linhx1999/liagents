python -m vllm.entrypoints.openai.api_server \
  --model  /root/models/Qwen3-30B-A3B-Instruct-2507 \
  --served-model-name Qwen3-30B-A3B-Instruct-2507 \
  --max-model-len 8196 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 8 \
  --max_num_seqs 4 \
  --dtype float16 \
  --port 8000 \
  --trust-remote-code