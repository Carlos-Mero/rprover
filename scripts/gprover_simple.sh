python main.py \
  --method gprover \
  --proof_model qwen3-30b-a3b \
  --eval_model gpt-5 \
  --guider_model gpt-5 \
  --eval_dataset NP_dataset/train_300.json \
  --prover_base_url <add_your_url_here> \
  --eval_base_url <add_your_url_here> \
  --prover_api_key <add_your_api_key_here> \
  --eval_api_key <add_your_api_key_here>
