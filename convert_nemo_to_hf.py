import os
import torch
from transformers import NVGPTForCausalLM, NVGPTTokenizer, GenerationConfig 

nemo_ckpt_file = f"./GPT-2B-001_bf16_tp1.nemo" 
tokenizer_model = "tokenizer_path"

OUTPUT_HF_MODEL = f"./models/GPT-2B-001-HF"

# Instantiate the model from a nemo file
model = NVGPTForCausalLM.from_nemo_file(nemo_file=nemo_ckpt_file).cuda().to(torch.bfloat16)
tokenizer = NVGPTTokenizer(tokenizer_model)

# Save the model as HF checkpoint
model.save_pretrained(OUTPUT_HF_MODEL, max_shard_size="1GB")
tokenizer.save_pretrained(OUTPUT_HF_MODEL)

