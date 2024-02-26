import torch
from transformers import NVGPTForCausalLM, NVGPTTokenizer, GenerationConfig 

model_name = "./models/GPT-2B-001-HF"

model = NVGPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, use_flash_attention=True).cuda()
tokenizer = NVGPTTokenizer.from_pretrained(model_name)

sentences = ["Life is like a"] 
inputs = tokenizer(sentences[0], return_tensors="pt")
inputs['input_ids'] = inputs['input_ids'].cuda()
inputs['input_ids'] = inputs['input_ids'][:,1:]
inputs['attention_mask'] = inputs['attention_mask'][:,1:]
inputs['attention_mask'] = inputs['attention_mask'].cuda()

with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    logits = model(**inputs).logits

original = torch.load("./example_pred")

print("Inference HF", logits)
print("Inference NeMo", original[0])
