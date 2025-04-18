from transformers import AutoTokenizer, Llama4ForConditionalGeneration, FbgemmFp8Config
import torch
from transformers import BitsAndBytesConfig
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # or load_in_4bit=True
    llm_int8_enable_fp32_cpu_offload=True  # <== THIS ALLOWS OFFLOAD TO CPU
)

messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="sequential",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])
print(outputs[0])
