import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")

# Model: Qwen2.5-32B-Instruct
# Size: ~32-34 GB
# Requires 2 GPUs (1 L4 of 24GB is insufficient).
model_id = "Qwen/Qwen2.5-32B-Instruct"

print(f"🚀 Initializing Qwen2.5-32B (8-bit Precision) on 2x L4 GPUs...")

# Configure for quantization with bitsandbytes (bnb) to reduce memory usage
# We use 8 bit 
bnb_config = BitsAndBytesConfig(
    load_in_8bit=False,                 # with 8 bit model approx 32GB
    llm_int8_enable_fp32_cpu_offload=True 
)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# model loading (pipeline parallelism)
print(f"Loading {model_id} across GPUs...")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",          # Automatic distribution across available GPUs
    quantization_config=bnb_config,
    trust_remote_code=True,
    token=token,
    low_cpu_mem_usage=True
)

print(f"Model Loaded!")
# Will show the distribution of layers across GPUs and the memory footprint
print(f"   - Memory Footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
print(f"   - Distribution: {model.hf_device_map}") 

# Inference
prompt = "Explain the difference between 'Data Parallelism' and 'Model Parallelism' in distributed computing."
messages = [{"role": "user", "content": prompt}]

inputs = tokenizer.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 
    return_tensors="pt"
).to("cuda")

print("Streaming response (Creative Mode)...")
print("-" * 30)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

_ = model.generate(
    inputs, 
    streamer=streamer, 
    max_new_tokens=512, 
    do_sample=True,       
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)
print("-" * 30)