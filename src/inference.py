import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from dotenv import load_dotenv

# 1. Setup
load_dotenv()
token = os.getenv("HF_TOKEN")

# We use the Unsloth 4-bit quantization to fit 70B into 48GB VRAM
model_id = "unsloth/llama-3-70b-Instruct-bnb-4bit"

print(f"🚀 Initializing Parallel Load for {model_id}...")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"


# Parallel Model Loading (The Critical Step)
# On 2x L4s, it will put ~20GB on GPU 0 and ~20GB on GPU 1.
print("📦 Distributing model across GPU 0 and GPU 1...")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto", #automaitcally split model across available GPUs
    trust_remote_code=True,
    token=token,
    dtype=torch.bfloat16, # Native for L4
    low_cpu_mem_usage=True
)

# Verify where the model lives
print(f"Model Loaded!")
print(f"   - Memory Footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
print(f"   - Distribution: {model.hf_device_map}") 

#Run Inference
prompt = "Explain the architecture of Transformer models for a 5-year old."

messages = [
    {"role": "user", "content": prompt},
]
inputs = tokenizer.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 
    return_tensors="pt"
).to("cuda") # "cuda" automatically points to the entry GPU

print("Streaming response from Multi-GPU Cluster...")
print("-" * 30)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

_ = model.generate(
    inputs, 
    streamer=streamer, 
    max_new_tokens=300, 
    temperature=0.6,
    do_sample=True,
    top_p = 0.9,
    pad_token_id=tokenizer.eos_token_id # Make sure it knows when to stop
)
print("-" * 30)