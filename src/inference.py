import os 
import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv

# Load api
load_dotenv()
token = os.getenv("HF_TOKEN")

# Config
model_id = "MultiverseComputingCAI/HyperNova-60B"

# Quantization -> 4 bit config
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit = True,
#     bnb_4bit_quant_type = "nf4", #nf4 is a new quantization method that is more efficient than the traditional 4 bit quantization. It is based on the normal distribution and it is more accurate than the traditional 4 bit quantization.
#     bnb_4bit_compute_dtype = torch.float16 # Use float16 for computation to maintain a good balance between performance and accuracy
# )

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token = token, trust_remote_code = True)

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # quantization_config = bnb_config,
    device_map = "auto", # Automatically map model layers to available devices (GPUs/CPU) so we can leverage multiple GPUs 
    trust_remote_code = True,
    token = token
)

print("Model loaded :)\n---> Initializing DeepSpeed inference engine...")


# Initialize DeepSpeed inference engine
ds_engine = deepspeed.init_inference(
    model,
    mp_size = 2, # sharding across 2 GPUs
    dtype = torch.float16,
    replace_with_kernel_inject = True # Replace standard model layers with DeepSpeed optimized kernels for faster inference
)

print("\nDeepSpeed inference engine initialized successfully! Ready for inference.")


# Run Inference
prompt = "Explain the concept of quantum tensor networks in non technical terms."
inputs = tokenizer(prompt, return_tensors = "pt").to(ds_engine.module.device)

print("Generating response for the prompt:\n", prompt)

# Generate
with torch.no_grad():
    outputs = ds_engine.module.generate(
        **inputs,
        max_new_tokens = 150,
        temperature = 0.7,
        do_sample = True,
        top_n = 5
    )


response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("-" * 30)
print("MODEL OUTPUT:")
print(response)
print("-" * 30)