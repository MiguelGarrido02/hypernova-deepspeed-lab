import os
from dotenv import load_dotenv
from vllm import LLM, SamplingParams

# token de HF
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# get model
model_id = "MultiverseComputingCAI/HyperNova-60B" 

print(f"🚀 Iniciando HyperNova-60B en 2x RTX 4090...")

# vLLM LLM initialization
# tensor_parallel_size=2 hace que el modelo se reparta en ambas GPUs automáticamente
llm = LLM(
    model=model_id,
    tensor_parallel_size=2,      # use both GPUs
    trust_remote_code=True,
    dtype="bfloat16",             # torch.mxfp4 needs bfloat16 
    quantization="mxfp4", # model requirements
    enforce_eager=True,          # 
    max_model_len=8192           # how much memory to allocate for xontext
)

# generation parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    presence_penalty=1.1
)

# inference
prompt = "Explain the attention mechanism in transformers in simple terms."

# vLLM waits for a list of prompts
outputs = llm.generate([prompt], sampling_params)

# results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"\nPrompt: {prompt}")
    print(f"\nResponse: {generated_text}")