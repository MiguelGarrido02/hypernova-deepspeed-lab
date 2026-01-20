import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer 

# 1. SETUP & CONFIG
app = FastAPI(title="HyperNova Inference Server")
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

model_id = "MultiverseComputingCAI/HyperNova-60B"

print(f"-----Initializing HyperNova-60B on 2x RTX 4090 with Tensor Parallelism...-----")

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# We load the model only when the server starts, not for every request.
llm = LLM(
    model=model_id,
    tensor_parallel_size=2,      # DISTRIBUTED COMPUTING (Key Requirement)
    trust_remote_code=True,
    dtype="bfloat16",            
    quantization="mxfp4",
    enforce_eager=True,          
    max_model_len=8192           
)

class ChatRequest(BaseModel):
    messages: list # Format: [{"role": "user", "content": "..."}]
    max_tokens: int = 512
    temperature: float = 0.1


@app.post("/generate")
def generate_text(request: ChatRequest):
    print(f"Received Chat Request...")

    # 1. USE THE DOCUMENTATION'S METHOD TO FORMAT PROMPT
    # This converts the list of messages into the EXACT string the model expects.
    prompt_tokenized = tokenizer.apply_chat_template(
        request.messages,
        tokenize=False, # We want the string string for vLLM
        add_generation_prompt=True
    )

    # 2. Define params
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=0.9,
        max_tokens=request.max_tokens,
        # We can now trust the model's own EOS token, but keep safety stops
        stop=["<|endoftext|>", "</s>"] 
    )

    # 3. Run Inference
    outputs = llm.generate([prompt_tokenized], sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    # 4. STILL KEEP THE PARSING LOGIC (Just in case thoughts persist)
    # Even with correct templates, "reasoning" models often still output thoughts.
    delimiter = "assistantfinal"
    if delimiter in generated_text:
        final_response = generated_text.split(delimiter)[-1].strip()
    else:
        final_response = generated_text.strip()

    return {"response": final_response}

if __name__ == "__main__":
    # Listen on all IPs (0.0.0.0) on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)