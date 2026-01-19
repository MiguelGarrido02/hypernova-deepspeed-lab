import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from vllm import LLM, SamplingParams

# 1. SETUP & CONFIG
app = FastAPI(title="HyperNova Inference Server")
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

model_id = "MultiverseComputingCAI/HyperNova-60B"

print(f"-----Initializing HyperNova-60B on 2x RTX 4090 with Tensor Parallelism...-----")


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

# request structure
class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7


@app.post("/generate")
def generate_text(request: PromptRequest):
    """
    Receives a prompt from the RAG Frontend, runs inference, returns text.
    """
    print(f"📩 Received Prompt: {request.prompt[:50]}...")

    # Define params based on request
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=0.9,
        max_tokens=request.max_tokens,
        presence_penalty=1.1
    )

    # Run Inference
    outputs = llm.generate([request.prompt], sampling_params)

    # Extract text
    generated_text = outputs[0].outputs[0].text
    
    return {"response": generated_text}


if __name__ == "__main__":
    # Listen on all IPs (0.0.0.0) on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)