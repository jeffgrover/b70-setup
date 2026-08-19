"""Historical OpenVINO serving prototype; not part of the active local stack."""

import sys
import openvino_genai as ov_genai
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()
gguf_path = sys.argv[1] 
device = "GPU" 

# These specific settings prevent the CPU RAM crash
ov_config = {
    "PERFORMANCE_HINT": "LATENCY",
    "CACHE_DIR": "./ov_cache",
    # This tells OpenVINO to use the GPU's memory directly for weights
    "GPU_UAV_ACCESS_STAGING_BUFFER_SIZE": "1", 
    "NUM_STREAMS": "1" # Reduces overhead during the loading phase
}

print(f"Loading {gguf_path} directly to B70 VRAM...")

# Construct the pipeline with the config
pipe = ov_genai.LLMPipeline(gguf_path, device, **ov_config)

class ChatRequest(BaseModel):
    model: str
    messages: list
    stream: bool = False

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    prompt = request.messages[-1]["content"]
    result = pipe.generate(prompt, max_new_tokens=512)
    return {"choices": [{"message": {"role": "assistant", "content": result}}]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
