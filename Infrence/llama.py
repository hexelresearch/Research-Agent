import torch
from transformers import pipeline
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

def model_setup(model_id: str, torch_dtype, device_map: str):
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    return pipe

def messages(prompt):
    messages = [
            {"role": "system", "content": "You are a Research Assistant Who always talk with Facts and In depth Knowledge"},
            {"role": "user", "content": prompt},
        ]
    return messages

meta_llama_3b_instruct = model_setup("meta-llama/Llama-3.2-3B-Instruct", torch.bfloat16, "cuda")
meta_llama_1b = model_setup("meta-llama/Llama-3.2-1B-Instruct", torch.bfloat16, "cuda")

app = FastAPI()

@app.get('/')
def index():
    return JSONResponse({"message":"Welcome to Hexel AI"}, status_code=status.HTTP_200_OK)

@app.get("/search1b")
def infrence(prompt: str):
    try:
        outputs = meta_llama_3b_instruct(
        messages(prompt),
        max_new_tokens=1024,
        )
        return {"message":outputs[0]["generated_text"][-1]}
    except Exception as e:
        return JSONResponse(e)
         
@app.get("/search3b")
def infrence(prompt: str):
    try:
        outputs = meta_llama_1b(
        messages(prompt),
        max_new_tokens=1024,
        )
        return {"message":outputs[0]["generated_text"][-1]}
    except Exception as e:
        return JSONResponse(e)
        
    
    