import torch
import langchain
from transformers import pipeline
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



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

# meta_llama_3b = model_setup("meta-llama/Llama-3.2-3B-Instruct", torch.bfloat16, "cuda")
# meta_llama_1b = model_setup("meta-llama/Llama-3.2-1B-Instruct", torch.bfloat16, "cuda")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=20)

loader = DirectoryLoader("dataset/text-to-image-generation")

loaded_docs = loader.load()

docs = text_splitter.split_documents(loaded_docs)

print(docs[:1])

print(len(docs))