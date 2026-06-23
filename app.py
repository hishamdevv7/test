from fastapi import FastAPI
from pydantic import BaseModel
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import torch

MODEL_DIR = "/app/onnx_model_int8"
BASE_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = ORTModelForFeatureExtraction.from_pretrained(MODEL_DIR)

app = FastAPI()

class TextIn(BaseModel):
    text: str

@app.post("/embed")
def embed(payload: TextIn):
    inputs = tokenizer(payload.text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return {"embedding": embedding}
