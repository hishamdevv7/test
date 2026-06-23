from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "ibm-granite/granite-embedding-97m-multilingual-r2",
    backend="openvino",
    model_kwargs={"file_name": "openvino/openvino_model_qint8_quantized.xml"},
)

app = FastAPI()

class TextIn(BaseModel):
    text: str

@app.post("/embed")
def embed(payload: TextIn):
    embedding = model.encode(payload.text).tolist()
    return {"embedding": embedding}
