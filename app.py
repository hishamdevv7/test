from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
from tokenizers import Tokenizer
import numpy as np

session = ort.InferenceSession(
    "/app/model/onnx/model.onnx",
    providers=["CPUExecutionProvider"],
    sess_options=ort.SessionOptions()
)
session.get_session_options().intra_op_num_threads = 1  # cuts thread-pool memory

tokenizer = Tokenizer.from_file("/app/model/tokenizer.json")

app = FastAPI()

class TextIn(BaseModel):
    text: str

@app.post("/embed")
def embed(payload: TextIn):
    enc = tokenizer.encode(payload.text)
    input_ids = np.array([enc.ids], dtype=np.int64)
    attention_mask = np.array([enc.attention_mask], dtype=np.int64)

    outputs = session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })

    last_hidden = outputs[0]  # (1, seq_len, 384)
    mask = attention_mask[..., None]
    pooled = (last_hidden * mask).sum(1) / mask.sum(1)

    return {"embedding": pooled[0].tolist()}
