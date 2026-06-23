from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
from tokenizers import Tokenizer
import numpy as np

# -------------------
# SESSION CONFIG
# -------------------
sess_options = ort.SessionOptions()

sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1

# 🔥 CRITICAL FOR 488MB RAM LIMIT
sess_options.enable_cpu_mem_arena = False
sess_options.enable_mem_pattern = False

session = ort.InferenceSession(
    "/app/model/onnx/model.int8.onnx",   # ✅ USE INT8 MODEL
    sess_options=sess_options,
    providers=["CPUExecutionProvider"]
)

tokenizer = Tokenizer.from_file("/app/model/tokenizer.json")

app = FastAPI()

class TextIn(BaseModel):
    text: str


@app.post("/embed")
def embed(payload: TextIn):
    enc = tokenizer.encode(payload.text)

    input_ids = np.array([enc.ids], dtype=np.int64)
    attention_mask = np.array([enc.attention_mask], dtype=np.int64)

    outputs = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    )

    last_hidden = outputs[0]

    # memory-efficient pooling
    mask = attention_mask.astype(np.float32)[..., None]

    pooled = (last_hidden * mask).sum(axis=1) / mask.sum(axis=1)

    return {"embedding": pooled[0].tolist()}
