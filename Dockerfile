FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    onnxruntime \
    tokenizers \
    numpy \
    huggingface_hub \
    fastapi \
    uvicorn

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Download model
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('ibm-granite/granite-embedding-97m-multilingual-r2', 'onnx/model.onnx', local_dir='/app/model'); \
    hf_hub_download('ibm-granite/granite-embedding-97m-multilingual-r2', 'tokenizer.json', local_dir='/app/model')"

# 🔥 RUN QUANTIZATION HERE (BUILD TIME ONLY)
COPY quantize.py /app/quantize.py
RUN python /app/quantize.py

# Copy app AFTER quantization
COPY app.py /app/app.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
