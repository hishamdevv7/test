FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    optimum[onnxruntime] \
    sentence-transformers \
    transformers \
    fastapi \
    uvicorn

# Export + quantize at build time (bakes the model into the image)
RUN optimum-cli export onnx \
    --model ibm-granite/granite-embedding-97m-multilingual-r2 \
    --task feature-extraction \
    /app/onnx_model

RUN optimum-cli onnxruntime quantize \
    --onnx_model /app/onnx_model \
    --avx512_vnni \
    -o /app/onnx_model_int8

COPY app.py /app/app.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
