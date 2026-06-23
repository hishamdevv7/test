FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    sentence-transformers \
    openvino \
    fastapi \
    uvicorn

# Pre-download weights at build time (bakes model into image)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('ibm-granite/granite-embedding-97m-multilingual-r2', \
    backend='openvino', \
    model_kwargs={'file_name': 'openvino/openvino_model_qint8_quantized.xml'})"

COPY app.py /app/app.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
