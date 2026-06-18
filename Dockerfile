FROM ghcr.io/huggingface/text-embeddings-inference:cpu-latest AS tei
FROM qdrant/qdrant:latest AS qdrant
FROM maximhq/bifrost:latest AS bifrost

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

COPY --from=tei / /
COPY --from=qdrant / /
COPY --from=bifrost / /

ENV HUGGINGFACE_HUB_CACHE=/data \
    PORT=80 \
    MKL_ENABLE_INSTRUCTIONS=AVX512_E4 \
    RAYON_NUM_THREADS=8 \
    LD_PRELOAD=/usr/local/libfakeintel.so \
    LD_LIBRARY_PATH=/usr/local/lib \
    LOG_LEVEL=info \
    BIFROST_CONFIG=/app/data/config.json

RUN mkdir -p /data /qdrant/storage /app/data

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 7997 6333 6334 8080

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]