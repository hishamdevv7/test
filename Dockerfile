FROM ghcr.io/huggingface/text-embeddings-inference:cpu-latest AS tei
FROM qdrant/qdrant:latest AS qdrant
FROM maximhq/bifrost:latest AS bifrost

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# TEI - copy libs and binary only
COPY --from=tei /usr/local/bin/text-embeddings-router /usr/local/bin/text-embeddings-router
COPY --from=tei /usr/local/lib /usr/local/lib
COPY --from=tei /usr/local/libfakeintel.so /usr/local/libfakeintel.so
COPY --from=tei /usr/lib /usr/lib
COPY --from=tei /lib /lib

# Qdrant
COPY --from=qdrant /qdrant /qdrant
COPY --from=qdrant /usr/lib /usr/lib
COPY --from=qdrant /lib /lib

# Bifrost
COPY --from=bifrost /app/main /app/main
COPY --from=bifrost /app/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/main /app/docker-entrypoint.sh

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
