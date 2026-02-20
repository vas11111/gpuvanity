FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
        python3-click \
        python3-base58 \
        python3-nacl \
        python3-numpy && \
    pip3 install --break-system-packages pycuda && \
    rm -rf /var/lib/apt/lists/*

COPY core/ /app/core/
COPY main.py /app/

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1

WORKDIR /app
ENTRYPOINT ["python3", "main.py"]
CMD ["--help"]
