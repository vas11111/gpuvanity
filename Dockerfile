FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 \
        python3-click \
        python3-base58 \
        python3-nacl \
        python3-numpy \
        python3-pyopencl && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
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
