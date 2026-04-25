FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget ffmpeg \
    libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# PyTorch
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu130

# ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI /comfyui
RUN pip install --no-cache-dir -r /comfyui/requirements.txt

# Custom nodes
RUN git clone --depth=1 https://github.com/scraed/LanPaint.git /comfyui/custom_nodes/LanPaint
RUN git clone --depth=1 https://github.com/rgthree/rgthree-comfy.git /comfyui/custom_nodes/rgthree-comfy
RUN git clone --depth=1 https://github.com/cubiq/ComfyUI_essentials.git /comfyui/custom_nodes/ComfyUI_essentials

RUN pip install --no-cache-dir -e /comfyui/custom_nodes/LanPaint || true
RUN pip install --no-cache-dir -r /comfyui/custom_nodes/rgthree-comfy/requirements.txt || true
RUN pip install --no-cache-dir -r /comfyui/custom_nodes/ComfyUI_essentials/requirements.txt || true
RUN pip install --no-cache-dir -r /comfyui/custom_nodes/LanPaint/requirements.txt || true

# ComfyUI server.py pydantic fix (remove assets routes that require pydantic v2)
RUN sed -i 's/\(\s*\).*register_assets_routes.*/\1pass/' /comfyui/server.py

# RunPod + HuggingFace (pydantic v2 compatible)
RUN pip install --no-cache-dir runpod huggingface_hub requests websocket-client

# Directories
RUN mkdir -p /comfyui/models/unet \
    /comfyui/models/vae \
    /comfyui/models/clip \
    /comfyui/models/loras \
    /comfyui/input \
    /comfyui/output

COPY workflow_api.json /workflow_api.json
COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
