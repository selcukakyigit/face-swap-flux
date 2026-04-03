FROM nvidia/cuda:12.3.2-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/comfyanonymous/ComfyUI /app/ComfyUI

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install -r /app/ComfyUI/requirements.txt
RUN pip3 install requests pillow numpy

COPY . /app

CMD ["/bin/bash"]
