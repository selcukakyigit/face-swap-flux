#!/bin/bash
set -e

TOKEN=$(cat /src/.hf_token 2>/dev/null || echo "")
AUTH="Authorization: Bearer $TOKEN"

echo "[DOWNLOAD] flux-2-klein-9b.safetensors"
wget -q --show-progress --header="$AUTH" \
  -O /src/ComfyUI/models/unet/flux-2-klein-9b.safetensors \
  "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/resolve/main/flux-2-klein-9b.safetensors"

echo "[DOWNLOAD] flux2-vae.safetensors"
wget -q --show-progress --header="$AUTH" \
  -O /src/ComfyUI/models/vae/flux2-vae.safetensors \
  "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/vae/flux2-vae.safetensors"

echo "[DOWNLOAD] qwen_3_8b_fp8mixed.safetensors"
wget -q --show-progress --header="$AUTH" \
  -O /src/ComfyUI/models/clip/qwen_3_8b_fp8mixed.safetensors \
  "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors"

echo "[DOWNLOAD] bfs_head_v1 lora"
wget -q --show-progress \
  -O /src/ComfyUI/models/loras/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors \
  "https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/resolve/main/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors"

echo "[DOWNLOAD] All models done."
