import json
import time
from pathlib import Path

import requests

COMFY = "http://127.0.0.1:8188"
WORKFLOW_FILE = Path("workflow_api.json")
OUT_FILE = Path("result_from_comfy.png")

with open(WORKFLOW_FILE, "r", encoding="utf-8") as f:
    workflow = json.load(f)

workflow["151"]["inputs"]["image"] = "Screenshot_29.jpg"
workflow["121"]["inputs"]["image"] = "portrait-smiley-little-kid.png"

r = requests.post(f"{COMFY}/prompt", json={"prompt": workflow}, timeout=60)
r.raise_for_status()
prompt_id = r.json()["prompt_id"]
print("prompt_id:", prompt_id)

image_info = None

for _ in range(300):
    h = requests.get(f"{COMFY}/history/{prompt_id}", timeout=30)
    h.raise_for_status()
    data = h.json()

    if prompt_id in data:
        outputs = data[prompt_id].get("outputs", {})
        for _, node_output in outputs.items():
            images = node_output.get("images", [])
            if images:
                image_info = images[0]
                break
    if image_info:
        break
    time.sleep(2)

if not image_info:
    raise RuntimeError("Çıktı bulunamadı.")

params = {
    "filename": image_info["filename"],
    "subfolder": image_info.get("subfolder", ""),
    "type": image_info.get("type", "output"),
}
img = requests.get(f"{COMFY}/view", params=params, timeout=60)
img.raise_for_status()

OUT_FILE.write_bytes(img.content)
print("Kaydedildi:", OUT_FILE.resolve())
print("image_info:", image_info)
