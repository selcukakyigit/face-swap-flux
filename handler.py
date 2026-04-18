import sys
import traceback
sys.stderr = sys.stdout

print("=== HANDLER STARTING ===", flush=True)

try:
    import json
    import os
    import shutil
    import subprocess
    import time
    import uuid
    import base64
    import requests
    import websocket
    from pathlib import Path
    print("[OK] imports done", flush=True)
    import runpod
    print("[OK] runpod imported", flush=True)
except Exception as e:
    print(f"[FATAL] Import error: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

COMFY_DIR = "/comfyui"
WORKFLOW_PATH = Path("/workflow_api.json")
INPUT_DIR = Path("/comfyui/input")
OUTPUT_DIR = Path("/comfyui/output")
COMFY_HOST = "127.0.0.1:8188"

VOLUME_MODEL_DIR = Path("/runpod-volume/models")
LOCAL_MODEL_DIR = Path("/comfyui/models")

MODELS = [
    {
        "repo": "black-forest-labs/FLUX.2-klein-9B",
        "filename": "flux-2-klein-9b.safetensors",
        "subfolder": "unet",
        "token": True,
    },
    {
        "repo": "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
        "filename": "split_files/vae/flux2-vae.safetensors",
        "dest_name": "flux2-vae.safetensors",
        "subfolder": "vae",
        "token": True,
    },
    {
        "repo": "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
        "filename": "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
        "dest_name": "qwen_3_8b_fp8mixed.safetensors",
        "subfolder": "clip",
        "token": True,
    },
    {
        "repo": "Alissonerdx/BFS-Best-Face-Swap",
        "filename": "bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors",
        "subfolder": "loras",
        "token": False,
    },
]


def get_model_path(m):
    name = m.get("dest_name", Path(m["filename"]).name)
    volume_root = Path("/runpod-volume")
    if volume_root.exists():
        VOLUME_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        vol_path = VOLUME_MODEL_DIR / m["subfolder"] / name
        local_path = LOCAL_MODEL_DIR / m["subfolder"] / name
        if vol_path.exists() and not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                local_path.symlink_to(vol_path)
            except FileExistsError:
                pass
        return vol_path
    return LOCAL_MODEL_DIR / m["subfolder"] / name


def download_models():
    from huggingface_hub import hf_hub_download
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print(f"[HF] Token: {hf_token[:8]}...{hf_token[-4:]}", flush=True)
    else:
        print("[HF] WARNING: HF_TOKEN not set!", flush=True)

    for m in MODELS:
        name = m.get("dest_name", Path(m["filename"]).name)
        dest = get_model_path(m)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() or dest.is_symlink():
            print(f"[MODEL] exists: {name}")
            local = LOCAL_MODEL_DIR / m["subfolder"] / name
            if not local.exists() and dest != local:
                local.parent.mkdir(parents=True, exist_ok=True)
                try:
                    local.symlink_to(dest)
                except FileExistsError:
                    pass
            continue

        print(f"[MODEL] downloading: {name}")
        token = hf_token if m["token"] else None
        tmp = hf_hub_download(
            repo_id=m["repo"],
            filename=m["filename"],
            token=token,
            local_dir=str(dest.parent),
            local_dir_use_symlinks=False,
        )
        downloaded = Path(tmp)
        if downloaded.name != name:
            shutil.move(str(downloaded), str(dest))
        print(f"[MODEL] done: {name}")

        local = LOCAL_MODEL_DIR / m["subfolder"] / name
        if not local.exists() and dest != local:
            local.parent.mkdir(parents=True, exist_ok=True)
            try:
                local.symlink_to(dest)
            except FileExistsError:
                pass


def start_comfy():
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    proc = subprocess.Popen(
        ["python", "-u", "main.py", "--listen", "0.0.0.0", "--port", "8188",
         "--input-directory", str(INPUT_DIR)],
        cwd=COMFY_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    startup_lines = []
    for _ in range(300):
        if proc.stdout:
            line = proc.stdout.readline()
            if line:
                startup_lines.append(line.rstrip())
                print("[COMFY]", line.rstrip())

        if proc.poll() is not None:
            remaining = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(
                f"ComfyUI crashed. Exit: {proc.returncode}\n"
                + "\n".join(startup_lines[-100:])
                + (remaining or "")
            )

        try:
            r = requests.get(f"http://{COMFY_HOST}/system_stats", timeout=2)
            if r.status_code == 200:
                print("[COMFY] ready")
                return proc
        except Exception:
            time.sleep(1)

    proc.kill()
    raise RuntimeError("ComfyUI timeout\n" + "\n".join(startup_lines[-50:]))


def upload_image(image_data: bytes, filename: str) -> str:
    files = {"image": (filename, image_data, "image/png")}
    r = requests.post(f"http://{COMFY_HOST}/upload/image", files=files, timeout=60)
    r.raise_for_status()
    return r.json().get("name", filename)


def get_image_bytes(url_or_b64: str) -> bytes:
    if url_or_b64.startswith("data:image") or not url_or_b64.startswith("http"):
        _, data = url_or_b64.split(",", 1) if "," in url_or_b64 else ("", url_or_b64)
        return base64.b64decode(data)
    r = requests.get(url_or_b64, timeout=60)
    r.raise_for_status()
    return r.content


try:
    print("[INIT] Downloading models...", flush=True)
    download_models()
    print("[INIT] Starting ComfyUI...", flush=True)
    COMFY_PROC = start_comfy()
    print("[INIT] Ready.", flush=True)
except Exception as e:
    print(f"[FATAL] Init error: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)


def handler(job):
    global COMFY_PROC

    job_input = job.get("input", {})
    base_image = job_input.get("base_image")
    face_image = job_input.get("face_image")
    prompt = job_input.get("prompt", "head_swap: Use image 1 as the base image, preserving its environment, background, camera perspective, framing, exposure, contrast, and lighting. Remove the head from image 1 and seamlessly replace it with the head from image 2. Match the original head size, face-to-body ratio, neck thickness, shoulder alignment, and camera distance so proportions remain natural and unchanged. Adapt the inserted head to the lighting of image 1 by matching light direction, intensity, softness, color temperature, shadows, and highlights, with no independent relighting. Preserve the identity of image 2, including hair texture, eye color, nose structure, facial proportions, and skin details. Match the pose and expression from image 1, including head tilt, rotation, eye direction, gaze, micro-expressions, and lip position. Ensure seamless neck and jaw blending, consistent skin tone, realistic shadow contact, natural skin texture, and uniform sharpness. Photorealistic, high quality, sharp details, 4K.")
    seed = int(job_input.get("seed", 0))
    steps = int(job_input.get("steps", 4))
    guidance = float(job_input.get("guidance", 1.0))
    cfg = float(job_input.get("cfg", 1.0))

    if not base_image or not face_image:
        return {"error": "base_image and face_image are required"}

    if COMFY_PROC.poll() is not None:
        print("[WARN] ComfyUI down, restarting...")
        COMFY_PROC = start_comfy()

    uid = uuid.uuid4().hex

    # Download images and upload to ComfyUI
    base_name = upload_image(get_image_bytes(base_image), f"base_{uid}.png")
    face_name = upload_image(get_image_bytes(face_image), f"face_{uid}.png")

    # Build workflow
    workflow = json.loads(WORKFLOW_PATH.read_bytes())
    workflow["151"]["inputs"]["image"] = base_name
    workflow["121"]["inputs"]["image"] = face_name
    workflow["107"]["inputs"]["text"] = prompt
    workflow["100"]["inputs"]["guidance"] = guidance
    workflow["156"]["inputs"]["steps"] = steps
    workflow["156"]["inputs"]["cfg"] = cfg
    workflow["156"]["inputs"]["seed"] = (
        int(time.time() * 1000) % 2147483647 if seed == 0 else seed
    )

    # Submit with client_id for websocket tracking
    client_id = uuid.uuid4().hex
    r = requests.post(
        f"http://{COMFY_HOST}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=120,
    )
    r.raise_for_status()
    prompt_data = r.json()

    if "prompt_id" not in prompt_data:
        return {"error": f"No prompt_id: {prompt_data}"}

    prompt_id = prompt_data["prompt_id"]
    print(f"[JOB] prompt_id: {prompt_id}")

    # Wait for completion via websocket
    ws = websocket.WebSocket()
    ws.connect(f"ws://{COMFY_HOST}/ws?clientId={client_id}")
    ws.settimeout(600)

    try:
        while True:
            if COMFY_PROC.poll() is not None:
                return {"error": "ComfyUI crashed during inference"}

            out = ws.recv()
            if not isinstance(out, str):
                continue

            msg = json.loads(out)
            msg_type = msg.get("type")

            if msg_type == "executing":
                data = msg.get("data", {})
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    break  # Done

            elif msg_type == "execution_error":
                data = msg.get("data", {})
                return {"error": f"ComfyUI error: {json.dumps(data)}"}

    finally:
        ws.close()

    # Fetch result from history
    h = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}", timeout=30)
    h.raise_for_status()
    history = h.json()

    if prompt_id not in history:
        return {"error": "No history entry found"}

    entry = history[prompt_id]

    if entry.get("status", {}).get("status_str") == "error":
        return {"error": f"ComfyUI error: {json.dumps(entry.get('status', {}))}"}

    for _, node in entry.get("outputs", {}).items():
        if "images" in node and node["images"]:
            image = node["images"][0]
            img = requests.get(
                f"http://{COMFY_HOST}/view",
                params={
                    "filename": image["filename"],
                    "subfolder": image.get("subfolder", ""),
                    "type": image.get("type", "output"),
                },
                timeout=120,
            )
            img.raise_for_status()
            b64 = base64.b64encode(img.content).decode("utf-8")
            print("[JOB] done")
            return {"image": f"data:image/png;base64,{b64}"}

    return {"error": "No output image in results"}


runpod.serverless.start({"handler": handler})
