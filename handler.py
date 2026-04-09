import sys
import traceback

# stderr'i stdout'a yönlendir — RunPod loglarında görünsün
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
    from pathlib import Path
    print("[OK] stdlib imports done", flush=True)

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

# Model storage: Network Volume varsa /runpod-volume/models kullan
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
        # Volume bağlı — models klasörünü oluştur
        VOLUME_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        vol_path = VOLUME_MODEL_DIR / m["subfolder"] / name
        local_path = LOCAL_MODEL_DIR / m["subfolder"] / name
        if vol_path.exists() and not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.symlink_to(vol_path)
        return vol_path
    return LOCAL_MODEL_DIR / m["subfolder"] / name


def download_models():
    from huggingface_hub import hf_hub_download
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print(f"[HF] Token found: {hf_token[:8]}...{hf_token[-4:]}", flush=True)
    else:
        print("[HF] WARNING: HF_TOKEN not set in environment!", flush=True)

    for m in MODELS:
        name = m.get("dest_name", Path(m["filename"]).name)
        dest = get_model_path(m)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Symlink veya dosya zaten varsa atla
        if dest.exists() or dest.is_symlink():
            print(f"[MODEL] exists: {name}")
            # ComfyUI local path için symlink garantile
            local = LOCAL_MODEL_DIR / m["subfolder"] / name
            if not local.exists() and dest != local:
                local.parent.mkdir(parents=True, exist_ok=True)
                local.symlink_to(dest)
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

        # Local symlink
        local = LOCAL_MODEL_DIR / m["subfolder"] / name
        if not local.exists() and dest != local:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.symlink_to(dest)


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
            r = requests.get("http://127.0.0.1:8188/system_stats", timeout=2)
            if r.status_code == 200:
                print("[COMFY] ready")
                return proc
        except Exception:
            time.sleep(1)

    proc.kill()
    raise RuntimeError("ComfyUI timeout\n" + "\n".join(startup_lines[-50:]))


def download_image(url_or_b64: str, dest: Path):
    if url_or_b64.startswith("data:image") or not url_or_b64.startswith("http"):
        # base64
        header, data = url_or_b64.split(",", 1) if "," in url_or_b64 else ("", url_or_b64)
        dest.write_bytes(base64.b64decode(data))
    else:
        r = requests.get(url_or_b64, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)


# Worker başlarken bir kez çalışır
try:
    print("[INIT] Modeller indiriliyor...", flush=True)
    download_models()
    print("[INIT] ComfyUI başlatılıyor...", flush=True)
    COMFY_PROC = start_comfy()
    print("[INIT] Hazır.", flush=True)
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

    # ComfyUI çöktüyse yeniden başlat
    if COMFY_PROC.poll() is not None:
        print("[WARN] ComfyUI down, restarting...")
        COMFY_PROC = start_comfy()

    uid = uuid.uuid4().hex
    base_dst = INPUT_DIR / f"base_{uid}.png"
    face_dst = INPUT_DIR / f"face_{uid}.png"

    download_image(base_image, base_dst)
    download_image(face_image, face_dst)

    workflow = json.loads(WORKFLOW_PATH.read_bytes())
    workflow["151"]["inputs"]["image"] = base_dst.name
    workflow["121"]["inputs"]["image"] = face_dst.name
    workflow["107"]["inputs"]["text"] = prompt
    workflow["100"]["inputs"]["guidance"] = guidance
    workflow["156"]["inputs"]["steps"] = steps
    workflow["156"]["inputs"]["cfg"] = cfg
    workflow["156"]["inputs"]["seed"] = (
        int(time.time() * 1000) % 2147483647 if seed == 0 else seed
    )

    r = requests.post("http://127.0.0.1:8188/prompt", json={"prompt": workflow}, timeout=120)
    r.raise_for_status()
    prompt_data = r.json()

    if "prompt_id" not in prompt_data:
        return {"error": f"No prompt_id: {prompt_data}"}

    prompt_id = prompt_data["prompt_id"]
    print(f"[JOB] prompt_id: {prompt_id}")

    for _ in range(600):
        if COMFY_PROC.poll() is not None:
            return {"error": "ComfyUI crashed during inference"}

        h = requests.get(f"http://127.0.0.1:8188/history/{prompt_id}", timeout=30)
        if h.status_code != 200:
            time.sleep(2)
            continue

        data = h.json()
        if prompt_id in data:
            entry = data[prompt_id]
            if entry.get("status", {}).get("status_str") == "error":
                msgs = [m.get("text", "") for m in entry["status"].get("messages", [])]
                return {"error": "ComfyUI error: " + "; ".join(msgs)}

            for _, node in entry.get("outputs", {}).items():
                if "images" in node and node["images"]:
                    image = node["images"][0]
                    img = requests.get(
                        "http://127.0.0.1:8188/view",
                        params={
                            "filename": image["filename"],
                            "subfolder": image.get("subfolder", ""),
                            "type": image.get("type", "output"),
                        },
                        timeout=120,
                    )
                    img.raise_for_status()

                    b64 = base64.b64encode(img.content).decode("utf-8")

                    # Temizlik
                    try:
                        base_dst.unlink(missing_ok=True)
                        face_dst.unlink(missing_ok=True)
                    except Exception:
                        pass

                    print("[JOB] done")
                    return {"image": f"data:image/png;base64,{b64}"}

        time.sleep(2)

    return {"error": "Timeout: no output in 20 minutes"}


runpod.serverless.start({"handler": handler})
