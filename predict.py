import json
import shutil
import os
import subprocess
import time
import uuid
from pathlib import Path as SysPath

import requests
from cog import BasePredictor, Input, Path


PROJECT_DIR = SysPath(__file__).parent
WORKFLOW_PATH = PROJECT_DIR / "workflow_api.json"
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"
COMFY_DIR = "/src/ComfyUI"

MODELS = [
    {
        "repo": "black-forest-labs/FLUX.2-klein-9B",
        "filename": "flux-2-klein-9b.safetensors",
        "dest": f"{COMFY_DIR}/models/unet/flux-2-klein-9b.safetensors",
        "token": True,
    },
    {
        "repo": "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
        "filename": "split_files/vae/flux2-vae.safetensors",
        "dest": f"{COMFY_DIR}/models/vae/flux2-vae.safetensors",
        "token": True,
    },
    {
        "repo": "Comfy-Org/vae-text-encorder-for-flux-klein-9b",
        "filename": "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
        "dest": f"{COMFY_DIR}/models/clip/qwen_3_8b_fp8mixed.safetensors",
        "token": True,
    },
    {
        "repo": "Alissonerdx/BFS-Best-Face-Swap",
        "filename": "bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors",
        "dest": f"{COMFY_DIR}/models/loras/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors",
        "token": False,
    },
]


def download_models():
    from huggingface_hub import hf_hub_download
    token_file = SysPath("/src/.hf_token")
    hf_token = token_file.read_text().strip() if token_file.exists() else os.environ.get("HF_TOKEN")

    for m in MODELS:
        dest = SysPath(m["dest"])
        if dest.exists():
            print(f"[DOWNLOAD] already exists: {dest.name}")
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[DOWNLOAD] downloading {m['filename']} from {m['repo']}...")
        token = hf_token if m["token"] else None
        tmp = hf_hub_download(
            repo_id=m["repo"],
            filename=m["filename"],
            token=token,
            local_dir=str(dest.parent),
            local_dir_use_symlinks=False,
        )
        # hf_hub_download may save with a different name — rename if needed
        downloaded = SysPath(tmp)
        if downloaded != dest:
            shutil.move(str(downloaded), str(dest))
        print(f"[DOWNLOAD] done: {dest.name}")


class Predictor(BasePredictor):
    def setup(self):
        INPUT_DIR.mkdir(exist_ok=True)
        OUTPUT_DIR.mkdir(exist_ok=True)

        print("[SETUP] Modeller indiriliyor...")
        download_models()
        print("[SETUP] Tüm modeller hazır.")

        print("[SETUP] ComfyUI başlatılıyor...")
        self.comfy_proc = self._start_comfy()
        print("[SETUP] ComfyUI hazır.")

    def _start_comfy(self):
        proc = subprocess.Popen(
            ["python", "-u", "main.py", "--listen", "0.0.0.0", "--port", "8188",
             "--input-directory", str(INPUT_DIR)],
            cwd=str(COMFY_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        startup_lines = []
        for _ in range(300):
            if proc.stdout is not None:
                line = proc.stdout.readline()
                if line:
                    startup_lines.append(line.rstrip())
                    print("[COMFY]", line.rstrip())

            if proc.poll() is not None:
                remaining = ""
                try:
                    if proc.stdout is not None:
                        remaining = proc.stdout.read() or ""
                except Exception:
                    pass
                raise RuntimeError(
                    f"ComfyUI erken kapandı. Exit code: {proc.returncode}\nFULL LOG:\n"
                    + "\n".join(startup_lines[-300:])
                    + ("\n" + remaining if remaining else "")
                )

            try:
                r = requests.get("http://127.0.0.1:8188/system_stats", timeout=2)
                if r.status_code == 200:
                    print("[COMFY] system_stats OK — hazır")
                    break
            except Exception:
                time.sleep(1)
        else:
            proc.kill()
            raise RuntimeError(
                "ComfyUI başlatılamadı (timeout).\n"
                + "\n".join(startup_lines[-80:])
            )

        return proc

    def _drain_logs(self, max_lines=50):
        lines = []
        if self.comfy_proc.stdout is None:
            return lines
        for _ in range(max_lines):
            try:
                line = self.comfy_proc.stdout.readline()
            except Exception:
                break
            if not line:
                break
            line = line.rstrip()
            lines.append(line)
            print("[COMFY]", line)
        return lines

    def predict(
        self,
        base_image: Path = Input(description="Base image"),
        face_image: Path = Input(description="Face image"),
        prompt: str = Input(
            description="Prompt",
            default="head_swap: Use image 1 as the base image, preserving its environment, background, camera perspective, framing, exposure, contrast, and lighting. Remove the head from image 1 and seamlessly replace it with the head from image 2. Match the original head size, face-to-body ratio, neck thickness, shoulder alignment, and camera distance so proportions remain natural and unchanged. Adapt the inserted head to the lighting of image 1 by matching light direction, intensity, softness, color temperature, shadows, and highlights, with no independent relighting. Preserve the identity of image 2, including hair texture, eye color, nose structure, facial proportions, and skin details. Match the pose and expression from image 1, including head tilt, rotation, eye direction, gaze, micro-expressions, and lip position. Ensure seamless neck and jaw blending, consistent skin tone, realistic shadow contact, natural skin texture, and uniform sharpness. Photorealistic, high quality, sharp details, 4K.",
        ),
        seed: int = Input(description="Seed (0 = random)", default=0),
        steps: int = Input(description="Inference steps", default=4, ge=1, le=20),
        guidance: float = Input(description="Guidance scale", default=1.0, ge=0.0, le=20.0),
        cfg: float = Input(description="CFG", default=1.0, ge=0.0, le=20.0),
    ) -> Path:

        if self.comfy_proc.poll() is not None:
            print("[WARN] ComfyUI kapanmış, yeniden başlatılıyor...")
            self.comfy_proc = self._start_comfy()

        base_dst = INPUT_DIR / f"base_{uuid.uuid4().hex}.png"
        face_dst = INPUT_DIR / f"face_{uuid.uuid4().hex}.png"
        shutil.copy(base_image, base_dst)
        shutil.copy(face_image, face_dst)

        import hashlib
        raw = WORKFLOW_PATH.read_bytes()
        workflow = json.loads(raw)
        print("[INFO] workflow md5:", hashlib.md5(raw).hexdigest())
        print("[INFO] base image:", base_dst.name)
        print("[INFO] face image:", face_dst.name)

        workflow["151"]["inputs"]["image"] = base_dst.name
        workflow["121"]["inputs"]["image"] = face_dst.name
        workflow["107"]["inputs"]["text"] = prompt
        workflow["100"]["inputs"]["guidance"] = guidance
        workflow["156"]["inputs"]["steps"] = steps
        workflow["156"]["inputs"]["cfg"] = cfg
        workflow["156"]["inputs"]["seed"] = (
            int(time.time() * 1000) % 2147483647 if seed == 0 else seed
        )

        print("[INFO] seed:", workflow["156"]["inputs"]["seed"])

        r = requests.post(
            "http://127.0.0.1:8188/prompt",
            json={"prompt": workflow},
            timeout=120,
        )
        print("[PROMPT STATUS]", r.status_code)
        r.raise_for_status()
        prompt_data = r.json()

        if "prompt_id" not in prompt_data:
            raise RuntimeError(f"/prompt prompt_id döndürmedi: {prompt_data}")

        prompt_id = prompt_data["prompt_id"]
        print("[PROMPT ID]", prompt_id)

        for _ in range(600):
            if self.comfy_proc.poll() is not None:
                logs = self._drain_logs(120)
                raise RuntimeError("ComfyUI işlem sırasında kapandı.\n" + "\n".join(logs))

            self._drain_logs(10)

            h = requests.get(f"http://127.0.0.1:8188/history/{prompt_id}", timeout=30)
            if h.status_code != 200:
                time.sleep(2)
                continue

            data = h.json()
            if prompt_id in data:
                entry = data[prompt_id]
                if entry.get("status", {}).get("status_str") == "error":
                    msgs = [m.get("text", "") for m in entry["status"].get("messages", [])]
                    raise RuntimeError("ComfyUI workflow hatası:\n" + "\n".join(msgs))

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
                        out = SysPath("/tmp") / f"{uuid.uuid4().hex}.png"
                        out.write_bytes(img.content)
                        print("[SUCCESS] output:", str(out))
                        try:
                            base_dst.unlink(missing_ok=True)
                            face_dst.unlink(missing_ok=True)
                        except Exception:
                            pass
                        return Path(str(out))

            time.sleep(2)

        raise RuntimeError("Zaman aşımı: 20 dakika içinde çıktı alınamadı.")
