import json
import shutil
import os
import subprocess
import sys
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

def ensure_comfy():
    if os.path.isdir(COMFY_DIR):
        return

    subprocess.check_call([
        "git", "clone", "--depth=1",
        "https://github.com/comfyanonymous/ComfyUI",
        COMFY_DIR
    ])

    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--no-cache-dir",
        "-r", f"{COMFY_DIR}/requirements.txt"
    ])

class Predictor(BasePredictor):
    def setup(self):
        INPUT_DIR.mkdir(exist_ok=True)
        OUTPUT_DIR.mkdir(exist_ok=True)

    def _start_comfy(self):
        ensure_comfy()
        proc = subprocess.Popen(
            ["python", "-u", "main.py", "--listen", "0.0.0.0", "--port", "8188",
             "--input-directory", str(INPUT_DIR)],
            cwd=str(COMFY_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        started = False
        startup_lines = []

        for _ in range(180):
            if proc.stdout is not None:
                line = proc.stdout.readline()
                if line:
                    startup_lines.append(line.rstrip())
                    print("[COMFY]", line.rstrip())

            try:
                r = requests.get("http://127.0.0.1:8188/system_stats", timeout=2)
                if r.status_code == 200:
                    started = True
                    print("[COMFY] system_stats OK")
                    break
            except Exception:
                time.sleep(1)

            if proc.poll() is not None:
                remaining = ""
                try:
                    if proc.stdout is not None:
                        remaining = proc.stdout.read() or ""
                except Exception:
                    pass

                full_log = startup_lines[-300:]
                if remaining:
                    full_log.append(remaining)

                raise RuntimeError(
                    f"ComfyUI erken kapandı. Exit code: {proc.returncode}\nFULL LOG:\n"
                    + "\n".join(full_log)
                )

        if not started:
            proc.kill()
            raise RuntimeError(
                "ComfyUI başlatılamadı.\n"
                + "\n".join(startup_lines[-80:])
            )

        return proc

    def _read_more_logs(self, proc, max_lines=80):
        lines = []
        if proc.stdout is None:
            return lines

        for _ in range(max_lines):
            try:
                line = proc.stdout.readline()
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
        base_dst = INPUT_DIR / f"base_{uuid.uuid4().hex}.png"
        face_dst = INPUT_DIR / f"face_{uuid.uuid4().hex}.png"

        shutil.copy(base_image, base_dst)
        shutil.copy(face_image, face_dst)

        with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        # LoadImage node'ları dosya adı bekliyor
        workflow["151"]["inputs"]["image"] = base_dst.name
        workflow["121"]["inputs"]["image"] = face_dst.name

        # Dinamik inputlar
        workflow["107"]["inputs"]["text"] = prompt
        workflow["100"]["inputs"]["guidance"] = guidance
        workflow["156"]["inputs"]["steps"] = steps
        workflow["156"]["inputs"]["cfg"] = cfg

        if seed == 0:
            workflow["156"]["inputs"]["seed"] = int(time.time() * 1000) % 2147483647
        else:
            workflow["156"]["inputs"]["seed"] = seed

        print("[INFO] base image:", base_dst.name)
        print("[INFO] face image:", face_dst.name)
        print("[INFO] seed:", workflow["156"]["inputs"]["seed"])
        print("[INFO] steps:", steps)
        print("[INFO] guidance:", guidance)
        print("[INFO] cfg:", cfg)

        proc = self._start_comfy()

        try:
            r = requests.post(
                "http://127.0.0.1:8188/prompt",
                json={"prompt": workflow},
                timeout=120,
            )

            print("[PROMPT STATUS]", r.status_code)
            print("[PROMPT RESPONSE]", r.text)

            r.raise_for_status()
            prompt_data = r.json()

            if "prompt_id" not in prompt_data:
                extra_logs = self._read_more_logs(proc, max_lines=120)
                raise RuntimeError(
                    "/prompt cevabı prompt_id döndürmedi.\n"
                    f"Response: {prompt_data}\n"
                    "ComfyUI logları:\n"
                    + "\n".join(extra_logs)
                )

            prompt_id = prompt_data["prompt_id"]
            print("[PROMPT ID]", prompt_id)

            for _ in range(300):
                if proc.poll() is not None:
                    extra_logs = self._read_more_logs(proc, max_lines=120)
                    raise RuntimeError(
                        "ComfyUI işlem sırasında kapandı.\n"
                        + "\n".join(extra_logs)
                    )

                h = requests.get(
                    f"http://127.0.0.1:8188/history/{prompt_id}",
                    timeout=120,
                )

                print("[HISTORY STATUS]", h.status_code)

                if h.status_code != 200:
                    print("[HISTORY RESPONSE]", h.text[:1500])

                h.raise_for_status()
                data = h.json()

                if prompt_id in data:
                    outputs = data[prompt_id].get("outputs", {})
                    for _, node in outputs.items():
                        if "images" in node and node["images"]:
                            image = node["images"][0]

                            params = {
                                "filename": image["filename"],
                                "subfolder": image.get("subfolder", ""),
                                "type": image.get("type", "output"),
                            }

                            img = requests.get(
                                "http://127.0.0.1:8188/view",
                                params=params,
                                timeout=120,
                            )
                            img.raise_for_status()

                            out = SysPath("/tmp") / f"{uuid.uuid4().hex}.png"
                            with open(out, "wb") as f:
                                f.write(img.content)

                            print("[SUCCESS] output:", str(out))
                            return Path(str(out))

                # Hata loglarını ara
                logs = self._read_more_logs(proc, max_lines=20)
                for line in logs:
                    lower = line.lower()
                    if (
                        "error" in lower
                        or "exception" in lower
                        or "traceback" in lower
                        or "killed" in lower
                        or "out of memory" in lower
                        or "oom" in lower
                    ):
                        raise RuntimeError(
                            "ComfyUI hata verdi:\n" + "\n".join(logs)
                        )

                time.sleep(2)

            extra_logs = self._read_more_logs(proc, max_lines=120)
            raise RuntimeError(
                "Zaman aşımı: çıktı alınamadı.\n"
                + "\n".join(extra_logs)
            )

        finally:
            try:
                proc.kill()
            except Exception:
                pass
