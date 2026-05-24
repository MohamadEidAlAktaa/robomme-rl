"""Modal app for robomme-rl.

Defines the Modal `App` and a smoke-test function that verifies the image
builds correctly and the GPU + key deps (JAX, PyTorch, RoboMME) are reachable
inside the container.

Run locally:
    modal run modal/app.py::smoke_test

Or pick a GPU type:
    modal run modal/app.py::smoke_test --gpu H100
"""

from __future__ import annotations

import modal

# `modal/` is a flat directory of scripts (no __init__.py), so locally `import
# image` resolves via Python prepending the script dir to sys.path. But Modal
# only ships the entrypoint file (app.py) into the container by default — to
# make `image.py` importable inside the running function too, we explicitly
# bundle it as local python source on the image.
from image import image as _base_image
image = _base_image.add_local_python_source("image")

app = modal.App("robomme-rl", image=image)

# Persistent, cross-run storage for large model checkpoints (~tens of GB).
# Created on first use; reattached on every subsequent run. Mounted at /cache
# inside any function that lists it in its `volumes=` argument. Paths under
# /cache survive container shutdown; paths outside it do not.
CACHE_VOLUME = modal.Volume.from_name("robomme-rl-cache", create_if_missing=True)
CACHE_MOUNT = "/cache"

# Layout we maintain on the volume (matches RoboMME upstream conventions where
# possible so their scripts work unmodified):
#   /cache/openpi/openpi-assets/checkpoints/pi05_base/   ← from gs://openpi-assets
#   /cache/runs/ckpts/pi05_baseline/                     ← HF: Yinpei/pi05_baseline
#   /cache/runs/ckpts/vlm_subgoal_predictor/             ← HF: Yinpei/vlm_subgoal_predictor
#   /cache/hf/                                           ← HF hub cache (HF_HOME)
CKPTS_DIR = f"{CACHE_MOUNT}/runs/ckpts"


@app.function(gpu="A10G", timeout=600, volumes={CACHE_MOUNT: CACHE_VOLUME})
def smoke_test() -> dict:
    """Verify the image works: imports, GPU visibility, RoboMME reachable."""
    import subprocess
    import sys

    info: dict = {"python": sys.version.split()[0]}

    # GPU info via nvidia-smi.
    try:
        nvsmi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        info["nvidia_smi"] = nvsmi.stdout.strip()
    except Exception as e:
        info["nvidia_smi_error"] = repr(e)

    # JAX (RoboMME uses jax[cuda12]).
    try:
        import jax
        info["jax_version"] = jax.__version__
        info["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception as e:
        info["jax_error"] = repr(e)

    # Torch.
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_device_count"] = torch.cuda.device_count()
    except Exception as e:
        info["torch_error"] = repr(e)

    # RoboMME / openpi imports (from /app/src per the upstream Dockerfile).
    try:
        import openpi  # noqa: F401
        info["openpi_import"] = "ok"
    except Exception as e:
        info["openpi_error"] = repr(e)

    try:
        import mme_vla_suite  # noqa: F401
        info["mme_vla_suite_import"] = "ok"
    except Exception as e:
        info["mme_vla_suite_error"] = repr(e)

    return info


@app.function(
    # No GPU needed — pure I/O. Long timeout because first run pulls ~tens of
    # GB from GCS + HF (subsequent runs are near-instant; each step is
    # idempotent and skips when targets already exist on the volume).
    timeout=60 * 60 * 2,
    volumes={CACHE_MOUNT: CACHE_VOLUME},
)
def download_checkpoints(
    skip_pi05_base: bool = False,
    skip_pi05_baseline: bool = False,
    skip_subgoal_predictor: bool = False,
) -> dict:
    """Populate the cache volume with the three checkpoints we need.

    Idempotent: each step short-circuits when its target is already present,
    so re-running after a partial failure is safe.

    Returns a dict describing what was done and on-disk sizes.
    """
    import shutil
    import subprocess
    import sys
    import time
    from pathlib import Path

    report: dict = {}

    def _du_mb(p: Path) -> int:
        """Disk usage of a path in MiB, or -1 if missing. Cheap sanity-check."""
        if not p.exists():
            return -1
        out = subprocess.run(
            ["du", "-sm", str(p)], capture_output=True, text=True, check=True,
        ).stdout
        return int(out.split()[0])

    # ----- 1. π0.5 base from GCS (public openpi-assets bucket) -----
    if not skip_pi05_base:
        t0 = time.time()
        from openpi.shared import download as openpi_download
        # `maybe_download` is openpi's own helper: it resolves the gs:// URI to
        # a local path under $OPENPI_DATA_HOME (which we set to /cache/openpi
        # in image.py) and downloads only if not already present.
        local_path = openpi_download.maybe_download(
            "gs://openpi-assets/checkpoints/pi05_base"
        )
        report["pi05_base"] = {
            "local_path": str(local_path),
            "size_mb": _du_mb(Path(local_path)),
            "elapsed_s": round(time.time() - t0, 1),
        }
    else:
        report["pi05_base"] = "skipped"

    # ----- 2. & 3. HuggingFace checkpoints -----
    # We use huggingface_hub.snapshot_download instead of `git clone` because
    # (a) it doesn't need git-lfs configured, (b) it has resumable downloads
    # and integrity checks, (c) it's idempotent via the HF cache.
    from huggingface_hub import snapshot_download

    Path(CKPTS_DIR).mkdir(parents=True, exist_ok=True)

    def _fetch_hf(repo_id: str, subdir: str) -> dict:
        t0 = time.time()
        local_dir = Path(CKPTS_DIR) / subdir
        local_dir.mkdir(parents=True, exist_ok=True)
        # snapshot_download uses HF_HOME (set to /cache/hf in image.py) as the
        # content-addressed blob cache, and populates local_dir via symlinks on
        # Linux. Resumable + dedup'd across repos.
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
        return {
            "local_dir": str(local_dir),
            "size_mb": _du_mb(local_dir),
            "elapsed_s": round(time.time() - t0, 1),
        }

    if not skip_pi05_baseline:
        report["pi05_baseline"] = _fetch_hf("Yinpei/pi05_baseline", "pi05_baseline")
    else:
        report["pi05_baseline"] = "skipped"

    if not skip_subgoal_predictor:
        report["vlm_subgoal_predictor"] = _fetch_hf(
            "Yinpei/vlm_subgoal_predictor", "vlm_subgoal_predictor",
        )
    else:
        report["vlm_subgoal_predictor"] = "skipped"

    # ----- 4. Unzip any .zip files in runs/ckpts (RoboMME ships ckpts zipped) -----
    # Re-use upstream's own unzipper rather than re-implementing it. It's at
    # /app/scripts/unzip_ckpt.py in the image (copied in by the Dockerfile),
    # and the `unzip_one` helper there is idempotent.
    sys.path.insert(0, "/app/scripts")
    try:
        from unzip_ckpt import find_zip_files, unzip_one  # type: ignore
    finally:
        sys.path.pop(0)

    t0 = time.time()
    zips = find_zip_files(Path(CKPTS_DIR))
    for z in zips:
        unzip_one(z, overwrite=False)
    report["unzip"] = {
        "n_zips_found": len(zips),
        "elapsed_s": round(time.time() - t0, 1),
    }

    # Flush volume writes to durable storage so other functions see them. Modal
    # commits at container shutdown too, but doing it explicitly here makes the
    # state visible right away to anything else we kick off in parallel.
    CACHE_VOLUME.commit()

    report["disk_free_gb"] = round(shutil.disk_usage(CACHE_MOUNT).free / 1e9, 1)
    return report


@app.function(gpu="A10G", timeout=900, volumes={CACHE_MOUNT: CACHE_VOLUME})
def test_load_planner() -> dict:
    """Load the SFT'd Qwen3-VL-4B planner + LoRA adapter and report VRAM.

    The planner lives in RoboMME's `robomme` micromamba env (it depends on
    ms_swift + transformers 4.57.3 + torch 2.9.1, which deliberately don't
    match the uv env that runs JAX/openpi). So we shell out via
    `micromamba run -n robomme` and parse a JSON line from stdout.

    First call also pulls Qwen/Qwen3-VL-4B-Instruct (~9 GB) from HuggingFace
    into HF_HOME (=/cache/hf on the volume), so subsequent calls are fast.

    A pass here means: the model + adapter load cleanly on an A10G and we
    know our VRAM budget for the planner side of the RL loop.
    """
    import json
    import subprocess
    import textwrap

    adapter_path = (
        f"{CKPTS_DIR}/vlm_subgoal_predictor/qwenvl/"
        "grounded_subgoal/checkpoint-1200"
    )

    # The script runs inside the `robomme` micromamba env. We deliberately
    # use attn_impl='sdpa' instead of upstream's 'flash_attention_2' because
    # flash-attn isn't pinned in robomme's requirements and we don't want a
    # build-time landmine on first invocation. SDPA is PyTorch-native and
    # supports the same Qwen3-VL forward path.
    script = textwrap.dedent(f"""
        import json, time, sys, traceback
        try:
            import torch
            t0 = time.time()
            from swift.llm import PtEngine
            engine = PtEngine(
                model_id_or_path='Qwen/Qwen3-VL-4B-Instruct',
                adapters=[{adapter_path!r}],
                attn_impl='sdpa',
            )
            torch.cuda.synchronize()
            out = {{
                'ok': True,
                'load_s': round(time.time() - t0, 1),
                'torch_version': torch.__version__,
                'vram_alloc_mb': torch.cuda.memory_allocated() // 1024**2,
                'vram_peak_mb': torch.cuda.max_memory_allocated() // 1024**2,
                'vram_total_mb': torch.cuda.get_device_properties(0).total_memory // 1024**2,
            }}
        except Exception as e:
            out = {{'ok': False, 'error': repr(e), 'traceback': traceback.format_exc()}}
        # Sentinel-wrapped so we can find it in noisy stdout from swift/HF.
        print('__RESULT_BEGIN__' + json.dumps(out) + '__RESULT_END__')
    """)

    result = subprocess.run(
        ["micromamba", "run", "-n", "robomme", "python", "-c", script],
        capture_output=True, text=True, timeout=60 * 14,
    )

    report: dict = {"returncode": result.returncode}
    # Find the sentinel-wrapped JSON anywhere in stdout (swift + HF chatter
    # before it is normal).
    stdout = result.stdout
    start = stdout.find("__RESULT_BEGIN__")
    end = stdout.find("__RESULT_END__")
    if start != -1 and end != -1:
        payload = stdout[start + len("__RESULT_BEGIN__"):end]
        try:
            report["result"] = json.loads(payload)
        except json.JSONDecodeError as e:
            report["parse_error"] = repr(e)
            report["payload_sample"] = payload[:500]
    else:
        # Script crashed before printing sentinel. Surface tails for debug.
        report["stdout_tail"] = stdout[-2000:]
        report["stderr_tail"] = result.stderr[-2000:]
    return report


@app.function(gpu="A10G", timeout=1500, volumes={CACHE_MOUNT: CACHE_VOLUME})
def test_serve_policy() -> dict:
    """Boot serve_policy.py for pi05_baseline, send one client ping, tear down.

    Mirrors RoboMME's upstream two-process design:
      * server (this function's default env = uv venv) — JAX/openpi, runs the
        π0.5 VLA, listens on a localhost websocket.
      * client (micromamba `robomme` env) — uses openpi-client to handshake +
        send a reset() ping.

    Pass conditions:
      1. serve_policy.py boots, binds to the port, doesn't crash.
      2. The client handshake succeeds and server metadata round-trips.
      3. client.reset() returns a parseable response.

    Doesn't run a real inference (would need valid sim obs); that's the next
    test. This isolates "can the VLA serve at all" from "do we have the right
    observation shapes".
    """
    import json
    import socket
    import subprocess
    import textwrap
    import time

    PORT = 8011
    VLA_CKPT_DIR = f"{CKPTS_DIR}/pi05_baseline/pi05_baseline/79999"
    SERVER_LOG = "/tmp/serve_policy.log"

    # ----- 1. Launch serve_policy.py in background (uv env, our default) -----
    # Tyro subcommand syntax (matches upstream scripts/eval.sh):
    #   positional `policy:checkpoint` selects the Checkpoint variant of
    #   the `policy: Checkpoint | Default` union; --policy.{dir,config}
    #   populate Checkpoint's fields.
    server_log_fp = open(SERVER_LOG, "wb")
    server = subprocess.Popen(
        [
            "python", "/app/scripts/serve_policy.py",
            f"--port={PORT}",
            "policy:checkpoint",
            f"--policy.dir={VLA_CKPT_DIR}",
            "--policy.config=pi05_baseline",
        ],
        stdout=server_log_fp,
        stderr=subprocess.STDOUT,
        cwd="/app",
    )

    def _read_log_tail(n: int = 3000) -> str:
        # Safe to call after _kill_server() has closed server_log_fp.
        try:
            if not server_log_fp.closed:
                server_log_fp.flush()
        except (ValueError, OSError):
            pass
        try:
            with open(SERVER_LOG, "r") as f:
                return f.read()[-n:]
        except Exception as e:  # noqa: BLE001
            return f"<could not read log: {e!r}>"

    def _kill_server() -> None:
        try:
            server.terminate()
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        server_log_fp.close()

    # ----- 2. Wait for the port to open (or the server process to die) -----
    # pi0.5 cold load: weight read from /cache + JAX kernel compile. Empirically
    # 1-3 min on first invocation in this image.
    BOOT_TIMEOUT_S = 600
    boot_t0 = time.time()
    ready = False
    while time.time() - boot_t0 < BOOT_TIMEOUT_S:
        if server.poll() is not None:
            break  # process died during startup
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                ready = True
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(2)
    boot_s = round(time.time() - boot_t0, 1)

    if not ready:
        rc = server.returncode
        _kill_server()
        return {
            "ok": False,
            "phase": "boot",
            "elapsed_s": boot_s,
            "server_returncode": rc,
            "server_log_tail": _read_log_tail(),
        }

    # ----- 3. GPU memory snapshot (server is the sole GPU user in this container) -----
    nvsmi = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    try:
        used_str, total_str = nvsmi.stdout.strip().split(",")
        vram_used_mb = int(used_str.strip())
        vram_total_mb = int(total_str.strip())
    except Exception:  # noqa: BLE001
        vram_used_mb = vram_total_mb = -1

    # ----- 4. Ping the server from the robomme env (where openpi-client lives) -----
    client_script = textwrap.dedent(f"""
        import json, time, traceback
        try:
            from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy
            # Constructor blocks until handshake + metadata received.
            t0 = time.time()
            client = MMEVLAWebsocketClientPolicy('127.0.0.1', {PORT})
            handshake_s = round(time.time() - t0, 2)

            md = client.get_server_metadata() or {{}}

            t0 = time.time()
            resp = client.reset() or {{}}
            reset_s = round(time.time() - t0, 2)

            out = {{
                'ok': True,
                'handshake_s': handshake_s,
                'reset_rtt_s': reset_s,
                # Truncate any individual value to keep the payload small.
                'server_metadata_keys': sorted(md.keys()),
                'reset_response': {{k: str(v)[:200] for k, v in resp.items()}},
            }}
        except Exception as e:
            out = {{'ok': False, 'error': repr(e), 'traceback': traceback.format_exc()}}
        print('__RESULT_BEGIN__' + json.dumps(out, default=str) + '__RESULT_END__')
    """)

    client_proc = subprocess.run(
        ["micromamba", "run", "-n", "robomme", "python", "-c", client_script],
        capture_output=True, text=True, timeout=120,
    )

    cs = client_proc.stdout
    a = cs.find("__RESULT_BEGIN__")
    b = cs.find("__RESULT_END__")
    if a >= 0 and b > a:
        try:
            client_report = json.loads(cs[a + len("__RESULT_BEGIN__"):b])
        except json.JSONDecodeError as e:
            client_report = {"parse_error": repr(e), "raw_tail": cs[-1500:]}
    else:
        client_report = {
            "no_sentinel": True,
            "stdout_tail": cs[-1500:],
            "stderr_tail": client_proc.stderr[-1500:],
        }

    # ----- 5. Tear down server -----
    _kill_server()

    return {
        "ok": bool(client_report.get("ok")),
        "server_boot_s": boot_s,
        "vram_used_mb": vram_used_mb,
        "vram_total_mb": vram_total_mb,
        "client": client_report,
        # Last bit of server log on success too — lets us spot any warnings
        # without having to redrive the test.
        "server_log_tail": _read_log_tail(1000),
    }


@app.function(gpu="A10G", timeout=300)
def sapien_vk_probe() -> dict:
    """Find out which libvulkan.so.1 Sapien actually links against, and what
    that loader sees when asked to create an instance.

    Hypothesis: Sapien (inside the robomme micromamba env) is using a
    libvulkan.so.1 that's different from the system one we've been configuring,
    so our VK_ICD_FILENAMES / ICD-manifest tweaks haven't been reaching it.
    """
    import os
    import subprocess

    def _sh(cmd: str, timeout: int = 30) -> str:
        try:
            r = subprocess.run(
                ["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout,
            )
            return f"rc={r.returncode}\n--stdout--\n{r.stdout}\n--stderr--\n{r.stderr}"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e!r}"

    report = {}
    # Where do libvulkan.so.* files live, system-wide?
    report["all_libvulkan"] = _sh("find / -name 'libvulkan.so*' 2>/dev/null")
    # Where is the lavapipe ICD JSON?
    report["lavapipe_icd"] = _sh(
        "ls -la /usr/share/vulkan/icd.d/ && cat /usr/share/vulkan/icd.d/lvp_icd.x86_64.json 2>/dev/null"
    )
    # What sapien.so does the robomme env load, and what does it link to?
    report["sapien_ldd"] = _sh(
        "find /root/.local/share/mamba/envs/robomme -name 'pysapien*' -o -name 'sapien*.so*' 2>/dev/null | head -5 "
        "&& for f in $(find /root/.local/share/mamba/envs/robomme -name 'pysapien*.so' 2>/dev/null); do "
        "  echo \"== $f ==\"; ldd \"$f\" 2>&1 | grep -Ei 'vulkan|nvidia|GLX' || true; "
        "done"
    )
    # Try vkCreateInstance from INSIDE the robomme env via ctypes.
    # This bypasses Sapien entirely and tells us if the loader-in-this-env works.
    report["robomme_vk_instance_test"] = _sh(
        "micromamba run -n robomme python -c "
        "\"import os, ctypes; "
        "print('VK_ICD_FILENAMES:', os.environ.get('VK_ICD_FILENAMES','<unset>')); "
        "print('which libvulkan loaded by ctypes:'); "
        "v = ctypes.CDLL('libvulkan.so.1'); "
        "print('  ', v._name); "
        "print('attempt vkCreateInstance via ctypes...'); "
        "import struct; "
        "VK_STRUCTURE_TYPE_APPLICATION_INFO = 0; "
        "VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1; "
        "print('symbols defined:', hasattr(v,'vkCreateInstance'))\"",
        timeout=60,
    )
    # And vulkaninfo from inside robomme (in case the env has it):
    report["vulkaninfo_in_robomme"] = _sh(
        "micromamba run -n robomme vulkaninfo --summary 2>&1 | head -30",
        timeout=30,
    )
    # And vulkaninfo from system env (uv) for comparison:
    report["vulkaninfo_in_system"] = _sh(
        "vulkaninfo --summary 2>&1 | head -30",
        timeout=30,
    )
    return report


@app.function(gpu="A10G", timeout=300)
def find_vulkan_icd() -> dict:
    """Scan every nvidia/GLX lib and report which one exports the Vulkan ICD
    entry points. Whichever lib has `vk_icdGetInstanceProcAddr` is the one
    our ICD JSON should reference.
    """
    import os
    import subprocess

    def _sh(cmd: str, timeout: int = 30) -> str:
        try:
            r = subprocess.run(
                ["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout,
            )
            return f"rc={r.returncode}\n--stdout--\n{r.stdout}\n--stderr--\n{r.stderr}"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {e!r}"

    report = {}
    # Scan every nvidia/GLX/vulkan-ish lib for the ICD entry-point symbols.
    # Whichever one exports `vk_icdGetInstanceProcAddr` is the real ICD; the
    # Vulkan loader uses that symbol as the well-known handshake.
    report["icd_symbol_scan"] = _sh(
        "for L in "
        "/usr/lib/x86_64-linux-gnu/libGLX_*.so* "
        "/usr/lib/x86_64-linux-gnu/libEGL_*.so* "
        "/usr/lib/x86_64-linux-gnu/libnvidia-*.so* "
        "/usr/lib/x86_64-linux-gnu/libvk*.so* "
        "/usr/local/nvidia/lib/libGLX_*.so* "
        "/usr/local/nvidia/lib64/libGLX_*.so* "
        "/usr/local/nvidia/lib64/libnvidia-*.so* ; do "
        "  [ -e \"$L\" ] || continue; "
        "  n=$(nm -D --defined-only \"$L\" 2>/dev/null "
        "      | grep -E 'vk_icdGetInstanceProcAddr|vk_icdNegotiateLoaderICDInterfaceVersion' "
        "      | wc -l); "
        "  [ \"$n\" -gt 0 ] && echo \"HIT  $L : $n vk_icd symbols\"; "
        "done"
    )
    # Re-run vulkaninfo with loader debug on so we see every ICD path tried.
    report["vulkaninfo_with_debug"] = _sh(
        "VK_LOADER_DEBUG=all vulkaninfo --summary 2>&1 | head -80",
        timeout=30,
    )
    # Any other ICD files on the filesystem we didn't put down?
    report["all_icd_jsons"] = _sh(
        "find /etc /usr -name '*icd*.json' -type f 2>/dev/null",
    )
    # What is libGLX_nvidia.so.0 actually exporting? (sanity check our diagnosis)
    report["libglx_nvidia_top_exports"] = _sh(
        "nm -D --defined-only /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 "
        "2>/dev/null | head -20"
    )
    return report


@app.function(gpu="A10G", timeout=300)
def vulkan_debug() -> dict:
    """Dump everything we need to diagnose why Sapien can't init Vulkan.

    Runs *no* sim code — just shells out to standard tools to inspect what
    the Vulkan loader can see. Output is meant to be human-read.
    """
    import os
    import subprocess

    def _run(cmd: list[str], timeout: int = 15) -> str:
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
            return f"$ {' '.join(cmd)}\nrc={r.returncode}\n--stdout--\n{r.stdout}\n--stderr--\n{r.stderr}"
        except Exception as e:  # noqa: BLE001
            return f"$ {' '.join(cmd)}\nERROR: {e!r}"

    report = {}
    report["nvidia_smi"] = _run(["nvidia-smi"])
    report["env_caps"] = {
        k: os.environ.get(k, "<unset>")
        for k in (
            "NVIDIA_DRIVER_CAPABILITIES",
            "NVIDIA_VISIBLE_DEVICES",
            "VK_ICD_FILENAMES",
            "VK_LAYER_PATH",
            "LD_LIBRARY_PATH",
            "SAPIEN_RENDER_DEVICE",
        )
    }
    # What ICDs does the loader actually see?
    report["icd_dir_listing"] = _run(["ls", "-la", "/usr/share/vulkan/icd.d/"])
    if os.path.exists("/usr/share/vulkan/icd.d/nvidia_icd.json"):
        with open("/usr/share/vulkan/icd.d/nvidia_icd.json") as f:
            report["nvidia_icd_json_content"] = f.read()
    # Driver libs the container toolkit was supposed to inject.
    report["libglx_nvidia"] = _run(
        ["bash", "-c",
         "ls -la /usr/lib/x86_64-linux-gnu/libGLX_nvidia* "
         "/usr/lib/x86_64-linux-gnu/libEGL_nvidia* "
         "/usr/lib/x86_64-linux-gnu/libnvidia-glcore* 2>&1 | head -30"],
    )
    report["ldconfig_nvidia"] = _run(
        ["bash", "-c", "ldconfig -p | grep -Ei 'nvidia|vulkan|glx' | head -40"],
    )
    # Definitive answer: does Vulkan see the GPU?
    report["vulkaninfo_summary"] = _run(["vulkaninfo", "--summary"], timeout=30)
    return report


@app.function(gpu="A10G", timeout=1800, volumes={CACHE_MOUNT: CACHE_VOLUME})
def test_oracle_episode(
    task_name: str = "MoveCube",
    max_steps: int = 200,
) -> dict:
    """Run ONE Oracle-driven episode end-to-end → first real `R ∈ {0,1}`.

    Pipeline (matches the proposal's Appendix A, with Oracle as planner):
      Oracle  → grounded_subgoal_oracle from sim state (no NN)
      VLA     → π0.5 baseline served on localhost:8011 (uv env)
      Sim     → ManiSkill + lavapipe (CPU Vulkan) + CUDA-shim monkey-patches

    Doesn't load Qwen3-VL (next step) — validates wiring + gives Kamal his
    first Oracle baseline number without VRAM contention.
    """
    import json
    import socket
    import subprocess
    import textwrap
    import time

    PORT = 8011
    VLA_CKPT_DIR = f"{CKPTS_DIR}/pi05_baseline/pi05_baseline/79999"
    SERVER_LOG = "/tmp/serve_policy.log"

    # ----- 1. Boot serve_policy.py in background (uv env, default PATH) -----
    server_log_fp = open(SERVER_LOG, "wb")
    server = subprocess.Popen(
        [
            "python", "/app/scripts/serve_policy.py",
            f"--port={PORT}",
            "policy:checkpoint",
            f"--policy.dir={VLA_CKPT_DIR}",
            "--policy.config=pi05_baseline",
        ],
        stdout=server_log_fp,
        stderr=subprocess.STDOUT,
        cwd="/app",
    )

    def _read_log_tail(n: int = 2000) -> str:
        try:
            if not server_log_fp.closed:
                server_log_fp.flush()
        except (ValueError, OSError):
            pass
        try:
            with open(SERVER_LOG, "r") as f:
                return f.read()[-n:]
        except Exception as e:  # noqa: BLE001
            return f"<could not read log: {e!r}>"

    def _kill_server() -> None:
        try:
            server.terminate()
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        server_log_fp.close()

    # Wait for port to open or process to die.
    boot_t0 = time.time()
    ready = False
    while time.time() - boot_t0 < 600:
        if server.poll() is not None:
            break
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                ready = True
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(2)
    boot_s = round(time.time() - boot_t0, 1)

    if not ready:
        rc = server.returncode
        _kill_server()
        return {
            "ok": False,
            "phase": "server_boot",
            "elapsed_s": boot_s,
            "server_returncode": rc,
            "server_log_tail": _read_log_tail(),
        }

    # ----- 2. Run the Oracle episode in the robomme env -----
    # Variables substituted at Python level; the body is a normal string so we
    # don't have to escape `{` `}` for f-string formatting.
    header = (
        f"PORT = {PORT}\n"
        f"MAX_STEPS = {max_steps}\n"
        f"TASK_NAME = {task_name!r}\n"
    )
    body = textwrap.dedent("""
        import json, os, sys, time, traceback
        from pathlib import Path

        # Same sapien monkey-patches that make test_sim work, applied BEFORE
        # any upstream module imports sapien.
        import sapien
        _orig_rs = sapien.render.RenderSystem
        def _patched_rs(_device_ignored, *args, **kwargs):
            return _orig_rs('cpu', *args, **kwargs)
        sapien.render.RenderSystem = _patched_rs

        import numpy as _np
        import torch as _torch
        _cam_cls = sapien.render.RenderCameraComponent
        class _CudaShim:
            __slots__ = ('_arr',)
            def __init__(self, arr):
                self._arr = arr
            def torch(self):
                return _torch.as_tensor(_np.asarray(self._arr))
        def _patched_get_picture_cuda(self, name):
            return _CudaShim(self.get_picture(name))
        _cam_cls.get_picture_cuda = _patched_get_picture_cuda

        # Upstream eval.py sets these — harmless for Oracle path.
        os.environ['IMAGE_MAX_TOKEN_NUM'] = '256'
        os.environ['VIDEO_MAX_TOKEN_NUM'] = '64'
        os.environ['FPS_MAX_FRAMES'] = '10'

        sys.path.insert(0, '/app/examples/robomme')

        out = {}
        try:
            from env_runner import EnvRunner
            from subgoal_predictor import build_subgoal_predictor
            from eval import EpisodeEvaluator, Args

            # Build Args manually (eval.py uses tyro.cli, but Args is a plain
            # dataclass we can construct directly). Only flags Oracle path needs:
            args = Args(
                host='127.0.0.1',
                port=PORT,
                use_oracle=True,
                subgoal_type='grounded_subgoal',
                policy_name='oracle_smoke',
                model_ckpt_id=79999,
                max_steps=MAX_STEPS,
                save_dir='/tmp/oracle_eval',
                overwrite=True,
            )

            save_dir = Path(args.save_dir) / 'oracle_smoke'
            save_dir.mkdir(parents=True, exist_ok=True)
            video_dir = save_dir / 'videos'
            video_dir.mkdir(parents=True, exist_ok=True)

            t0 = time.time()
            env_runner = EnvRunner(
                env_id=TASK_NAME,
                video_save_dir=str(video_dir),
                max_steps=args.max_steps,
            )
            env_runner.make_env(0)
            setup_s = round(time.time() - t0, 1)

            subgoal_predictor = build_subgoal_predictor(args, save_dir)
            evaluator = EpisodeEvaluator(args, save_dir)

            t0 = time.time()
            success_flag = evaluator.eval_each_episode(
                env_runner, subgoal_predictor, video_dir,
            )
            episode_s = round(time.time() - t0, 1)
            env_runner.close_env()

            out = {
                'ok': True,
                'task': TASK_NAME,
                'episode_id': 0,
                'success_flag': success_flag,
                'reward': int(success_flag == 'success'),
                'setup_s': setup_s,
                'episode_s': episode_s,
            }
        except Exception as e:
            out = {
                'ok': False,
                'error': repr(e),
                'traceback': traceback.format_exc(),
            }

        print('__RESULT_BEGIN__' + json.dumps(out, default=str) + '__RESULT_END__')
    """)
    client_script = header + body

    t0 = time.time()
    client_proc = subprocess.run(
        ["micromamba", "run", "-n", "robomme", "python", "-c", client_script],
        capture_output=True, text=True, timeout=60 * 20,
        cwd="/app/examples/robomme",
    )
    episode_total_s = round(time.time() - t0, 1)

    # Parse sentinel-wrapped JSON.
    cs = client_proc.stdout
    a = cs.find("__RESULT_BEGIN__")
    b = cs.find("__RESULT_END__")
    if a >= 0 and b > a:
        try:
            client_report = json.loads(cs[a + len("__RESULT_BEGIN__"):b])
        except json.JSONDecodeError as e:
            client_report = {"parse_error": repr(e), "raw_tail": cs[-2500:]}
    else:
        client_report = {
            "no_sentinel": True,
            "stdout_tail": cs[-2500:],
            "stderr_tail": client_proc.stderr[-2500:],
        }

    _kill_server()

    return {
        "ok": bool(client_report.get("ok")),
        "server_boot_s": boot_s,
        "episode_total_s": episode_total_s,
        "client": client_report,
        "server_log_tail": _read_log_tail(800),
    }


@app.function(
    # 10-hour timeout to support up to ~50 episodes of Qwen3-VL at ~6 min each
    # (the proposal's eval split is 50 episodes per task). Episode loop is
    # serial within one container; parallel collection is the next layer up
    # (Modal `.map()` over multiple containers).
    gpu="A10G:2", timeout=10 * 3600,
    volumes={CACHE_MOUNT: CACHE_VOLUME},
)
def collect_rollouts(
    task_name: str = "MoveCube",
    n_episodes: int = 5,
    planner: str = "qwenvl",
    max_steps: int = 200,
) -> dict:
    """Boot the VLA server + planner ONCE, then run `n_episodes` of `task_name`.

    Returns a per-episode list of {episode_id, success_flag, reward, episode_s}
    plus a summary success_rate. This is the building block for:
      * Baseline reproduction — `collect_rollouts("MoveCube", 50, "oracle")`
        for Kamal's Oracle numbers, ditto with `"qwenvl"` for SFT numbers.
      * RestEM iteration step — collect N rollouts with the current planner
        LoRA, filter to successes, SFT-update on the (state, memory, subgoal)
        tuples from successes, swap in the new adapter, repeat.
      * Parallel scaling later — wrap this in Modal `.map()` to fan out
        across multiple A10G:2 containers, one per (task, seed) pair.

    `planner`:
      * 'oracle': uses RoboMME's `OracleSubgoalPredictor` (reads ground-truth
        sim state, no neural net, no GPU 1 usage).
      * 'qwenvl': uses `QwenVLSubgoalPredictor` (Qwen3-VL-4B + grounded LoRA
        on GPU 1).
    """
    import json
    import os
    import socket
    import subprocess
    import textwrap
    import time

    assert planner in ("oracle", "qwenvl"), f"unknown planner: {planner!r}"

    PORT = 8011
    VLA_CKPT_DIR = f"{CKPTS_DIR}/pi05_baseline/pi05_baseline/79999"
    QWEN_ADAPTER_PATH = (
        f"{CKPTS_DIR}/vlm_subgoal_predictor/qwenvl/"
        "grounded_subgoal/checkpoint-1200"
    )
    SERVER_LOG = "/tmp/serve_policy.log"

    # ----- 1. Boot serve_policy.py on GPU 0 -----
    server_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    server_log_fp = open(SERVER_LOG, "wb")
    server = subprocess.Popen(
        [
            "python", "/app/scripts/serve_policy.py",
            f"--port={PORT}",
            "policy:checkpoint",
            f"--policy.dir={VLA_CKPT_DIR}",
            "--policy.config=pi05_baseline",
        ],
        stdout=server_log_fp,
        stderr=subprocess.STDOUT,
        cwd="/app",
        env=server_env,
    )

    def _read_log_tail(n: int = 2000) -> str:
        try:
            if not server_log_fp.closed:
                server_log_fp.flush()
        except (ValueError, OSError):
            pass
        try:
            with open(SERVER_LOG, "r") as f:
                return f.read()[-n:]
        except Exception as e:  # noqa: BLE001
            return f"<could not read log: {e!r}>"

    def _kill_server() -> None:
        try:
            server.terminate()
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        server_log_fp.close()

    boot_t0 = time.time()
    ready = False
    while time.time() - boot_t0 < 600:
        if server.poll() is not None:
            break
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                ready = True
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(2)
    boot_s = round(time.time() - boot_t0, 1)

    if not ready:
        rc = server.returncode
        _kill_server()
        return {
            "ok": False,
            "phase": "server_boot",
            "elapsed_s": boot_s,
            "server_returncode": rc,
            "server_log_tail": _read_log_tail(),
        }

    # ----- 2. Multi-episode loop in the robomme env -----
    # Volume-resident dirs so episode videos + structured trajectory JSONs
    # survive container shutdown. Layout:
    #   /cache/runs/collect/{planner}/{task}/videos/{env}_ep{N}_{flag}_*.mp4
    #   /cache/runs/collect/{planner}/{task}/videos/{env}_ep{N}_{flag}_*.json
    # The JSONs are what RestEM iteration 0 will read tomorrow to build the
    # SFT dataset from successful trajectories.
    COLLECT_DIR = "/cache/runs/collect"

    header = (
        f"PORT = {PORT}\n"
        f"N_EPISODES = {n_episodes}\n"
        f"MAX_STEPS = {max_steps}\n"
        f"TASK_NAME = {task_name!r}\n"
        f"PLANNER = {planner!r}\n"
        f"QWEN_ADAPTER_PATH = {QWEN_ADAPTER_PATH!r}\n"
        f"COLLECT_DIR = {COLLECT_DIR!r}\n"
    )
    body = textwrap.dedent("""
        import json, os, sys, time, traceback
        from pathlib import Path

        # Sapien monkey-patches (same as Oracle/QwenVL one-shot tests).
        import sapien
        _orig_rs = sapien.render.RenderSystem
        def _patched_rs(_device_ignored, *args, **kwargs):
            return _orig_rs('cpu', *args, **kwargs)
        sapien.render.RenderSystem = _patched_rs

        import numpy as _np
        import torch as _torch
        _cam_cls = sapien.render.RenderCameraComponent
        class _CudaShim:
            __slots__ = ('_arr',)
            def __init__(self, arr):
                self._arr = arr
            def torch(self):
                return _torch.as_tensor(_np.asarray(self._arr))
        def _patched_get_picture_cuda(self, name):
            return _CudaShim(self.get_picture(name))
        _cam_cls.get_picture_cuda = _patched_get_picture_cuda

        os.environ['IMAGE_MAX_TOKEN_NUM'] = '256'
        os.environ['VIDEO_MAX_TOKEN_NUM'] = '64'
        os.environ['FPS_MAX_FRAMES'] = '10'

        sys.path.insert(0, '/app/examples/robomme')

        out = {'episodes': []}
        try:
            from env_runner import EnvRunner
            from subgoal_predictor import build_subgoal_predictor
            from eval import EpisodeEvaluator, Args
            from utils import RolloutRecorder

            # ---- RolloutRecorder monkey-patches: also dump structured JSON ----
            # Upstream RolloutRecorder records per-step (state, action, subgoal)
            # into composited video frames but never exposes them as data.
            # Patch __init__ to add a structured buffer, .record to append, and
            # .save_video to also write the buffer to JSON alongside the video.
            # The JSON is what RestEM consumes to build SFT training data from
            # successful trajectories — episode_id + success_flag are encoded
            # in the filename by upstream's save_video call.
            import json as _json
            import imageio as _imageio
            _orig_recorder_init = RolloutRecorder.__init__
            _orig_recorder_record = RolloutRecorder.record
            _orig_recorder_save_video = RolloutRecorder.save_video

            def _patched_init(self, save_dir, task_goal, fps=30):
                _orig_recorder_init(self, save_dir, task_goal, fps)
                self._traj_steps = []
                self._task_goal_text = task_goal
                self._last_subgoal = None
                # Pending raw images for subgoal-change moments. We hold them
                # in memory and write to disk in save_video() so the file
                # naming can incorporate the success_flag.
                self._pending_imgs = []

            _INIT_MARKER = '[initializing...]'

            def _patched_record(self, image, wrist_image, state, action=None,
                                is_video_demo=False, subgoal=None):
                _orig_recorder_record(self, image, wrist_image, state, action,
                                      is_video_demo, subgoal)
                step_idx = len(self._traj_steps)
                is_real_subgoal = (
                    subgoal is not None
                    and subgoal != _INIT_MARKER
                    and not is_video_demo
                )
                is_new_subgoal = (
                    is_real_subgoal and subgoal != self._last_subgoal
                )
                self._traj_steps.append({
                    'step_idx': step_idx,
                    'state': state.tolist() if hasattr(state, 'tolist') else list(state),
                    'action': (action.tolist() if action is not None
                               and hasattr(action, 'tolist')
                               else (list(action) if action is not None else None)),
                    'subgoal': subgoal,
                    'is_video_demo': is_video_demo,
                    'is_new_subgoal': is_new_subgoal,
                })
                if is_new_subgoal:
                    # Buffer raw observations at this moment — these become
                    # SFT inputs for RestEM (paired with the subgoal text).
                    self._pending_imgs.append({
                        'step_idx': step_idx,
                        'subgoal': subgoal,
                        'image': image.copy(),
                        'wrist_image': wrist_image.copy(),
                    })
                if is_real_subgoal:
                    self._last_subgoal = subgoal

            def _patched_save_video(self, filename):
                _orig_recorder_save_video(self, filename)
                base = filename.rsplit('.', 1)[0]
                json_filename = base + '.json'

                # Flush raw images to a sibling directory `{base}_imgs/`. Each
                # subgoal-change moment gets a (front, wrist) PNG pair.
                imgs_dirname = base + '_imgs'
                imgs_dir = Path(str(self.save_dir)) / imgs_dirname
                imgs_dir.mkdir(parents=True, exist_ok=True)
                image_records = []
                for rec in self._pending_imgs:
                    si = rec['step_idx']
                    front_rel = f'{imgs_dirname}/step{si:04d}_front.png'
                    wrist_rel = f'{imgs_dirname}/step{si:04d}_wrist.png'
                    _imageio.imwrite(
                        str(Path(str(self.save_dir)) / front_rel),
                        rec['image'],
                    )
                    _imageio.imwrite(
                        str(Path(str(self.save_dir)) / wrist_rel),
                        rec['wrist_image'],
                    )
                    image_records.append({
                        'step_idx': si,
                        'subgoal': rec['subgoal'],
                        'front_path': front_rel,
                        'wrist_path': wrist_rel,
                    })

                payload = {
                    'task_goal': self._task_goal_text,
                    'video_filename': filename,
                    'n_steps': len(self._traj_steps),
                    'steps': self._traj_steps,
                    # image_records is the slim list RestEM actually consumes:
                    # one entry per subgoal-change moment with paths to the
                    # raw obs PNGs and the target subgoal text.
                    'image_records': image_records,
                }
                with open(os.path.join(str(self.save_dir), json_filename), 'w') as _f:
                    _json.dump(payload, _f)

            RolloutRecorder.__init__ = _patched_init
            RolloutRecorder.record = _patched_record
            RolloutRecorder.save_video = _patched_save_video
            # ---- end RolloutRecorder monkey-patches ----

            args = Args(
                host='127.0.0.1',
                port=PORT,
                use_oracle=(PLANNER == 'oracle'),
                use_qwenvl=(PLANNER == 'qwenvl'),
                subgoal_type='grounded_subgoal',
                qwenvl_groundSG_adapter_path=QWEN_ADAPTER_PATH,
                policy_name=f'collect_{PLANNER}_smoke',
                model_ckpt_id=79999,
                max_steps=MAX_STEPS,
                # Save into the cache volume so trajectories survive container
                # shutdown and can be read by tomorrow's RestEM step.
                save_dir=f'{COLLECT_DIR}/{PLANNER}',
                overwrite=False,  # don't wipe prior runs of the same task
            )

            save_dir = Path(args.save_dir) / args.policy_name / TASK_NAME
            save_dir.mkdir(parents=True, exist_ok=True)
            video_dir = save_dir / 'videos'
            video_dir.mkdir(parents=True, exist_ok=True)

            # EnvRunner is task-bound (BenchmarkEnvBuilder is initialized for
            # one env_id), so we build one per call. Per-episode env is built
            # fresh via .make_env(episode_id) inside the loop.
            t0 = time.time()
            env_runner = EnvRunner(
                env_id=TASK_NAME,
                video_save_dir=str(video_dir),
                max_steps=args.max_steps,
            )
            ctor_s = round(time.time() - t0, 1)
            task_n_episodes = env_runner.num_episodes
            out['task_n_episodes'] = task_n_episodes
            out['ctor_s'] = ctor_s

            # Build the planner ONCE (heavy for qwenvl, free for oracle).
            t0 = time.time()
            subgoal_predictor = build_subgoal_predictor(args, save_dir)
            out['planner_load_s'] = round(time.time() - t0, 1)

            evaluator = EpisodeEvaluator(args, save_dir)

            # Episode loop. We run min(N_EPISODES, task_n_episodes) so we
            # never exceed the benchmark's per-task budget.
            episodes_to_run = min(N_EPISODES, task_n_episodes)
            out['episodes_planned'] = episodes_to_run

            import gc as _gc

            for ep_id in range(episodes_to_run):
                ep_record = {'episode_id': ep_id}
                ep_t0 = time.time()
                # Heartbeat so a hang shows WHERE it is in the loop.
                print(
                    f'[collect] ep {ep_id+1}/{episodes_to_run} starting '
                    f'make_env...', flush=True,
                )
                try:
                    env_runner.make_env(ep_id)
                    print(
                        f'[collect] ep {ep_id+1}/{episodes_to_run} '
                        f'env built, starting eval_each_episode...',
                        flush=True,
                    )
                    success_flag = evaluator.eval_each_episode(
                        env_runner, subgoal_predictor, video_dir,
                    )
                    env_runner.close_env()
                    ep_record['success_flag'] = success_flag
                    ep_record['reward'] = int(success_flag == 'success')
                except Exception as e:
                    ep_record['error'] = repr(e)
                    ep_record['traceback'] = traceback.format_exc()[-1500:]
                    # Try to clean up env even on error.
                    try:
                        env_runner.close_env()
                    except Exception:
                        pass
                ep_record['episode_s'] = round(time.time() - ep_t0, 1)
                out['episodes'].append(ep_record)

                # Explicit between-episode cleanup. The most plausible cause
                # of the QwenVL multi-episode hang is one of these things
                # accumulating across episodes: KV cache fragments in the
                # planner, stale websocket clients from prior episodes,
                # video buffers in the subgoal predictor, sapien GPU resources.
                # Belt-and-suspenders cleanup; harmless if not strictly needed.
                try:
                    # Reset predictor's video buffer (cleared by get_subgoal
                    # mid-episode but no between-episode reset upstream).
                    if hasattr(subgoal_predictor, 'video_buffer'):
                        subgoal_predictor.video_buffer.clear()
                except Exception:
                    pass
                try:
                    _torch.cuda.empty_cache()
                except Exception:
                    pass
                _gc.collect()
                # Stream progress to stdout so we can watch episode-by-episode
                # on the Modal dashboard rather than waiting for the final
                # JSON sentinel.
                print(
                    f'[collect] ep {ep_id+1}/{episodes_to_run} '
                    f'flag={ep_record.get(\"success_flag\", \"ERR\")} '
                    f'reward={ep_record.get(\"reward\", -1)} '
                    f'in {ep_record[\"episode_s\"]}s',
                    flush=True,
                )

            # Summary
            rewards = [
                ep.get('reward', 0) for ep in out['episodes']
                if 'reward' in ep
            ]
            out['n_ok'] = len(rewards)
            out['n_errors'] = sum(
                1 for ep in out['episodes'] if 'error' in ep
            )
            out['success_rate'] = (
                round(sum(rewards) / len(rewards), 3) if rewards else None
            )
            out['ok'] = True
        except Exception as e:
            out['ok'] = False
            out['error'] = repr(e)
            out['traceback'] = traceback.format_exc()

        print('__RESULT_BEGIN__' + json.dumps(out, default=str) + '__RESULT_END__')
    """)
    client_script = header + body

    # Steer the planner process to GPU 1 (only relevant for qwenvl; oracle
    # doesn't use the GPU). The sim part runs on CPU regardless.
    client_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "1"}

    t0 = time.time()
    client_proc = subprocess.run(
        ["micromamba", "run", "-n", "robomme", "python", "-c", client_script],
        capture_output=True, text=True,
        # Per-episode budget × episodes + buffer; capped by Modal timeout above.
        timeout=10 * 3600,
        cwd="/app/examples/robomme",
        env=client_env,
    )
    total_s = round(time.time() - t0, 1)

    cs = client_proc.stdout
    a = cs.find("__RESULT_BEGIN__")
    b = cs.find("__RESULT_END__")
    if a >= 0 and b > a:
        try:
            client_report = json.loads(cs[a + len("__RESULT_BEGIN__"):b])
        except json.JSONDecodeError as e:
            client_report = {"parse_error": repr(e), "raw_tail": cs[-2500:]}
    else:
        client_report = {
            "no_sentinel": True,
            "stdout_tail": cs[-3000:],
            "stderr_tail": client_proc.stderr[-2500:],
        }

    _kill_server()

    # Flush volume writes (trajectory JSONs + videos) to durable storage so
    # they're visible to other functions immediately, not just on container
    # shutdown.
    CACHE_VOLUME.commit()

    # Quick count of how many trajectory JSONs landed on the volume for this
    # task — diagnostic so we can confirm dumps are working at scale.
    import glob
    traj_glob = (
        f"{CACHE_MOUNT}/runs/collect/{planner}/collect_{planner}_smoke/"
        f"{task_name}/videos/*.json"
    )
    traj_count = len(glob.glob(traj_glob))

    return {
        "ok": bool(client_report.get("ok")),
        "task": task_name,
        "planner": planner,
        "server_boot_s": boot_s,
        "wallclock_s": total_s,
        "trajectories_dumped": traj_count,
        "client": client_report,
        "server_log_tail": _read_log_tail(600),
    }


@app.function(
    gpu="A10G:2",
    timeout=4 * 3600,
    volumes={CACHE_MOUNT: CACHE_VOLUME},
)
def restem_sft_step(
    dataset_jsonl_path: str,
    adapter_in_path: str = "",
    adapter_out_dir: str = "",
    num_train_epochs: float = 2.0,
    learning_rate: float = 5e-5,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    nproc_per_node: int = 1,
) -> dict:
    """Run `swift sft` on the dataset → produces a new LoRA adapter.

    Mirrors upstream `scripts/finetune_vlm_subgoal_predictor.sh` flags
    (freeze_vit, sdpa attn, bfloat16, lora_rank 16, etc.) so the training
    config matches what produced the original checkpoint.

    If `adapter_in_path` is provided, swift loads it and continues training
    (RestEM-style). If empty, trains a fresh LoRA from scratch.

    Runs in the robomme micromamba env (where swift.llm + Qwen3-VL deps
    live). Uses both A10Gs via NPROC_PER_NODE=2.
    """
    import os
    import subprocess
    import time
    from pathlib import Path

    if not adapter_out_dir:
        ts = time.strftime("%Y%m%d_%H%M%S")
        adapter_out_dir = f"/cache/runs/restem_adapters/iter_{ts}"
    Path(adapter_out_dir).mkdir(parents=True, exist_ok=True)

    # Build swift sft command. Match upstream's
    # finetune_vlm_subgoal_predictor.sh except:
    #   * 2 GPUs not 4 (NPROC_PER_NODE=2)
    #   * Smaller batch + accum to be safe on small dataset (~100 examples)
    #   * Slightly lower LR (5e-5 vs 1e-4) to avoid overfitting tiny data
    #   * Drop --deepspeed (overkill for 2 GPUs + small dataset)
    #   * Add --adapters if continuing from existing LoRA
    swift_cmd = [
        "swift", "sft",
        "--model", "Qwen/Qwen3-VL-4B-Instruct",
        "--dataset", dataset_jsonl_path,
        "--split_dataset_ratio", "0.0",
        "--load_from_cache_file", "true",
        "--packing", "false",
        "--train_type", "lora",
        "--torch_dtype", "bfloat16",
        "--num_train_epochs", str(num_train_epochs),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--attn_impl", "sdpa",
        "--padding_free", "false",
        "--learning_rate", str(learning_rate),
        "--lora_rank", str(lora_rank),
        "--lora_alpha", str(lora_alpha),
        "--target_modules", "all-linear",
        "--freeze_vit", "true",
        "--freeze_aligner", "true",
        "--gradient_checkpointing", "true",
        "--vit_gradient_checkpointing", "false",
        "--save_steps", "50",
        "--save_total_limit", "2",
        "--logging_steps", "10",
        "--max_length", "3200",
        "--output_dir", adapter_out_dir,
        "--warmup_ratio", "0.05",
        "--dataset_num_proc", "4",
        "--dataloader_num_workers", "4",
    ]
    if adapter_in_path:
        swift_cmd += ["--adapters", adapter_in_path]

    # GPU visibility — single GPU runs swift sft directly (no elastic launcher,
    # so any Python error surfaces in stdout/stderr). Multi-GPU wraps in
    # torch.distributed.run which swallows tracebacks (error_file: <N/A>).
    # Start with 1 GPU to debug; bump to 2 once we know it works.
    visible = ",".join(str(i) for i in range(nproc_per_node))
    env = {
        **os.environ,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "IMAGE_MAX_TOKEN_NUM": "256",
        "VIDEO_MAX_TOKEN_NUM": "64",
        "FPS_MAX_FRAMES": "10",
        "NPROC_PER_NODE": str(nproc_per_node),
        "CUDA_VISIBLE_DEVICES": visible,
        # swift.llm defaults to ModelScope (Chinese hub). Force HuggingFace
        # so it uses the model we already cached at /cache/hf from earlier
        # test_load_planner runs. Without this, swift tries to download from
        # modelscope.cn which is unreachable from Modal and fails fast.
        "USE_HF": "1",
        # Make torch elastic errors more verbose if multi-GPU does fail later.
        "TORCHELASTIC_ERROR_FILE": "/tmp/torchelastic_error.json",
    }

    # Shell out to micromamba (swift.llm lives in the robomme env). We launch
    # the swift CLI directly — micromamba run inherits our env vars including
    # CUDA_VISIBLE_DEVICES and the SDPA-related flags.
    full_cmd = ["micromamba", "run", "-n", "robomme"] + swift_cmd

    t0 = time.time()
    result = subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=4 * 3600,
    )
    elapsed_s = round(time.time() - t0, 1)

    CACHE_VOLUME.commit()

    # Find the most recently saved checkpoint in adapter_out_dir/ — swift's
    # output structure is adapter_out_dir/vXX-..._{date}/checkpoint-NN/.
    saved = []
    try:
        for p in Path(adapter_out_dir).rglob("checkpoint-*"):
            if p.is_dir():
                saved.append((p.stat().st_mtime, str(p)))
        saved.sort(reverse=True)
    except Exception:
        pass

    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "elapsed_s": elapsed_s,
        "adapter_out_dir": adapter_out_dir,
        "saved_checkpoints": [s[1] for s in saved[:5]],
        "latest_checkpoint": saved[0][1] if saved else None,
        "stdout_tail": result.stdout[-3000:],
        "stderr_tail": result.stderr[-3000:],
    }


@app.function(volumes={CACHE_MOUNT: CACHE_VOLUME}, timeout=600)
def build_restem_dataset(
    planner: str = "qwenvl",
    output_jsonl_path: str = "",
    only_success: bool = True,
    since_hours: float = 0.0,
) -> dict:
    """Walk successful trajectories on the volume → build SFT JSONL for swift.

    For each successful trajectory's image_records (one per subgoal-change
    moment), emit one swift-format training example:

        {"messages": [
            {"role": "system", "content": <upstream grounded-subgoal system prompt>},
            {"role": "user",   "content": "The task goal is: ...\\n
                                           The history of previous predicted grounded
                                           language subgoals are: [s1; s2]\\n
                                           <image>What's the next grounded language
                                           subgoal based on current observation?"},
            {"role": "assistant", "content": <the subgoal that led to success>}],
         "images": [<absolute path to front image>]}

    The prompts mirror QwenVLSubgoalPredictor's inference-time format
    (see examples/robomme/subgoal_prediction/qwenvl/api.py), so the model
    sees the same template at train and inference.

    Output JSONL goes to `output_jsonl_path` (default: timestamped path on
    the volume). Returns dataset stats.
    """
    import glob
    import json
    import time
    from collections import defaultdict
    from pathlib import Path

    # Upstream constants — keep in sync with subgoal_prediction/qwenvl/api.py
    SYSTEM_PROMPT = (
        "You are a helpful assistant to help guide the robot to complete "
        "the task by predicting a sequence of grounded language subgoals"
    )

    base = Path(f"/cache/runs/collect/{planner}")
    if not base.exists():
        return {"ok": False, "error": f"no {base} on volume"}

    cutoff_ts = (time.time() - since_hours * 3600) if since_hours > 0 else 0.0

    # Default output path: timestamped under /cache/runs/restem_data/
    if not output_jsonl_path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        Path("/cache/runs/restem_data").mkdir(parents=True, exist_ok=True)
        output_jsonl_path = f"/cache/runs/restem_data/{planner}_{ts}.jsonl"

    n_trajs_scanned = 0
    n_trajs_used = 0
    n_examples_written = 0
    per_task = defaultdict(int)
    missing_images = 0

    with open(output_jsonl_path, "w") as out_f:
        for json_path in base.rglob("*.json"):
            if "_imgs" in str(json_path):
                continue
            parts = json_path.stem.split("_")
            if len(parts) < 5 or not parts[1].startswith("ep"):
                continue
            env, flag = parts[0], parts[2]
            if flag not in ("success", "fail", "timeout", "unknown"):
                continue
            if only_success and flag != "success":
                continue

            try:
                mtime = json_path.stat().st_mtime
            except OSError:
                continue
            if cutoff_ts and mtime < cutoff_ts:
                continue

            n_trajs_scanned += 1
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except Exception:
                continue

            task_goal = data.get("task_goal", "")
            image_records = data.get("image_records", []) or []
            if not image_records:
                continue  # no usable training data

            # Sort by step_idx so "previous subgoals" history is correct.
            image_records = sorted(image_records, key=lambda r: r.get("step_idx", 0))

            traj_dir = json_path.parent  # .../videos/

            # Build training example per image_record. The Nth example's
            # history is the previous (N-1) subgoals; the assistant target
            # is image_records[N]['subgoal'].
            for i, rec in enumerate(image_records):
                front_path = traj_dir / rec.get("front_path", "")
                if not front_path.exists():
                    missing_images += 1
                    continue
                subgoal = rec.get("subgoal", "")
                if not subgoal or subgoal == "[initializing...]":
                    continue

                prev_subgoals = [
                    r.get("subgoal", "")
                    for r in image_records[:i]
                    if r.get("subgoal") and r.get("subgoal") != "[initializing...]"
                ]
                if prev_subgoals:
                    history_str = "; ".join(prev_subgoals)
                    user_content = (
                        f"The task goal is: {task_goal}\n"
                        f"The history of previous predicted grounded language "
                        f"subgoals are: [{history_str}]\n"
                        "<image>What's the next grounded language subgoal "
                        "based on current observation?"
                    )
                else:
                    user_content = (
                        f"The task goal is: {task_goal}\n"
                        "<image>What's the next grounded language subgoal "
                        "based on current observation?"
                    )

                example = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": subgoal},
                    ],
                    "images": [str(front_path)],
                }
                out_f.write(json.dumps(example) + "\n")
                n_examples_written += 1
                per_task[env] += 1
            n_trajs_used += 1

    CACHE_VOLUME.commit()
    return {
        "ok": True,
        "output_jsonl_path": output_jsonl_path,
        "n_trajs_scanned": n_trajs_scanned,
        "n_trajs_used": n_trajs_used,
        "n_examples_written": n_examples_written,
        "missing_images": missing_images,
        "examples_per_task": dict(per_task),
    }


@app.function(volumes={CACHE_MOUNT: CACHE_VOLUME}, timeout=300)
def aggregate_baselines(
    planner: str = "",
    since_hours: float = 0.0,
    max_per_task: int = 0,
) -> dict:
    """Walk /cache/runs/collect/* and tally per-(planner, task) success rates.

    Filename pattern (set by upstream eval.py's RolloutRecorder.save_video):
        {env}_ep{N}_{success_flag}_{task_goal_with_spaces}_{difficulty}.json
    where success_flag ∈ {success, fail, timeout, unknown}.

    Args:
        planner: filter to one planner subdir (empty = all).
        since_hours: if > 0, ignore JSON files older than this many hours.
            Use to exclude trajectories from earlier debug runs that share
            the same directory layout.
        max_per_task: if > 0, keep only the N most recently modified JSON
            files per (planner, task). Useful when you want exactly n=50
            episodes per task and the dir has accumulated extras.
    """
    import time
    from collections import defaultdict
    from pathlib import Path

    base = Path("/cache/runs/collect")
    if not base.exists():
        return {"ok": False, "error": f"no {base} on volume"}

    if planner:
        planner_dirs = [base / planner] if (base / planner).is_dir() else []
    else:
        planner_dirs = sorted([p for p in base.iterdir() if p.is_dir()])

    cutoff_ts = (time.time() - since_hours * 3600) if since_hours > 0 else 0.0

    # First pass: gather (json_path, mtime) per (planner, task), apply since_hours.
    # entries[planner][task] = list of (mtime, path)
    entries: dict = defaultdict(lambda: defaultdict(list))
    for p_dir in planner_dirs:
        for json_path in p_dir.rglob("*.json"):
            if "_imgs/" in str(json_path) or json_path.parent.name.endswith("_imgs"):
                continue
            parts = json_path.stem.split("_")
            if len(parts) < 5:
                continue
            env, ep_part, flag = parts[0], parts[1], parts[2]
            if not ep_part.startswith("ep"):
                continue
            if flag not in ("success", "fail", "timeout", "unknown"):
                continue
            try:
                mtime = json_path.stat().st_mtime
            except OSError:
                continue
            if cutoff_ts and mtime < cutoff_ts:
                continue
            entries[p_dir.name][env].append((mtime, flag, str(json_path)))

    # Second pass: optionally cap to max_per_task most-recent files.
    # Count flags per (planner, task).
    results: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for p_name, tasks in entries.items():
        for task, items in tasks.items():
            items.sort(key=lambda x: x[0], reverse=True)  # newest first
            if max_per_task > 0:
                items = items[:max_per_task]
            for _mtime, flag, _path in items:
                results[p_name][task][flag] += 1

    # Compute success rates.
    output: dict = {}
    for p_name, tasks in results.items():
        output[p_name] = {}
        for task, counts in tasks.items():
            success = counts.get("success", 0)
            fail = counts.get("fail", 0)
            timeout = counts.get("timeout", 0)
            unknown = counts.get("unknown", 0)
            total = success + fail + timeout + unknown
            output[p_name][task] = {
                "success": success,
                "fail": fail,
                "timeout": timeout,
                "unknown": unknown,
                "total": total,
                "success_rate": round(success / total, 3) if total else None,
            }
    return {
        "ok": True,
        "results": output,
        "filter": {
            "planner": planner or "<all>",
            "since_hours": since_hours,
            "max_per_task": max_per_task,
        },
    }


@app.function(volumes={CACHE_MOUNT: CACHE_VOLUME}, timeout=120)
def inspect_trajectories(planner: str = "oracle", task: str = "MoveCube") -> dict:
    """Read recent trajectory JSONs off the volume and report their structure.

    Run this after a collect_rollouts pass to verify the dumped data has
    everything tomorrow's RestEM-iter-0 step needs. We want to confirm:
      - JSON files exist on the volume (not just /tmp)
      - Each JSON has per-step records with `state`, `action`, `subgoal`
      - Subgoals are non-null on the steps that triggered planner calls
      - File naming encodes success_flag so we can filter

    No GPU needed — pure file read.
    """
    import glob
    import json
    from pathlib import Path

    pattern = (
        f"/cache/runs/collect/{planner}/collect_{planner}_smoke/"
        f"{task}/videos/*.json"
    )
    files = sorted(glob.glob(pattern))

    if not files:
        # Try a more permissive glob in case the path differs.
        broader = sorted(glob.glob(
            f"/cache/runs/collect/{planner}/**/*.json", recursive=True,
        ))
        return {
            "ok": False,
            "queried_pattern": pattern,
            "found_under_planner_dir": broader[:10],
            "msg": "no JSONs at expected path",
        }

    with open(files[0]) as f:
        data = json.load(f)

    steps = data.get("steps", [])
    n_with_subgoal = sum(1 for s in steps if s.get("subgoal"))
    unique_subgoals = sorted({s["subgoal"] for s in steps if s.get("subgoal")})

    # Count subgoal *changes* (each is a planner call, which is what RestEM
    # trains on — not every step, just every state where a new subgoal was
    # generated).
    n_subgoal_changes = 0
    prev = None
    for s in steps:
        sg = s.get("subgoal")
        if sg and sg != prev:
            n_subgoal_changes += 1
        prev = sg

    # Parse success_flag from filename:
    # `{env}_ep{N}_{success_flag}_{task_goal}_{difficulty}.mp4` / .json
    fn = Path(files[0]).name
    parts = fn.split("_")
    inferred_success_flag = parts[2] if len(parts) >= 3 else "?"

    image_records = data.get("image_records", [])
    # Verify the image files referenced by image_records actually exist on disk.
    first_file_dir = Path(files[0]).parent
    images_present = sum(
        1 for ir in image_records
        if (first_file_dir / ir.get("front_path", "")).exists()
        and (first_file_dir / ir.get("wrist_path", "")).exists()
    )

    return {
        "ok": True,
        "n_files": len(files),
        "all_filenames": [Path(f).name for f in files],
        "first_file": fn,
        "first_file_keys": sorted(data.keys()),
        "task_goal": data.get("task_goal", "")[:200],
        "n_steps": len(steps),
        "n_steps_with_subgoal": n_with_subgoal,
        "n_unique_subgoals": len(unique_subgoals),
        "n_subgoal_changes": n_subgoal_changes,
        "n_image_records": len(image_records),
        "n_image_records_present_on_disk": images_present,
        "sample_image_record": image_records[0] if image_records else None,
        "inferred_success_flag_from_filename": inferred_success_flag,
        "sample_subgoals_first_3": unique_subgoals[:3],
        "first_step_record": steps[0] if steps else None,
        "first_step_with_subgoal": (
            next((s for s in steps if s.get("subgoal")), None)
        ),
    }


@app.function(gpu="A10G:2", timeout=1800, volumes={CACHE_MOUNT: CACHE_VOLUME})
def test_qwenvl_episode(
    task_name: str = "MoveCube",
    max_steps: int = 200,
) -> dict:
    """Run ONE Qwen3-VL-planner episode end-to-end → the SFT baseline.

    This is structurally identical to test_oracle_episode but swaps in the
    Qwen3-VL-4B + grounded-LoRA planner. It's the policy the proposal aims to
    RL-finetune — getting it through one closed-loop episode is the gate to
    starting the RestEM/GRPO work.

    VRAM math (17.2 GB VLA + 8.6 GB planner = 25.8 GB > 23 GB on one A10G)
    forces us to use 2 GPUs:
      * GPU 0 (CUDA_VISIBLE_DEVICES=0): VLA server in uv env.
      * GPU 1 (CUDA_VISIBLE_DEVICES=1): Qwen3-VL planner + sim in robomme env.
    Mirrors upstream scripts/eval.sh's GPU_ID_server / GPU_ID_client split.
    """
    import json
    import os
    import socket
    import subprocess
    import textwrap
    import time

    PORT = 8011
    VLA_CKPT_DIR = f"{CKPTS_DIR}/pi05_baseline/pi05_baseline/79999"
    QWEN_ADAPTER_PATH = (
        f"{CKPTS_DIR}/vlm_subgoal_predictor/qwenvl/"
        "grounded_subgoal/checkpoint-1200"
    )
    SERVER_LOG = "/tmp/serve_policy.log"

    # ----- 1. Boot serve_policy.py on GPU 0 (uv env) -----
    server_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    server_log_fp = open(SERVER_LOG, "wb")
    server = subprocess.Popen(
        [
            "python", "/app/scripts/serve_policy.py",
            f"--port={PORT}",
            "policy:checkpoint",
            f"--policy.dir={VLA_CKPT_DIR}",
            "--policy.config=pi05_baseline",
        ],
        stdout=server_log_fp,
        stderr=subprocess.STDOUT,
        cwd="/app",
        env=server_env,
    )

    def _read_log_tail(n: int = 2000) -> str:
        try:
            if not server_log_fp.closed:
                server_log_fp.flush()
        except (ValueError, OSError):
            pass
        try:
            with open(SERVER_LOG, "r") as f:
                return f.read()[-n:]
        except Exception as e:  # noqa: BLE001
            return f"<could not read log: {e!r}>"

    def _kill_server() -> None:
        try:
            server.terminate()
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        server_log_fp.close()

    # Wait for the VLA port to open.
    boot_t0 = time.time()
    ready = False
    while time.time() - boot_t0 < 600:
        if server.poll() is not None:
            break
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=1):
                ready = True
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(2)
    boot_s = round(time.time() - boot_t0, 1)

    if not ready:
        rc = server.returncode
        _kill_server()
        return {
            "ok": False,
            "phase": "server_boot",
            "elapsed_s": boot_s,
            "server_returncode": rc,
            "server_log_tail": _read_log_tail(),
        }

    # ----- 2. Run the Qwen3-VL episode on GPU 1 (robomme micromamba env) -----
    # Same sapien monkey-patches as test_oracle_episode; same eval.Args path;
    # only differences: use_qwenvl=True instead of use_oracle=True, and we
    # override qwenvl_groundSG_adapter_path to absolute /cache/... path.
    header = (
        f"PORT = {PORT}\n"
        f"MAX_STEPS = {max_steps}\n"
        f"TASK_NAME = {task_name!r}\n"
        f"QWEN_ADAPTER_PATH = {QWEN_ADAPTER_PATH!r}\n"
    )
    body = textwrap.dedent("""
        import json, os, sys, time, traceback
        from pathlib import Path

        # Sapien monkey-patches (same as Oracle path). Must precede any
        # upstream module import that pulls in sapien.
        import sapien
        _orig_rs = sapien.render.RenderSystem
        def _patched_rs(_device_ignored, *args, **kwargs):
            return _orig_rs('cpu', *args, **kwargs)
        sapien.render.RenderSystem = _patched_rs

        import numpy as _np
        import torch as _torch
        _cam_cls = sapien.render.RenderCameraComponent
        class _CudaShim:
            __slots__ = ('_arr',)
            def __init__(self, arr):
                self._arr = arr
            def torch(self):
                return _torch.as_tensor(_np.asarray(self._arr))
        def _patched_get_picture_cuda(self, name):
            return _CudaShim(self.get_picture(name))
        _cam_cls.get_picture_cuda = _patched_get_picture_cuda

        # Qwen3-VL env vars (eval.py sets these at module level; we set them
        # explicitly in case anything imports earlier).
        os.environ['IMAGE_MAX_TOKEN_NUM'] = '256'
        os.environ['VIDEO_MAX_TOKEN_NUM'] = '64'
        os.environ['FPS_MAX_FRAMES'] = '10'

        sys.path.insert(0, '/app/examples/robomme')

        out = {}
        try:
            # Sanity check: this process should see exactly one GPU (the one
            # we steered to via CUDA_VISIBLE_DEVICES=1 in the parent).
            visible = os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')
            torch_dev_count = _torch.cuda.device_count()
            torch_dev_name = (
                _torch.cuda.get_device_name(0) if torch_dev_count else 'none'
            )

            from env_runner import EnvRunner
            from subgoal_predictor import build_subgoal_predictor
            from eval import EpisodeEvaluator, Args

            args = Args(
                host='127.0.0.1',
                port=PORT,
                use_qwenvl=True,
                subgoal_type='grounded_subgoal',
                # Absolute path on the cache volume, instead of upstream's
                # `runs/ckpts/...` relative path that assumes cwd.
                qwenvl_groundSG_adapter_path=QWEN_ADAPTER_PATH,
                policy_name='qwenvl_smoke',
                model_ckpt_id=79999,
                max_steps=MAX_STEPS,
                save_dir='/tmp/qwenvl_eval',
                overwrite=True,
            )

            save_dir = Path(args.save_dir) / 'qwenvl_smoke'
            save_dir.mkdir(parents=True, exist_ok=True)
            video_dir = save_dir / 'videos'
            video_dir.mkdir(parents=True, exist_ok=True)

            t0 = time.time()
            env_runner = EnvRunner(
                env_id=TASK_NAME,
                video_save_dir=str(video_dir),
                max_steps=args.max_steps,
            )
            env_runner.make_env(0)
            setup_s = round(time.time() - t0, 1)

            # Building the subgoal predictor triggers the Qwen3-VL load
            # (PtEngine constructor inside QwenVLSubgoalPredictor.setup_api).
            t0 = time.time()
            subgoal_predictor = build_subgoal_predictor(args, save_dir)
            planner_load_s = round(time.time() - t0, 1)

            evaluator = EpisodeEvaluator(args, save_dir)

            t0 = time.time()
            success_flag = evaluator.eval_each_episode(
                env_runner, subgoal_predictor, video_dir,
            )
            episode_s = round(time.time() - t0, 1)
            env_runner.close_env()

            # VRAM after the episode (max in-use).
            vram_peak_mb = (
                _torch.cuda.max_memory_allocated() // 1024**2
                if torch_dev_count else -1
            )

            out = {
                'ok': True,
                'task': TASK_NAME,
                'episode_id': 0,
                'success_flag': success_flag,
                'reward': int(success_flag == 'success'),
                'setup_s': setup_s,
                'planner_load_s': planner_load_s,
                'episode_s': episode_s,
                'cuda_visible': visible,
                'torch_device_count': torch_dev_count,
                'torch_device_name': torch_dev_name,
                'planner_vram_peak_mb': vram_peak_mb,
            }
        except Exception as e:
            out = {
                'ok': False,
                'error': repr(e),
                'traceback': traceback.format_exc(),
            }

        print('__RESULT_BEGIN__' + json.dumps(out, default=str) + '__RESULT_END__')
    """)
    client_script = header + body

    # Client (planner + sim) runs on GPU 1.
    client_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "1"}

    t0 = time.time()
    client_proc = subprocess.run(
        ["micromamba", "run", "-n", "robomme", "python", "-c", client_script],
        capture_output=True, text=True, timeout=60 * 25,
        cwd="/app/examples/robomme",
        env=client_env,
    )
    episode_total_s = round(time.time() - t0, 1)

    cs = client_proc.stdout
    a = cs.find("__RESULT_BEGIN__")
    b = cs.find("__RESULT_END__")
    if a >= 0 and b > a:
        try:
            client_report = json.loads(cs[a + len("__RESULT_BEGIN__"):b])
        except json.JSONDecodeError as e:
            client_report = {"parse_error": repr(e), "raw_tail": cs[-2500:]}
    else:
        client_report = {
            "no_sentinel": True,
            "stdout_tail": cs[-2500:],
            "stderr_tail": client_proc.stderr[-2500:],
        }

    _kill_server()

    return {
        "ok": bool(client_report.get("ok")),
        "server_boot_s": boot_s,
        "episode_total_s": episode_total_s,
        "client": client_report,
        "server_log_tail": _read_log_tail(800),
    }


@app.function(gpu="A10G", timeout=900, volumes={CACHE_MOUNT: CACHE_VOLUME})
def test_sim(
    task_name: str = "MoveCube",
    n_steps: int = 50,
    render_device: str = "cpu",
) -> dict:
    """Boot one RoboMME episode in the sim and step it forward — no models.

    Retires the biggest unknown left in our infra: does ManiSkill / Sapien
    render headless inside our Modal container? Upstream's Dockerfile sets
    SAPIEN_RENDER_DEVICE=cuda + installs libegl1/libvulkan1, but we've never
    actually opened a scene.

    The sim + EnvRunner live in the `robomme` micromamba env (ManiSkill +
    Sapien + robomme_benchmark are installed there), so we shell out the
    same way we do for the planner.

    No success expected — we drive the robot with a noisy neutral joint
    pose for n_steps, mostly to confirm step() doesn't crash. Returns the
    final success_flag (almost certainly "timeout" or "fail"), step latency,
    and observation shapes.
    """
    import json
    import subprocess
    import textwrap

    # Compact script that mirrors examples/robomme/simple_test.py — the
    # canonical "does the sim work" check from upstream.
    script = textwrap.dedent(f"""
        import json, os, time, traceback
        import numpy as np

        TASK = {task_name!r}
        N_STEPS = {n_steps}

        # IMPORTANT: override Sapien/MuJoCo render device BEFORE any of their
        # modules import — they cache the choice at import time. The image
        # default is SAPIEN_RENDER_DEVICE=cuda (Vulkan-backed) which fails on
        # this Modal container. 'cpu' uses Sapien's software rasterizer.
        os.environ['SAPIEN_RENDER_DEVICE'] = {render_device!r}
        if {render_device!r} == 'cpu':
            os.environ['MUJOCO_GL'] = 'osmesa'

        # examples/robomme/ uses sibling imports (`from env_runner import ...`,
        # `from utils import ...`), so we need to run with that as cwd / on
        # sys.path. Modal subprocess.run sets cwd, but we add to sys.path too
        # in case anything else looks there.
        import sys
        sys.path.insert(0, '/app/examples/robomme')

        # ManiSkill's BaseEnv hardcodes render_device='cuda:0' (wrapped in a
        # sapien.Device object), which Sapien rejects when lavapipe is the
        # only available Vulkan device. Monkey-patch RenderSystem to ALWAYS
        # use whichever device spec actually constructs a working RenderSystem,
        # discovered by trial below.
        sapien_probe = {{}}
        try:
            import sapien

            # 1. Probe API surface so we can see what's actually available.
            sapien_probe['render_dir'] = sorted(
                a for a in dir(sapien.render)
                if not a.startswith('_')
                and ('device' in a.lower() or 'gpu' in a.lower() or 'render' in a.lower())
            )[:30]

            # 2. Grab the summary (string) for human reading.
            try:
                summary = sapien.render.get_device_summary()
                if isinstance(summary, str):
                    sapien_probe['summary'] = summary[:1200]
                else:
                    sapien_probe['summary_type'] = type(summary).__name__
            except Exception as _e:
                sapien_probe['summary_err'] = repr(_e)

            # 3. Build a list of candidate device specs to try.
            _candidates = []
            # Sapien-native string aliases that have historically worked:
            for s in ('cpu', 'llvmpipe', 'GPU: llvmpipe', '0'):
                _candidates.append(('str', s, lambda s=s: s))
            # And sapien.Device(...) constructed from those:
            if hasattr(sapien, 'Device'):
                for s in ('cpu', 'llvmpipe', 'cuda:0'):
                    _candidates.append(
                        ('Device', s,
                         lambda s=s: sapien.Device(s)),
                    )

            _orig_rs = sapien.render.RenderSystem
            _target = None
            _target_label = None
            _attempts = []
            for kind, label, ctor in _candidates:
                try:
                    spec = ctor()
                    _rs = _orig_rs(spec)
                    _target = spec
                    _target_label = f'{{kind}}({{label!r}})'
                    _attempts.append((label, kind, 'OK'))
                    del _rs  # drop test instance
                    break
                except Exception as e:
                    _attempts.append((label, kind, repr(e)[:160]))
            sapien_probe['attempts'] = _attempts
            sapien_probe['target_chosen'] = _target_label

            def _patched_rs(_device_ignored, *args, **kwargs):
                use = _target if _target is not None else 'cpu'
                return _orig_rs(use, *args, **kwargs)
            sapien.render.RenderSystem = _patched_rs
            sapien_probe['patch_installed'] = _target is not None

            # 5. Patch camera CUDA interop. ManiSkill calls
            # `camera.get_picture_cuda(name).torch()[None, ...]` for fast
            # GPU-side texture transfer. With lavapipe (CPU Vulkan, cudaId=-1)
            # there's no CUDA peer to interop with, so we route through the
            # plain `get_picture` (numpy) path and wrap the result so it
            # still has the `.torch()` method ManiSkill expects.
            try:
                import numpy as _np
                import torch as _torch

                _cam_cls = sapien.render.RenderCameraComponent
                _orig_get_picture_cuda = getattr(_cam_cls, 'get_picture_cuda', None)
                _has_get_picture = hasattr(_cam_cls, 'get_picture')

                sapien_probe['camera_has_get_picture'] = _has_get_picture
                sapien_probe['camera_methods'] = sorted(
                    m for m in dir(_cam_cls)
                    if 'picture' in m.lower() or 'capture' in m.lower()
                )

                # _CudaShim pretends to be the result of `get_picture_cuda`;
                # the only method ManiSkill calls on it is `.torch()`.
                class _CudaShim:
                    __slots__ = ('_arr',)
                    def __init__(self, arr):
                        self._arr = arr
                    def torch(self):
                        # ManiSkill does `... .torch()[None, ...]` then often
                        # `.cuda()` later; we return a CPU tensor here and
                        # let downstream code move it if needed.
                        return _torch.as_tensor(_np.asarray(self._arr))

                def _patched_get_picture_cuda(self, name):
                    arr = self.get_picture(name)
                    return _CudaShim(arr)

                if _has_get_picture:
                    _cam_cls.get_picture_cuda = _patched_get_picture_cuda
                    sapien_probe['camera_patch_installed'] = True
                else:
                    sapien_probe['camera_patch_installed'] = False
                    sapien_probe['camera_patch_skip_reason'] = (
                        'no get_picture method to fall back to'
                    )
            except Exception as _e:
                sapien_probe['camera_patch_error'] = repr(_e)
        except Exception as e:
            sapien_probe['setup_error'] = repr(e)
            import traceback as _tb
            sapien_probe['setup_traceback'] = _tb.format_exc()

        out = {{}}
        try:
            from env_runner import EnvRunner
            BASE_ACTION = np.array(
                [0.0, 0.0, 0.0, -np.pi/2, 0.0, np.pi/2, np.pi/4, 1.0],
                dtype=np.float32,
            )

            # No video for the smoke test — but EnvRunner still wants a dir.
            video_dir = '/tmp/sim_smoke_videos'
            os.makedirs(video_dir, exist_ok=True)

            t0 = time.time()
            runner = EnvRunner(env_id=TASK, video_save_dir=video_dir, max_steps=N_STEPS)
            n_eps = runner.num_episodes
            ctor_s = round(time.time() - t0, 2)

            t0 = time.time()
            runner.make_env(0)
            make_env_s = round(time.time() - t0, 2)

            t0 = time.time()
            init = runner.get_init_obs()
            reset_s = round(time.time() - t0, 2)

            # Sanity-check observation shapes / dtypes so we know what the
            # rollout loop will be carrying around later.
            obs_shapes = {{
                'n_init_images': len(init['images']),
                'image_shape': tuple(init['images'][0].shape),
                'image_dtype': str(init['images'][0].dtype),
                'wrist_image_shape': tuple(init['wrist_images'][0].shape),
                'state_shape': tuple(init['states'][0].shape),
                'task_goal': init['task_goal'][:200],
            }}

            t0 = time.time()
            stop_flag = False
            success_flag = 'unknown'
            steps_done = 0
            for _ in range(N_STEPS):
                noise = np.random.normal(0, 0.01, BASE_ACTION.shape).astype(np.float32)
                noise[-1] = 0.0  # don't perturb gripper command
                action = BASE_ACTION + noise
                _obs, stop_flag, success_flag = runner.step(action)
                steps_done += 1
                if stop_flag:
                    break
            step_total_s = round(time.time() - t0, 2)
            runner.close_env()

            out = {{
                'ok': True,
                'task': TASK,
                'num_episodes_for_task': n_eps,
                'ctor_s': ctor_s,
                'make_env_s': make_env_s,
                'reset_s': reset_s,
                'steps_done': steps_done,
                'step_total_s': step_total_s,
                'step_avg_ms': round(step_total_s * 1000 / max(steps_done, 1), 1),
                'final_success_flag': success_flag,
                'stop_flag': stop_flag,
                'obs': obs_shapes,
            }}
        except Exception as e:
            out = {{'ok': False, 'error': repr(e), 'traceback': traceback.format_exc()}}
        # Always include sapien_probe so we can see what devices were available
        # and which one we picked, whether or not the sim test passed.
        out['sapien_probe'] = sapien_probe
        print('__RESULT_BEGIN__' + json.dumps(out, default=str) + '__RESULT_END__')
    """)

    result = subprocess.run(
        ["micromamba", "run", "-n", "robomme", "python", "-c", script],
        capture_output=True, text=True, timeout=60 * 12,
        cwd="/app/examples/robomme",
    )

    report: dict = {"returncode": result.returncode}
    stdout = result.stdout
    start = stdout.find("__RESULT_BEGIN__")
    end = stdout.find("__RESULT_END__")
    if start != -1 and end != -1:
        payload = stdout[start + len("__RESULT_BEGIN__"):end]
        try:
            report["result"] = json.loads(payload)
        except json.JSONDecodeError as e:
            report["parse_error"] = repr(e)
            report["payload_sample"] = payload[:500]
    else:
        report["stdout_tail"] = stdout[-2500:]
        report["stderr_tail"] = result.stderr[-2500:]
    return report


@app.local_entrypoint()
def main() -> None:
    """Local entrypoint: runs the smoke test and prints results."""
    result = smoke_test.remote()
    print("=== robomme-rl smoke test ===")
    for k, v in result.items():
        print(f"{k}: {v}")


@app.local_entrypoint()
def vk_sapien() -> None:
    """Local entrypoint: probe Sapien's actual Vulkan loader.

    Tells us whether our system-level Vulkan config is even reaching Sapien,
    or whether it's loading its own libvulkan from inside the micromamba env.
    """
    result = sapien_vk_probe.remote()
    print("=== sapien Vulkan loader probe ===")
    for k, v in result.items():
        print(f"\n--- {k} ---\n{v}")


@app.local_entrypoint()
def vk_find_icd() -> None:
    """Local entrypoint: scan for the actual Vulkan ICD library.

    Use after vk_debug shows the ICD JSON exists but vulkaninfo still fails.
    Tells us which library to point the ICD JSON at.
    """
    result = find_vulkan_icd.remote()
    print("=== robomme-rl Vulkan ICD scan ===")
    for k, v in result.items():
        print(f"\n--- {k} ---\n{v}")


@app.local_entrypoint()
def vk_debug() -> None:
    """Local entrypoint: dump Vulkan/driver state from inside an A10G container.

    Usage:
        modal run modal/app.py::vk_debug

    Use this when sim_check fails with `vk::PhysicalDevice::createDeviceUnique:
    ErrorInitializationFailed` — the output tells us which piece is missing
    (ICD manifest, driver lib, capability flag, etc.).
    """
    result = vulkan_debug.remote()
    print("=== robomme-rl vulkan diagnostics ===")
    for k, v in result.items():
        print(f"\n--- {k} ---")
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"  {kk}: {vv}")
        else:
            print(v)


# RoboMME's 16-task slate, grouped by the four cognitive memory categories
# (counting/temporal, permanence/spatial, reference/object, imitation/procedural).
# Mirrors examples/robomme/utils.py:TASK_NAME_LIST upstream.
ROBOMME_TASKS = [
    # counting / temporal
    "BinFill", "StopCube", "PickXtimes", "SwingXtimes",
    # permanence / spatial
    "ButtonUnmask", "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap",
    # reference / object
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    # imitation / procedural
    "MoveCube", "InsertPeg", "PatternLock",
]


@app.local_entrypoint()
def baselines(
    planner: str = "oracle",
    n_episodes: int = 30,
    max_steps: int = 400,
    tasks: str = "",
) -> None:
    """Fan out `collect_rollouts` across many tasks in parallel.

    Each task gets its own 2× A10G container; they run concurrently. This is
    the layer that turns "5-day baselines" into "few-hour baselines" — single
    biggest leverage point in the project's compute story.

    Usage:
        # Smoke test (2 tasks × 1 ep oracle, ~10-15 min wallclock):
        modal run modal/app.py::baselines \\
            --tasks MoveCube,BinFill --n-episodes 1 --planner oracle

        # Run 1 baseline (full 16 tasks × 30 ep oracle, ~2-3 hr wallclock):
        modal run modal/app.py::baselines --planner oracle

        # Run 1 baseline (full 16 tasks × 30 ep qwenvl, ~4-5 hr wallclock):
        modal run modal/app.py::baselines --planner qwenvl
    """
    task_list = (
        [t.strip() for t in tasks.split(",") if t.strip()]
        if tasks else ROBOMME_TASKS
    )

    print(
        f"=== baselines (planner={planner}, n_episodes={n_episodes}, "
        f"max_steps={max_steps}, n_tasks={len(task_list)}) ==="
    )

    # Spawn one call per task (Modal scheduler handles GPU availability).
    # `.spawn()` returns immediately with a FunctionCall handle; we collect
    # results via .get() below.
    spawns = []
    for t in task_list:
        spawn = collect_rollouts.spawn(
            task_name=t,
            n_episodes=n_episodes,
            planner=planner,
            max_steps=max_steps,
        )
        spawns.append((t, spawn))
        print(f"  spawned: {t}")

    print(f"\nWaiting for {len(spawns)} containers to finish...")
    print("(Watch the Modal dashboard URL printed above for live progress.)\n")

    results: dict = {}
    for t, spawn in spawns:
        try:
            r = spawn.get()
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "spawn_error": repr(e)}
        results[t] = r
        client = r.get("client", {}) if isinstance(r, dict) else {}
        sr = client.get("success_rate")
        n_ok = client.get("n_ok")
        n_err = client.get("n_errors")
        wallclock = r.get("wallclock_s") if isinstance(r, dict) else None
        ok = r.get("ok") if isinstance(r, dict) else False
        ok_str = "OK " if ok else "ERR"
        traj = r.get("trajectories_dumped") if isinstance(r, dict) else None
        print(
            f"  [{ok_str}] {t:<22} success_rate={sr} "
            f"n_ok={n_ok} n_errors={n_err} traj={traj} wallclock={wallclock}s"
        )

    # Aggregate summary
    print(f"\n=== summary (planner={planner}) ===")
    rates = []
    for t, r in results.items():
        if not isinstance(r, dict):
            continue
        client = r.get("client", {})
        sr = client.get("success_rate") if isinstance(client, dict) else None
        if sr is not None:
            rates.append(sr)
    if rates:
        avg = round(sum(rates) / len(rates), 3)
        print(f"  avg success_rate across {len(rates)} tasks: {avg}")
    else:
        print("  no successful tasks to aggregate")


@app.local_entrypoint()
def sft_step(
    dataset_jsonl_path: str,
    adapter_in_path: str = "",
    num_epochs: float = 2.0,
    learning_rate: float = 5e-5,
    nproc_per_node: int = 1,
) -> None:
    """Local entrypoint: run one RestEM SFT step.

    Usage:
        # Train fresh LoRA from scratch (no warm start):
        modal run modal/app.py::sft_step \\
            --dataset-jsonl-path /cache/runs/restem_data/qwenvl_<timestamp>.jsonl

        # Continue training from the SFT'd Qwen3-VL adapter (RestEM iter 1):
        modal run modal/app.py::sft_step \\
            --dataset-jsonl-path /cache/runs/restem_data/qwenvl_<timestamp>.jsonl \\
            --adapter-in-path /cache/runs/ckpts/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200
    """
    result = restem_sft_step.remote(
        dataset_jsonl_path=dataset_jsonl_path,
        adapter_in_path=adapter_in_path,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        nproc_per_node=nproc_per_node,
    )
    print(f"=== restem_sft_step ===")
    for k, v in result.items():
        if k in ("stdout_tail", "stderr_tail"):
            print(f"--- {k} ---\n{v}\n--- end {k} ---")
        else:
            print(f"{k}: {v}")


@app.local_entrypoint()
def build_dataset(
    planner: str = "qwenvl",
    since_hours: float = 0.0,
) -> None:
    """Local entrypoint: build the RestEM iteration-0 SFT dataset from
    successful trajectories already on the volume.

    Usage:
        modal run modal/app.py::build_dataset
        modal run modal/app.py::build_dataset --planner qwenvl --since-hours 24
    """
    result = build_restem_dataset.remote(
        planner=planner,
        since_hours=since_hours,
    )
    print(f"=== build_restem_dataset (planner={planner}) ===")
    for k, v in result.items():
        if k == "examples_per_task" and isinstance(v, dict):
            print("examples_per_task:")
            for t in sorted(v.keys()):
                print(f"  {t:<22} {v[t]}")
        else:
            print(f"{k}: {v}")


@app.local_entrypoint()
def report(
    planner: str = "",
    since_hours: float = 0.0,
    max_per_task: int = 0,
) -> None:
    """Print baseline success-rate table from dumped trajectory JSONs.

    Usage:
        modal run modal/app.py::report                               # all data, both planners
        modal run modal/app.py::report --planner oracle              # filter to one planner
        modal run modal/app.py::report --since-hours 24              # only files modified
                                                                       in the last 24 hours
        modal run modal/app.py::report --max-per-task 50             # only the N most
                                                                       recent files per task
        modal run modal/app.py::report --since-hours 24 --max-per-task 50  # both filters

    The filters are most useful when the volume has accumulated trajectory
    dumps from older debug runs that share the same dir layout as the real
    baseline runs.
    """
    # Hard-coded suite grouping for nicer per-suite breakdown in the table.
    suite_of = {
        "BinFill": "counting", "StopCube": "counting",
        "PickXtimes": "counting", "SwingXtimes": "counting",
        "ButtonUnmask": "permanence", "VideoUnmask": "permanence",
        "VideoUnmaskSwap": "permanence", "ButtonUnmaskSwap": "permanence",
        "PickHighlight": "reference", "VideoRepick": "reference",
        "VideoPlaceButton": "reference", "VideoPlaceOrder": "reference",
        "MoveCube": "procedural", "InsertPeg": "procedural",
        "PatternLock": "procedural",
    }

    resp = aggregate_baselines.remote(
        planner=planner,
        since_hours=since_hours,
        max_per_task=max_per_task,
    )
    if not resp.get("ok"):
        print(f"ERROR: {resp.get('error')}")
        return

    f = resp.get("filter", {})
    print(
        f"[filter] planner={f.get('planner')}  "
        f"since_hours={f.get('since_hours')}  "
        f"max_per_task={f.get('max_per_task')}"
    )

    results = resp["results"]
    if not results:
        print("No trajectory JSONs found matching the filter.")
        return

    for p_name in sorted(results.keys()):
        tasks = results[p_name]
        print(f"\n=== planner: {p_name} ===")
        header = (
            f"{'task':<20} {'suite':<11} "
            f"{'succ':>4} {'fail':>4} {'tout':>4} {'tot':>4} {'rate':>6}"
        )
        print(header)
        print("-" * len(header))

        suite_rates = {}
        all_rates = []

        for task in sorted(tasks.keys(), key=lambda t: (suite_of.get(t, "zzz"), t)):
            t = tasks[task]
            sr = t["success_rate"]
            sr_str = f"{sr:.3f}" if sr is not None else "  -- "
            suite = suite_of.get(task, "?")
            print(
                f"{task:<20} {suite:<11} "
                f"{t['success']:>4} {t['fail']:>4} {t['timeout']:>4} "
                f"{t['total']:>4} {sr_str:>6}"
            )
            if sr is not None:
                suite_rates.setdefault(suite, []).append(sr)
                all_rates.append(sr)

        print("-" * len(header))
        for suite in sorted(suite_rates.keys()):
            rates = suite_rates[suite]
            print(
                f"  {suite:<28} avg over {len(rates):>2} tasks: "
                f"{sum(rates)/len(rates):.3f}"
            )
        if all_rates:
            print(
                f"  {'OVERALL':<28} avg over {len(all_rates):>2} tasks: "
                f"{sum(all_rates)/len(all_rates):.3f}"
            )


@app.local_entrypoint()
def inspect(planner: str = "oracle", task: str = "MoveCube") -> None:
    """Local entrypoint: print structure of dumped trajectory JSONs."""
    result = inspect_trajectories.remote(planner=planner, task=task)
    print(f"=== trajectory inspection (planner={planner}, task={task}) ===")
    for k, v in result.items():
        v_str = str(v)
        if len(v_str) > 800:
            v_str = v_str[:800] + " ...<truncated>"
        print(f"{k}: {v_str}")


@app.local_entrypoint()
def collect(
    task_name: str = "MoveCube",
    n_episodes: int = 5,
    planner: str = "qwenvl",
    max_steps: int = 200,
) -> None:
    """Local entrypoint: collect N episodes of one task on Modal.

    Usage:
        modal run modal/app.py::collect                    # 5 ep MoveCube, qwenvl
        modal run modal/app.py::collect --planner oracle   # 5 ep MoveCube, oracle
        modal run modal/app.py::collect --n-episodes 50 --planner oracle \\
            --task-name InsertPeg                          # full Kamal-style baseline run
    """
    result = collect_rollouts.remote(
        task_name=task_name,
        n_episodes=n_episodes,
        planner=planner,
        max_steps=max_steps,
    )
    print(
        f"=== robomme-rl rollout collection "
        f"(task={task_name}, n_episodes={n_episodes}, planner={planner}) ==="
    )
    print(f"ok: {result.get('ok')}")
    print(f"server_boot_s: {result.get('server_boot_s')}")
    print(f"wallclock_s: {result.get('wallclock_s')}")
    print(f"trajectories_dumped: {result.get('trajectories_dumped')}")
    client = result.get("client", {})
    if isinstance(client, dict):
        print(f"planner_load_s: {client.get('planner_load_s')}")
        print(f"success_rate: {client.get('success_rate')} "
              f"(n_ok={client.get('n_ok')}, n_errors={client.get('n_errors')})")
        eps = client.get("episodes", [])
        if eps:
            print("episodes:")
            for ep in eps:
                flag = ep.get("success_flag", "ERR")
                reward = ep.get("reward", "-")
                s = ep.get("episode_s", "-")
                err = ep.get("error", "")
                err_str = f" error={err[:80]}" if err else ""
                print(f"  ep {ep['episode_id']:>3}: flag={flag:>8} "
                      f"reward={reward} in {s}s{err_str}")
        if not client.get("ok"):
            # Surface anything that might tell us what happened.
            for k in ("error", "traceback", "parse_error", "no_sentinel"):
                v = client.get(k)
                if v:
                    print(f"--- {k} ---\n{str(v)[-2000:]}\n--- end ---")
            for k in ("stdout_tail", "stderr_tail"):
                v = client.get(k)
                if v:
                    print(f"--- {k} ---\n{v[-3000:]}\n--- end ---")


@app.local_entrypoint()
def qwenvl_check(
    task_name: str = "MoveCube",
    max_steps: int = 200,
) -> None:
    """Local entrypoint: run one Qwen3-VL-planner episode on Modal (2× A10G).

    Usage:
        modal run modal/app.py::qwenvl_check
        modal run modal/app.py::qwenvl_check --task-name InsertPeg --max-steps 400
    """
    result = test_qwenvl_episode.remote(task_name=task_name, max_steps=max_steps)
    print(
        f"=== robomme-rl Qwen3-VL episode "
        f"(task={task_name}, max_steps={max_steps}) ==="
    )
    for k, v in result.items():
        if k == "server_log_tail":
            print(f"--- server_log_tail ---\n{v}\n--- end log tail ---")
        elif k == "client" and isinstance(v, dict):
            print("client:")
            for ck, cv in v.items():
                cv_str = str(cv)
                if len(cv_str) > 1500:
                    cv_str = cv_str[:1500] + " ...<truncated>"
                print(f"  {ck}: {cv_str}")
        else:
            print(f"{k}: {v}")


@app.local_entrypoint()
def oracle_check(
    task_name: str = "MoveCube",
    max_steps: int = 200,
) -> None:
    """Local entrypoint: run one Oracle-driven episode (no Qwen3-VL) on Modal.

    Usage:
        modal run modal/app.py::oracle_check
        modal run modal/app.py::oracle_check --task-name InsertPeg --max-steps 400
    """
    result = test_oracle_episode.remote(task_name=task_name, max_steps=max_steps)
    print(f"=== robomme-rl oracle episode (task={task_name}, max_steps={max_steps}) ===")
    for k, v in result.items():
        if k == "server_log_tail":
            print(f"--- server_log_tail ---\n{v}\n--- end log tail ---")
        elif k == "client" and isinstance(v, dict):
            print("client:")
            for ck, cv in v.items():
                # Trim long fields for readability.
                cv_str = str(cv)
                if len(cv_str) > 1200:
                    cv_str = cv_str[:1200] + " ...<truncated>"
                print(f"  {ck}: {cv_str}")
        else:
            print(f"{k}: {v}")


@app.local_entrypoint()
def sim_check(
    task_name: str = "MoveCube",
    n_steps: int = 50,
    render_device: str = "cpu",
) -> None:
    """Local entrypoint: run one RoboMME episode in the sim (no models).

    Usage:
        modal run modal/app.py::sim_check
        modal run modal/app.py::sim_check --task-name InsertPeg --n-steps 100
        modal run modal/app.py::sim_check --render-device cuda  # retry GPU rendering
    """
    result = test_sim.remote(
        task_name=task_name, n_steps=n_steps, render_device=render_device,
    )
    print(
        f"=== robomme-rl sim smoke test "
        f"(task={task_name}, n_steps={n_steps}, render={render_device}) ==="
    )
    for k, v in result.items():
        print(f"{k}: {v}")


@app.local_entrypoint()
def serve_check() -> None:
    """Local entrypoint: boot the π0.5 VLA server, send one ping, tear down.

    Usage:
        modal run modal/app.py::serve_check

    First run is the slowest (~3-5 min) due to JAX kernel compilation on cold
    cache. Subsequent runs benefit from JAX's persistent compilation cache —
    we may want to point that at the volume too if it becomes a bottleneck.
    """
    result = test_serve_policy.remote()
    print("=== robomme-rl VLA serve smoke test ===")
    for k, v in result.items():
        if k == "server_log_tail":
            print(f"--- server_log_tail ---\n{v}\n--- end log tail ---")
        elif k == "client" and isinstance(v, dict):
            print("client:")
            for ck, cv in v.items():
                print(f"  {ck}: {cv}")
        else:
            print(f"{k}: {v}")


@app.local_entrypoint()
def test_planner() -> None:
    """Local entrypoint: load the SFT'd planner on an A10G and report VRAM.

    Usage:
        modal run modal/app.py::test_planner

    First run pulls Qwen3-VL-4B base weights from HF (~9 GB, ~5-10 min).
    Subsequent runs reuse the HF cache on the volume and are much faster.
    """
    result = test_load_planner.remote()
    print("=== robomme-rl planner load test ===")
    for k, v in result.items():
        print(f"{k}: {v}")


@app.local_entrypoint()
def download() -> None:
    """Local entrypoint: populates the cache volume with all checkpoints.

    Usage:
        modal run modal/app.py::download

    First run is slow (tens of GB from GCS + HF, ~10-30 min on a good link).
    Subsequent runs short-circuit and finish in seconds.
    """
    result = download_checkpoints.remote()
    print("=== robomme-rl checkpoint download ===")
    for k, v in result.items():
        print(f"{k}: {v}")
