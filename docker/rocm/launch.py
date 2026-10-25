"""
VoxCPM WebUI launcher for the ROCm container.

Runs the upstream app.py unchanged, with a few runtime tweaks controlled by
environment variables:

  PORT / HOST          Gradio bind (default 8808 / 0.0.0.0)
  VOXCPM_MODEL         HF repo id or local path (default openbmb/VoxCPM2)
  VOXCPM_DEVICE        auto | cuda | cuda:N | cpu (ROCm shows up as "cuda")
  VOXCPM_DTYPE         float16 (default) | bfloat16 | float32. RDNA2 has no
                       fast bf16 path: fp16 is ~2.7x faster end-to-end on a
                       6700 XT (RTF 1.6 vs 4.3). Empty = checkpoint dtype.
  VOXCPM_OPTIMIZE      1 = torch.compile (CUDA graphs). Off by default: on
                       gfx1031 "reduce-overhead" crashes the GPU (memory
                       access fault) during warm-up.
  ASR_BACKEND          whisper (default, good for Indonesian) | sensevoice
                       (upstream: zh/en/ja/ko/yue only)
  ASR_MODEL            whisper model id (default openai/whisper-large-v3-turbo)
  ASR_LANGUAGE         force whisper language, e.g. "id"; empty = auto-detect
  ASR_DEVICE           device for ASR (default: same GPU as VoxCPM)
  TEXT_NUMBER_LANG     expand digits to words before synthesis, e.g. "id"
                       ("2026" -> "dua ribu dua puluh enam"); empty = off.
                       Upstream "normalize" only knows zh/en, so leave the
                       Text Normalization checkbox off for Indonesian.
"""

import argparse
import logging
import os
import re
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(APP_DIR, "app.py")) and APP_DIR != "/":
    APP_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)

import app as vox  # noqa: E402
import voxcpm.model.voxcpm as _vox_m1  # noqa: E402
import voxcpm.model.voxcpm2 as _vox_m2  # noqa: E402

logger = logging.getLogger("voxcpm.launch")


# ── runtime dtype override ───────────────────────────────────────────
def _make_dtype_picker(orig):
    def pick(device, configured_dtype):
        forced = os.environ.get("VOXCPM_DTYPE", "").strip().lower()
        if forced and device.startswith("cuda"):
            return forced
        return orig(device, configured_dtype)

    return pick


for _mod in (_vox_m1, _vox_m2):
    if hasattr(_mod, "pick_runtime_dtype"):
        _mod.pick_runtime_dtype = _make_dtype_picker(_mod.pick_runtime_dtype)


def env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ── torch.compile toggle ─────────────────────────────────────────────
_orig_init = vox.VoxCPMDemo.__init__


def _init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self.optimize = env_bool("VOXCPM_OPTIMIZE") and self.device.startswith("cuda")
    if os.environ.get("ASR_DEVICE"):
        self.asr_device = os.environ["ASR_DEVICE"]
    self._whisper = None
    logger.info(f"torch.compile optimize={self.optimize}, ASR={os.environ.get('ASR_BACKEND', 'whisper')}")


vox.VoxCPMDemo.__init__ = _init


# ── Whisper ASR (multilingual prompt transcription) ──────────────────
def _get_whisper(self):
    if self._whisper is not None:
        return self._whisper
    import torch
    from transformers import pipeline

    model_id = os.environ.get("ASR_MODEL", "openai/whisper-large-v3-turbo")
    on_gpu = self.asr_device.startswith("cuda")
    logger.info(f"Loading Whisper ASR: {model_id} on {self.asr_device}")
    self._whisper = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=self.asr_device,
        torch_dtype=torch.float16 if on_gpu else torch.float32,
    )
    return self._whisper


_orig_asr = vox.VoxCPMDemo.prompt_wav_recognition


def _prompt_wav_recognition(self, prompt_wav):
    if os.environ.get("ASR_BACKEND", "whisper").lower() != "whisper":
        return _orig_asr(self, prompt_wav)
    if not prompt_wav:
        return ""
    gen_kwargs = {"task": "transcribe"}
    lang = os.environ.get("ASR_LANGUAGE", "").strip()
    if lang:
        gen_kwargs["language"] = lang
    out = _get_whisper(self)(prompt_wav, chunk_length_s=30, generate_kwargs=gen_kwargs)
    return out["text"].strip()


vox.VoxCPMDemo.prompt_wav_recognition = _prompt_wav_recognition


# ── Number expansion (Indonesian by default) ─────────────────────────
_NUM_RE = re.compile(r"(?<![\w.,])(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?(?![\w])")


def expand_numbers(text: str, lang: str) -> str:
    from num2words import num2words

    comma = {"id": "koma", "ms": "perpuluhan"}.get(lang, "point")
    percent = {"id": "persen", "ms": "peratus"}.get(lang, "percent")

    def repl(m):
        whole = int(m.group(1).replace(".", ""))
        words = num2words(whole, lang=lang)
        if m.group(2):
            # read decimals digit by digit: 3,25 -> tiga koma dua lima
            words += f" {comma} " + " ".join(num2words(int(d), lang=lang) for d in m.group(2))
        return words

    text = re.sub(r"(\d)\s*%", rf"\1 {percent}", text)
    if lang == "id":
        # Rp 1.500 -> 1.500 rupiah
        text = re.sub(r"\bRp\.?\s*(\d[\d.,]*\d|\d)", r"\1 rupiah", text)
        # 07.30 / 7:30 -> tujuh tiga puluh (only HH.MM with a valid hour/minute)
        text = re.sub(
            r"(?<![\d.,])([01]?\d|2[0-3])[.:]([0-5]\d)(?!\d|[.,]\d)",
            lambda m: num2words(int(m.group(1)), lang=lang)
            + ("" if m.group(2) == "00" else " " + num2words(int(m.group(2)), lang=lang)),
            text,
        )
    return _NUM_RE.sub(repl, text)


_orig_build = vox.VoxCPMDemo._build_generate_kwargs


def _build_generate_kwargs(self, *, final_text, **kwargs):
    lang = os.environ.get("TEXT_NUMBER_LANG", "").strip()
    if lang and final_text:
        try:
            final_text = expand_numbers(final_text, lang)
        except Exception as e:  # never block synthesis on normalization
            logger.warning(f"number expansion failed: {e}")
    return _orig_build(self, final_text=final_text, **kwargs)


vox.VoxCPMDemo._build_generate_kwargs = _build_generate_kwargs


def check_gpu_kernels():
    """Fail fast with a useful hint instead of 'invalid device function' mid-request."""
    import torch

    if not torch.cuda.is_available():
        logger.warning("No ROCm GPU visible (check /dev/kfd, /dev/dri and render group); running on CPU")
        return
    props = torch.cuda.get_device_properties(0)
    try:
        (torch.ones(8, device="cuda") * 2).sum().item()
    except RuntimeError as e:
        override = os.environ.get("HSA_OVERRIDE_GFX_VERSION") or "unset"
        sys.exit(
            f"GPU {props.name} ({props.gcnArchName}) can't run the bundled ROCm kernels: {e}\n"
            f"HSA_OVERRIDE_GFX_VERSION={override}. RDNA2 cards (gfx1031-1036) need 10.3.0, "
            "RDNA3 gfx1101-1103 need 11.0.0; docker/rocm/run.sh sets this automatically."
        )
    logger.info(f"GPU: {props.name} ({props.gcnArchName}), {props.total_memory / 2**30:.1f} GB")


if __name__ == "__main__":
    check_gpu_kernels()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8808")))
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--device", default=os.environ.get("VOXCPM_DEVICE", "auto"))
    args = parser.parse_args()
    vox.run_demo(
        model_id=args.model_id,
        server_name=args.host,
        server_port=args.port,
        device=args.device,
    )
