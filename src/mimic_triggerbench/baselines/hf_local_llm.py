"""Single-GPU Hugging Face causal LM for LLM/RAG baselines (replaces cloud APIs).

Environment (typically from ``.env`` via ``load_dotenv``):

- ``HF_HOME`` — Hugging Face cache root (e.g. ``/data/Amin``); standard for ``huggingface_hub``.
- ``TRIGGERBENCH_HF_MODEL_ID`` — model repo id (default: ``Qwen/Qwen2.5-7B-Instruct``).
- ``TRIGGERBENCH_CUDA_DEVICE`` — device string (default: ``cuda:0``; set ``cuda:1`` when GPU 0 is in use).

No silent fallbacks: load, generate, and JSON parse failures raise.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import threading
from typing import Any, Dict, Optional


def _patch_transformers_if_flash_attn_broken() -> None:
    """If ``flash_attn`` is installed but fails to load (e.g. torch ABI mismatch), tell
    Transformers flash-attn is unavailable so Qwen2 and similar models import cleanly.
    Must run before any ``transformers`` model module is imported.
    """
    if importlib.util.find_spec("flash_attn") is None:
        return
    try:
        import flash_attn  # noqa: F401, PLC0415
    except Exception:
        import transformers.utils.import_utils as _iu
        import transformers.utils as _transformers_utils

        _fn_false = lambda: False
        _fn_false_ge = lambda *args, **kwargs: False
        # Patch defining module …
        _iu.is_flash_attn_2_available = _fn_false  # type: ignore[method-assign]
        _iu.is_flash_attn_greater_or_equal_2_10 = _fn_false  # type: ignore[method-assign]
        _iu.is_flash_attn_greater_or_equal = _fn_false_ge  # type: ignore[method-assign]
        # … and ``transformers.utils`` re-exports (``from .utils import is_flash_attn_2_available``).
        _transformers_utils.is_flash_attn_2_available = _fn_false  # type: ignore[method-assign]
        _transformers_utils.is_flash_attn_greater_or_equal_2_10 = _fn_false  # type: ignore[method-assign]
        _transformers_utils.is_flash_attn_greater_or_equal = _fn_false_ge  # type: ignore[method-assign]


_patch_transformers_if_flash_attn_broken()

from mimic_triggerbench.labeling.episode_models import Episode
from mimic_triggerbench.labeling.task_spec_models import TaskSpec

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
_MAX_NEW_TOKENS = 768

_shared_generator: Optional["LocalHFGenerator"] = None
_shared_lock = threading.Lock()


class LocalHFGeneratorError(RuntimeError):
    """Raised when the local model cannot load or generate."""


def get_default_model_id() -> str:
    return os.environ.get("TRIGGERBENCH_HF_MODEL_ID", _DEFAULT_MODEL_ID).strip()


def get_default_device() -> str:
    return os.environ.get("TRIGGERBENCH_CUDA_DEVICE", "cuda:0").strip()


def get_shared_local_generator(
    *,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
) -> "LocalHFGenerator":
    """Return a process-wide shared generator (one model load per process)."""
    global _shared_generator
    mid = model_id or get_default_model_id()
    dev = device or get_default_device()
    with _shared_lock:
        if _shared_generator is None:
            _shared_generator = LocalHFGenerator(model_id=mid, device=dev)
        elif _shared_generator.model_id != mid or _shared_generator.device_str != dev:
            raise LocalHFGeneratorError(
                "Shared generator already created with different model_id/device; "
                "use a single TRIGGERBENCH_HF_MODEL_ID / TRIGGERBENCH_CUDA_DEVICE per process."
            )
        return _shared_generator


def reset_shared_local_generator_for_tests() -> None:
    """Release the shared model (for pytest isolation)."""
    global _shared_generator
    with _shared_lock:
        if _shared_generator is not None:
            _shared_generator.unload()
            _shared_generator = None


class LocalHFGenerator:
    """Causal LM on one GPU: chat-formatted prompt in, text out."""

    def __init__(self, *, model_id: str, device: str) -> None:
        self.model_id = model_id
        self.device_str = device
        self._model = None
        self._tokenizer = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise LocalHFGeneratorError(
                "Install LLM extras: pip install -e '.[llm]' (torch, transformers, accelerate)."
            ) from e

        if self.device_str.startswith("cuda") and not torch.cuda.is_available():
            raise LocalHFGeneratorError(
                f"Device {self.device_str!r} requested but CUDA is not available."
            )

        logger.info("Loading HF model %s on %s …", self.model_id, self.device_str)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        dtype = torch.bfloat16 if self.device_str.startswith("cuda") else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        self._model.to(self.device_str)
        self._model.eval()

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        self._tokenizer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate(self, user_prompt: str, *, temperature: float = 0.1) -> str:
        """Run one generation; returns decoded new tokens only."""
        if self._model is None or self._tokenizer is None:
            raise LocalHFGeneratorError("Model not loaded.")

        import torch

        messages = [{"role": "user", "content": user_prompt}]
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = user_prompt

        inputs = self._tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.device_str) for k, v in inputs.items()}

        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._tokenizer.eos_token_id

        try:
            from transformers import GenerationConfig
        except ImportError as e:
            raise LocalHFGeneratorError(
                "Install LLM extras: pip install -e '.[llm]' (torch, transformers, accelerate)."
            ) from e

        if temperature and temperature > 0:
            gen_cfg = GenerationConfig(
                max_new_tokens=_MAX_NEW_TOKENS,
                pad_token_id=pad_id,
                do_sample=True,
                temperature=max(float(temperature), 1e-5),
                top_p=0.95,
            )
        else:
            # Fresh config avoids inheriting sample defaults (temperature/top_p/top_k) from the model card.
            gen_cfg = GenerationConfig(
                max_new_tokens=_MAX_NEW_TOKENS,
                pad_token_id=pad_id,
                do_sample=False,
            )

        with torch.inference_mode():
            out = self._model.generate(**inputs, generation_config=gen_cfg)

        in_len = inputs["input_ids"].shape[1]
        gen_ids = out[0, in_len:]
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.strip()


def _collapse_duplicate_json_root_open(text: str) -> str:
    """Some chat LMs emit two opening braces before the JSON object."""
    t = text.lstrip()
    if not t.startswith("{"):
        return text
    i = 1
    while i < len(t) and t[i] in " \t\n\r":
        i += 1
    if i < len(t) and t[i] == "{":
        return t[i:]
    return text


def _strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    return cleaned


def _repair_hf_json_text(s: str) -> str:
    """Best-effort fixes for common local-LM JSON glitches (spurious spaces in keys/values)."""
    out = s.replace("recommeded_action_families", "recommended_action_families")
    out = out.replace("recommeded_next_steps", "recommended_next_steps")
    out = out.replace("recommed_action_families", "recommended_action_families")
    out = out.replace("recommed_next_steps", "recommended_next_steps")
    # Extra `"` after opening quote: `" "urgency_level":` → `"urgency_level":`
    out = re.sub(r'"\s+"\s*([a-z_][a-z0-9_]*)"\s*:', r'"\1":', out)
    out = re.sub(
        r':\s*"\s+"\s*(high|low|medium|critical)"\s*([,}])',
        r':"\1"\2',
        out,
        flags=re.IGNORECASE,
    )
    # Array strings: do not consume the trailing comma (lookahead), or the next element is missed.
    _arr_end = r'(?=\s*(?:,|\]))'
    out = re.sub(r',\s*"\s+"\s*([a-z_][a-z0-9_]*)"' + _arr_end, r', "\1"', out)
    out = re.sub(r'\[\s*"\s+"\s*([a-z_][a-z0-9_]*)"' + _arr_end, r'["\1"', out)
    # Keys like `" urgency_level":` (single spurious space, no extra quote)
    out = re.sub(r'"\s+([a-z_][a-z0-9_]*)"\s*:', r'"\1":', out)
    out = re.sub(
        r':\s*"\s+(high|low|medium|critical)"\s*([,}])',
        r':"\1"\2',
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r',\s*"\s+([a-z_][a-z0-9_]*)"' + _arr_end, r', "\1"', out)
    out = re.sub(r'\[\s*"\s+([a-z_][a-z0-9_]*)"' + _arr_end, r'["\1"', out)
    # Array strings like `" "Administer IV …"` (spurious quote+space after opening quote).
    out = re.sub(r'(\[|,)\s*"\s+"\s*([^"]+)"', r'\1 "\2"', out)
    return out


def _first_balanced_json_object(s: str) -> str:
    """Return substring from first ``{`` through matching ``}``, respecting JSON string escapes."""
    start = s.find("{")
    if start < 0:
        raise LocalHFGeneratorError("No JSON object start '{' in model output.")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    raise LocalHFGeneratorError("Unclosed JSON object in model output.")


def parse_benchmark_json_response(text: str, spec: TaskSpec, _episode: Episode) -> Dict[str, Any]:
    """Parse model output as JSON; validate required fields. Raises on any failure."""
    cleaned = _repair_hf_json_text(_collapse_duplicate_json_root_open(_strip_markdown_fence(text)))
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        cleaned = _repair_hf_json_text(_first_balanced_json_object(cleaned))
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LocalHFGeneratorError(
            f"Model output is not valid JSON: {e}\n--- snippet ---\n{text[:800]}"
        ) from e

    if not isinstance(parsed, dict):
        raise LocalHFGeneratorError("Model JSON root must be an object.")

    required = (
        "trigger_detected",
        "urgency_level",
        "recommended_action_families",
        "recommended_next_steps",
        "confidence",
        "abstain",
    )
    missing = [k for k in required if k not in parsed]
    if missing:
        raise LocalHFGeneratorError(f"JSON missing required keys: {missing}")

    if not isinstance(parsed["trigger_detected"], bool):
        raise LocalHFGeneratorError("trigger_detected must be a boolean.")
    if parsed["urgency_level"] not in ("low", "medium", "high", "critical"):
        raise LocalHFGeneratorError(f"Invalid urgency_level: {parsed['urgency_level']!r}")
    if not isinstance(parsed["recommended_action_families"], list):
        raise LocalHFGeneratorError("recommended_action_families must be a list.")
    if not isinstance(parsed["recommended_next_steps"], list):
        raise LocalHFGeneratorError("recommended_next_steps must be a list.")
    if not isinstance(parsed["confidence"], (int, float)):
        raise LocalHFGeneratorError("confidence must be a number.")
    if not isinstance(parsed["abstain"], bool):
        raise LocalHFGeneratorError("abstain must be a boolean.")
    ar = parsed.get("abstain_reason")
    if ar is not None and not isinstance(ar, str):
        raise LocalHFGeneratorError("abstain_reason must be string or null.")

    valid_families = {af.name for af in spec.action_families}
    families = [f for f in parsed["recommended_action_families"] if f in valid_families]
    unknown = [f for f in parsed["recommended_action_families"] if f not in valid_families]
    if unknown:
        logger.warning("Ignoring unknown action families from model: %s", unknown)

    return {
        "trigger_detected": parsed["trigger_detected"],
        "urgency_level": parsed["urgency_level"],
        "recommended_action_families": families,
        "recommended_next_steps": [str(s) for s in parsed["recommended_next_steps"]],
        "confidence": max(0.0, min(1.0, float(parsed["confidence"]))),
        "abstain": parsed["abstain"],
        "abstain_reason": ar,
    }
