"""End-to-end Qwen2-VL adapter for extraction and document question answering."""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import settings


EXTRACTION_PROMPT = """Đọc hóa đơn/tài liệu trong ảnh và trả về DUY NHẤT một JSON hợp lệ.
Không thêm Markdown hay giải thích. Dùng chính xác các khoá sau:
{
  "store_name": "... hoặc null",
  "invoice_number": "... hoặc null",
  "tax_code": "... hoặc null",
  "invoice_date": "... hoặc null",
  "total_amount": "... hoặc null"
}
Chỉ điền thông tin nhìn thấy rõ trong tài liệu."""


@dataclass(frozen=True)
class VLMField:
    key: str
    value: str
    confidence: float | None = None


class VLMRuntimeError(RuntimeError):
    """Raised when the local Qwen2-VL runtime or checkpoints are unavailable."""


def _complete_adapter_path() -> str | None:
    adapter_path = settings.vlm_adapter_path or (
        settings.model_dir / "stage1_vlm" / "output" / "lora_adapters"
    )
    adapter_dir = Path(adapter_path)
    if adapter_dir.is_dir() and any(adapter_dir.glob("adapter_model.*")):
        return str(adapter_dir)
    return None


@lru_cache(maxsize=1)
def _get_engine():
    """Load the team's fine-tuned VLM once per FastAPI worker."""
    project_root = str(settings.model_dir.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    adapter_path = _complete_adapter_path()
    if not adapter_path and not settings.vlm_allow_base_model:
        raise VLMRuntimeError(
            "Không tìm thấy trọng số adapter_model.* trong VLM_ADAPTER_PATH. "
            "Hãy mount LoRA adapter hoặc đặt VLM_ALLOW_BASE_MODEL=true."
        )
    try:
        from model.stage1_vlm.src.inference import VQAEngine

        return VQAEngine(adapter_dir=adapter_path, base_model=settings.vlm_base_model)
    except ImportError as exc:
        raise VLMRuntimeError(
            "Thiếu VLM dependencies. Cài backend requirements trước khi chạy Qwen2-VL."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - keep model tracebacks out of API responses
        raise VLMRuntimeError(f"Không thể tải Qwen2-VL: {exc}") from exc


def answer_question(image_path: str, question: str) -> str:
    """Run the VLM on a preprocessed document image and a natural-language question."""
    if not Path(image_path).is_file():
        raise VLMRuntimeError("Không tìm thấy ảnh đã tiền xử lý cho document này.")
    return _get_engine().extract_and_answer(image_path=image_path, question=question)


def _parse_json_response(response: str) -> dict[str, Any] | None:
    """Accept bare JSON or fenced JSON emitted by a generative VLM."""
    candidate = response.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1)
    else:
        object_match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if object_match:
            candidate = object_match.group(0)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_fields(image_path: str) -> tuple[list[VLMField], str]:
    """Ask Qwen2-VL for structured fields and preserve raw output for auditability."""
    response = answer_question(image_path, EXTRACTION_PROMPT)
    parsed = _parse_json_response(response)
    if not parsed:
        return [VLMField(key="vlm_response", value=response)], response

    fields: list[VLMField] = []
    for key in ("store_name", "invoice_number", "tax_code", "invoice_date", "total_amount"):
        value = parsed.get(key)
        if value is not None and str(value).strip():
            fields.append(VLMField(key=key, value=str(value).strip()))
    return fields or [VLMField(key="vlm_response", value=response)], response
