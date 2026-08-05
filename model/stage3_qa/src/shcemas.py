from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class ExtractedField(BaseModel):
    value: str
    confidence: float
    bbox: List[int]


class KIEResult(BaseModel):
    invoice_number: Optional[ExtractedField] = None
    tax_code: Optional[ExtractedField] = None
    date: Optional[ExtractedField] = None
    total_amount: Optional[ExtractedField] = None


class PreprocessingMeta(BaseModel):
    deskew: bool = False
    clahe: bool = False
    sharpen: bool = False
    perspective_crop: bool = False


class OCRTextBlock(BaseModel):
    text: str
    confidence: float
    bbox: List[List[int]]


class PipelineOutput(BaseModel):
    status: str
    preprocessing: PreprocessingMeta
    extracted_data: Dict[str, Any]
    accounting_audit: Optional[Dict[str, Any]] = None
