# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Project: SmartDoc AI - Document Visual Question Answering & Information Extraction
> Team: Team Boboiboys (Team 5)
> Repo: https://github.com/nbquoclong-bit/Document-Visual-Question-Answering-System

## Role

Model Lead covering Stage 0 through Stage 3 of the AI pipeline. Responsible
for preprocessing, OCR, layout understanding, and the accounting QA model.
Backend and frontend are handled by other team members.

## Team Assignments

- Le Minh Sang: Stage 2 (LayoutLMv3 KIE + Visual Grounding)
- Nguyen Van Nhat Nam: Stage 0 (adaptive preprocessing) and Stage 1 (PaddleOCR fine-tuning)
- Nguyen Ba Quoc Long: Stage 3 (Qwen2.5-1.5B LLM with QLoRA) and MLOps lead
- Tran Hoang Minh Thien: Backend (FastAPI)
- Trinh Minh Duc Hoang: Frontend (React)

## Pipeline Overview

The full pipeline has four stages:

- Stage 0 - Adaptive Preprocessing with OpenCV. Handles quality assessment, smart auto-routing (digital PDF vs scan/image), conditional deskew, CLAHE contrast enhancement, and perspective crop. Input is PDF or image.
- Stage 1 - PaddleOCR fine-tuned for Vietnamese invoices. Outputs text plus bounding boxes.
- Stage 2 - LayoutLMv3 for Key Information Extraction. Takes OCR output and produces structured JSON with entity values, confidence scores, and bounding box coordinates. Handles token classification with IOB/BIO tagging.
- Stage 3 - Qwen2.5-1.5B fine-tuned with Unsloth QLoRA for accounting QA. Reads JSON from Stage 2 only, performs math validation, and answers user questions. Zero hallucination guarantee because it never sees raw images.

## Common Commands

Backend (FastAPI) - run from backend/ directory:

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Frontend (React) - run from frontend/ directory:

```bash
npm install
npm run dev
```

## Tech Stack

- Stage 0: OpenCV, NumPy
- Stage 1: PaddleOCR, OpenCV
- Stage 2: microsoft/layoutlmv3-base, PyTorch, Hugging Face Transformers
- Stage 3: Qwen/Qwen2.5-1.5B-Instruct, Unsloth, QLoRA, PyTorch
- Backend: FastAPI, Uvicorn, LangChain
- Frontend: React, Zustand, HTML5 Canvas
- Database: SQLite
- Deployment: Docker, Hugging Face Spaces

## Training Data

- OCR training: MC_OCR 2021 (~1.1k invoice images) plus internal invoice dataset
- KIE training: ViOCRVQA (~28k images), DocVQA (~12k+ images), SROIE and CORD (2k+ standardized IOB-tagged invoices)
- LLM training: Synthetic Vietnamese accounting QA dataset

## Output Example

{
  "status": "success",
  "processing_metadata": {
    "preprocessing_applied": ["Deskew", "CLAHE"],
    "latency_seconds": 1.15
  },
  "extracted_data": {
    "invoice_number": {"value": "HD0082391", "confidence": 0.985, "box": [450, 120, 580, 145]},
    "tax_code": {"value": "0312345678", "confidence": 0.991, "box": [120, 210, 280, 230]},
    "total_amount": {"value": "1,500,000", "confidence": 0.978, "box": [610, 820, 750, 845]}
  },
  "accounting_audit": {
    "is_math_valid": true,
    "audit_note": "Tien hang (1,363,636 VND) + Thue VAT 10% (136,364 VND) khop hoan toan voi Tong tien (1,500,000 VND)."
  }
}

## Work Style

Communicate in Vietnamese. Focus on AI pipeline stages 0-3 (preprocessing,
OCR, LayoutLMv3 KIE, Qwen LLM). Prioritize working code and direct impact
over reports and diverse documentation. No prompt limit - I will review
everything and remove what is unnecessary.
