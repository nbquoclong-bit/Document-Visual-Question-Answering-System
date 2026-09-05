"""
Module Schema Adapter chuẩn công nghiệp (Adapter Pattern).
Chuyển đổi dữ liệu hóa đơn trích xuất từ VLM (Canonical Schema)
sang cấu trúc dữ liệu của các phần mềm kế toán / ERP phổ biến:
- MISA meInvoice / MISA SME
- FAST Accounting
- SAP S/4HANA (BAPI Incoming Invoice)
- Canonical Standard (Chuẩn hóa quốc tế)
"""

import json
import re
from typing import Any, Dict, List, Optional


class SchemaAdapter:
    """
    Adapter Pattern để chuyển đổi thực thể trích xuất từ VLM
    thành schema đầu vào của từng hệ thống kế toán doanh nghiệp.
    """

    CANONICAL_KEYS = {
        "store_name": ["store_name", "seller", "company", "vendor", "ten_don_vi", "tên đơn vị", "tên bên bán"],
        "invoice_number": ["invoice_number", "invoice_no", "receipt_no", "so_hoa_don", "số hóa đơn", "so_hd"],
        "tax_code": ["tax_code", "tax_id", "mst", "ma_so_thue", "mã số thuế"],
        "invoice_date": ["invoice_date", "date", "timestamp", "ngay_lap", "ngày lập", "ngay_hd"],
        "total_amount": ["total_amount", "total", "total_cost", "tong_tien", "tổng tiền", "thanh_toan"],
        "address": ["address", "dia_chi", "địa chỉ", "vendor_address"],
        "items": ["items", "item_name", "items_list", "danh_sach_hang", "hang_hoa", "mặt hàng"]
    }

    @classmethod
    def parse_vlm_response(cls, raw_input: Any) -> Dict[str, Any]:
        """Chuẩn hóa dữ liệu thô từ VLM thành Dictionary."""
        if isinstance(raw_input, dict):
            return raw_input

        if not isinstance(raw_input, str):
            return {}

        text = raw_input.strip()
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            text = json_match.group(1).strip()
        else:
            curly_match = re.search(r"(\{[\s\S]*\})", text)
            if curly_match:
                text = curly_match.group(1).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        result = {}
        lines = text.split("\n")
        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                k_clean = k.strip().lower().replace("-", "").replace("*", "").strip()
                v_clean = v.strip().replace("*", "").strip()
                result[k_clean] = v_clean
        return result

    @classmethod
    def to_canonical(cls, raw_data: Any) -> Dict[str, Any]:
        """BƯỚC 1: Bóc tách và chuẩn hóa về Canonical Schema (Schema trung gian bất biến)."""
        data = cls.parse_vlm_response(raw_data)
        canonical = {
            "store_name": "",
            "invoice_number": "",
            "tax_code": "",
            "invoice_date": "",
            "total_amount": "",
            "address": "",
            "items": []
        }

        for std_key, alias_list in cls.CANONICAL_KEYS.items():
            for key, val in data.items():
                k_norm = str(key).strip().lower().replace(" ", "_").replace("-", "_")
                if k_norm == std_key or any(alias in k_norm for alias in alias_list):
                    canonical[std_key] = val
                    break

        if "items" in canonical and isinstance(canonical["items"], str):
            items_raw = canonical["items"].split("\n")
            canonical["items"] = [it.strip().lstrip("-*•123456789. ") for it in items_raw if it.strip()]

        return canonical

    @classmethod
    def to_misa(cls, raw_data: Any) -> Dict[str, Any]:
        """BƯỚC 2A: Ánh xạ sang schema MISA meInvoice / MISA SME chuẩn."""
        can = cls.to_canonical(raw_data)
        raw_total = str(can.get("total_amount", "0")).replace(",", "").replace(".", "").replace("VND", "").replace("đ", "").strip()
        num_total = int(re.sub(r"\D", "", raw_total)) if re.sub(r"\D", "", raw_total) else 0

        misa_items = []
        raw_items = can.get("items", [])
        if isinstance(raw_items, list):
            for idx, it in enumerate(raw_items, start=1):
                if isinstance(it, dict):
                    misa_items.append({
                        "LineNumber": idx,
                        "ItemName": it.get("name", it.get("item", "Hàng hóa")),
                        "Quantity": it.get("quantity", 1),
                        "UnitPrice": it.get("unit_price", 0),
                        "Amount": it.get("amount", 0)
                    })
                else:
                    misa_items.append({
                        "LineNumber": idx,
                        "ItemName": str(it),
                        "Quantity": 1,
                        "UnitPrice": 0,
                        "Amount": 0
                    })

        return {
            "TargetSystem": "MISA meInvoice / MISA SME",
            "SchemaVersion": "v2.1",
            "HoaDon": {
                "ThongTinChung": {
                    "SoHoaDon": can.get("invoice_number", ""),
                    "NgayLap": can.get("invoice_date", ""),
                    "HinhThucThanhToan": "TM/CK"
                },
                "DonViBanHang": {
                    "TenDonVi": can.get("store_name", ""),
                    "MaSoThue": can.get("tax_code", ""),
                    "DiaChi": can.get("address", "")
                },
                "ChiTietHangHoa": misa_items,
                "TongTienThanhToan": num_total,
                "DonViTienTe": "VND"
            }
        }

    @classmethod
    def to_sap_erp(cls, raw_data: Any) -> Dict[str, Any]:
        """BƯỚC 2B: Ánh xạ sang schema SAP S/4HANA BAPI Incoming Invoice."""
        can = cls.to_canonical(raw_data)
        raw_total = str(can.get("total_amount", "0")).replace(",", "").replace(".", "").replace("VND", "").replace("đ", "").strip()
        num_total = float(re.sub(r"\D", "", raw_total)) if re.sub(r"\D", "", raw_total) else 0.0

        sap_lines = []
        raw_items = can.get("items", [])
        if isinstance(raw_items, list):
            for idx, it in enumerate(raw_items, start=1):
                item_desc = it.get("name", str(it)) if isinstance(it, dict) else str(it)
                sap_lines.append({
                    "INVOICE_DOC_ITEM": f"{idx:06d}",
                    "ITEM_TEXT": item_desc,
                    "QUANTITY": 1.0,
                    "ITEM_AMOUNT": 0.0
                })

        return {
            "TargetSystem": "SAP S/4HANA ERP (BAPI_INCOMINGINVOICE_CREATE)",
            "HEADERDATA": {
                "INVOICE_DOC_TYPE": "RE",
                "REF_DOC_NO": can.get("invoice_number", ""),
                "DOC_DATE": can.get("invoice_date", ""),
                "VENDOR_NAME": can.get("store_name", ""),
                "TAX_NUMBER_1": can.get("tax_code", ""),
                "VENDOR_ADDRESS": can.get("address", ""),
                "GROSS_AMOUNT": num_total,
                "CURRENCY": "VND"
            },
            "ITEMDATA": sap_lines
        }

    @classmethod
    def to_fast_accounting(cls, raw_data: Any) -> Dict[str, Any]:
        """BƯỚC 2C: Ánh xạ sang schema FAST Accounting 11."""
        can = cls.to_canonical(raw_data)
        raw_total = str(can.get("total_amount", "0")).replace(",", "").replace(".", "").replace("VND", "").replace("đ", "").strip()
        num_total = int(re.sub(r"\D", "", raw_total)) if re.sub(r"\D", "", raw_total) else 0

        fast_details = []
        raw_items = can.get("items", [])
        if isinstance(raw_items, list):
            for it in raw_items:
                fast_details.append({
                    "ten_vt": it.get("name", str(it)) if isinstance(it, dict) else str(it),
                    "so_luong": 1,
                    "tien": 0
                })

        return {
            "TargetSystem": "FAST Accounting System",
            "ChungTu": {
                "so_ct": can.get("invoice_number", ""),
                "ngay_ct": can.get("invoice_date", ""),
                "ten_dt": can.get("store_name", ""),
                "ma_so_thue": can.get("tax_code", ""),
                "dia_chi": can.get("address", ""),
                "t_tt": num_total,
                "danh_sach_hang": fast_details
            }
        }

    @classmethod
    def adapt(cls, raw_data: Any, target_format: str = "canonical") -> Dict[str, Any]:
        """Hàm điều hướng chung (Facade Method)."""
        fmt = str(target_format).lower().strip()
        if "misa" in fmt:
            return cls.to_misa(raw_data)
        elif "sap" in fmt:
            return cls.to_sap_erp(raw_data)
        elif "fast" in fmt:
            return cls.to_fast_accounting(raw_data)
        else:
            return cls.to_canonical(raw_data)
