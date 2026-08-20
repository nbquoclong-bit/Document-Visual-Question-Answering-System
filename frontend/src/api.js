/**
 * api.js — nơi DUY NHẤT trong frontend gọi tới backend.
 *
 * Mọi component khác không tự gọi axios/fetch — chúng import các hàm ở đây.
 * Lý do: nếu địa chỉ backend, cấu trúc response, hay header xác thực đổi
 * sau này, chỉ cần sửa một file, không phải lục từng component.
 */
import axios from "axios";

// Relative URL works through nginx in Docker; VITE_API_URL may point to FastAPI in dev.
const BASE_URL = import.meta.env.VITE_API_URL || "/api/v1";

const client = axios.create({ baseURL: BASE_URL });

/** Rút gọn message lỗi từ response FastAPI ({"detail": "..."}) hoặc lỗi mạng. */
export function getErrorMessage(error) {
  if (axios.isAxiosError(error)) {
    return error.response?.data?.detail || error.message;
  }
  return error?.message || "Đã có lỗi không xác định";
}

/** Upload ảnh hoá đơn. Trả về { document_id, status, ... } */
export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await client.post("/documents/upload", formData);
  return data;
}

/** Chạy OCR + trích xuất field cho một document đã upload. */
export async function processDocument(documentId) {
  const { data } = await client.post(`/documents/${documentId}/process`);
  return data;
}

/** Lấy toàn bộ trạng thái + kết quả hiện tại của một document. */
export async function getDocument(documentId) {
  const { data } = await client.get(`/documents/${documentId}`);
  return data;
}

/** Đặt câu hỏi trên một document đã processed. */
export async function askQuestion(documentId, question) {
  const { data } = await client.post(`/documents/${documentId}/ask`, { question });
  return data;
}

/** URL ảnh gốc — dùng trực tiếp trong thẻ <img src=...>. */
export function getImageUrl(documentId) {
  return `${BASE_URL}/documents/${documentId}/image`;
}

/** URL export JSON — dùng cho link tải file. */
export function getExportUrl(documentId) {
  return `${BASE_URL}/documents/${documentId}/export`;
}
