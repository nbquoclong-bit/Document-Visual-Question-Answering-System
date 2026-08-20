# Frontend — Sổ Hoá Đơn (Document VQA)

React 19 + Vite, Tailwind (qua CDN — không cần cấu hình PostCSS riêng), axios,
lucide-react. Tích hợp trực tiếp với backend FastAPI trong `backend/`.

## 1. Cài đặt & chạy

```bash
cd frontend
npm install
cp .env.example .env      # sửa lại nếu backend chạy ở địa chỉ khác
npm run dev
```

Mở **http://localhost:5173** — nhớ **backend phải đang chạy** ở `http://localhost:8000`
(hoặc địa chỉ khai trong `.env`), nếu không mọi lời gọi API sẽ báo lỗi.

## 2. Luồng sử dụng (khớp đúng API backend)

```
Upload ảnh → Xử lý tài liệu → Xem field trích xuất → Hỏi đáp → Xuất JSON
```

1. Kéo-thả hoặc chọn ảnh hoá đơn (`UploadZone`) → gọi `POST /documents/upload`.
2. Bấm **"Xử lý tài liệu"** → gọi `POST /documents/{id}/process` → hiển thị field
   trích xuất theo dạng dòng hoá đơn (`FieldsLedger`).
3. Bấm vào một dòng field, hoặc đặt câu hỏi ở khung **Hỏi đáp** → gọi
   `POST /documents/{id}/ask` → khung đỏ ("con dấu") tự động hiện lên đúng vị trí
   bằng chứng trên ảnh gốc (`DocumentViewer`).
4. Bấm **"Xuất kết quả JSON"** → mở `GET /documents/{id}/export` ở tab mới.

## 3. Cấu trúc thư mục

```
frontend/
├── index.html             # cấu hình theme Tailwind (màu/font) nằm ở đây
├── .env.example            # VITE_API_URL — địa chỉ backend
├── src/
│   ├── main.jsx             # entry point
│   ├── App.jsx              # điều phối toàn bộ state + luồng nghiệp vụ
│   ├── api.js               # NƠI DUY NHẤT gọi backend (axios) — sửa ở đây nếu API đổi
│   ├── index.css            # chỉ chứa phần Tailwind CDN không làm được (nền, scrollbar...)
│   └── components/
│       ├── UploadZone.jsx      # kéo-thả / chọn file
│       ├── DocumentViewer.jsx  # ảnh + khung highlight bằng chứng (bbox)
│       ├── FieldsLedger.jsx    # danh sách field trích xuất
│       ├── QAConsole.jsx       # hỏi-đáp
│       └── StatusBadge.jsx     # con dấu trạng thái
```

## 4. Vì sao Tailwind dùng CDN thay vì cài qua npm

`index.html` nạp Tailwind qua `<script src="https://cdn.tailwindcss.com">` (đúng như
bản khung ban đầu của team) thay vì cài `tailwindcss` qua npm + PostCSS. Cấu hình màu/
font tuỳ biến (theme "sổ hoá đơn") nằm trong thẻ `<script>` ngay dưới đó trong
`index.html`, dùng `tailwind.config = {...}`.

**Đánh đổi cần biết:** cách này không cần bước build riêng cho CSS (đơn giản hơn cho
demo/đồ án), nhưng Tailwind phải tải và xử lý class ngay trong trình duyệt mỗi lần tải
trang — không tối ưu cho production thật. Nếu sau này muốn nâng cấp, có thể chuyển sang
cài `tailwindcss` qua npm (`npm install -D tailwindcss @tailwindcss/vite`) mà không cần
đổi lại bất kỳ class nào đã viết trong JSX.

## 5. Điểm quan trọng khi tích hợp

- **`api.js` là ranh giới duy nhất với backend.** Nếu backend đổi cấu trúc response
  hay thêm endpoint, chỉ cần sửa file này.
- **`evidence_bbox` / field `bbox`** phải là mảng 4 số `[x1, y1, x2, y2]` tính bằng
  pixel trên ảnh GỐC — `DocumentViewer.jsx` tự quy đổi sang phần trăm dựa vào
  `naturalWidth/naturalHeight` của ảnh, nên hiển thị đúng dù màn hình to nhỏ thế nào.
- Muốn đổi cổng/domain backend: sửa `VITE_API_URL` trong `.env`, không cần sửa code.

## 6. Lint

```bash
npm run lint    # chạy oxlint theo .oxlintrc.json đã có sẵn
```

## 7. Build production

```bash
npm run build      # xuất ra thư mục dist/
npm run preview    # xem thử bản build
```
