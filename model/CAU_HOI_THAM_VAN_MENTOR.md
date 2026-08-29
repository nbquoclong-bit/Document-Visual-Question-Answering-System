# 📩 MẪU TIN NHẮN THAM VẤN ANH MENTOR (DÀNH CHO NHÓM)

Bạn có thể copy trực tiếp đoạn tin nhắn dưới đây để gửi cho Anh Mentor:

---

> **Chào anh ạ, nhóm em chuẩn bị cho buổi báo cáo đồ án phần Model & Đánh giá hiệu năng DocVQA (Qwen2-VL-2B + QLoRA). Tụi em có một số điểm kỹ thuật muốn xin ý kiến tham vấn từ anh để bài báo cáo thuyết phục nhất ạ:**
> 
> 1. **Về Metrics đánh giá:** Nhóm em hiện đang dùng bộ đo chuẩn của **DocVQA Challenge** gồm **ANLS ($\tau=0.5$)** và **Exact Match (EM)** trên tập validation (Base Model: ANLS 54.3%, EM 30% $\rightarrow$ LoRA Model: ANLS 89.2%, EM 75%). Theo anh, khi báo cáo với Hội đồng thì có cần bổ sung thêm chỉ số **F1-Token** hoặc **Latency P95** không ạ?
> 
> 2. **Về Kiến trúc & Edge/IoT Deployment:** Adapter LoRA của nhóm em sau khi train chỉ nặng **73.9 MB** (huấn luyện 0.2% tham số qua 7 module). Nếu nói về tiềm năng tích hợp lên thiết bị Edge/IoT hoặc App mobile, nhóm nên nhấn mạnh điểm mạnh OTA cập nhật adapter nhẹ nhàng hay việc chuyển đổi sang ONNX Runtime/TensorRT-LLM ạ?
> 
> 3. **Về Kịch bản Demo:** Khi demo trực tiếp, tụi em đang chạy Web Gradio trên GPU Tesla T4 (thời gian phản hồi ~1.5s/câu hỏi). Để phòng ngừa rủi ro mạng chập chờn trong phòng phản biện, anh có gợi ý phương án chuẩn bị video demo dự phòng như thế nào cho chuyên nghiệp nhất không ạ?
> 
> 4. **Về Bố cục Slide phần Model:** Nhóm em chia 3 slide chính:
>    - Slide 1: Kiến trúc Qwen2-VL-2B, cơ chế M-RoPE 2D và cấu hình QLoRA ($r=16, \alpha=32$).
>    - Slide 2: Bảng so sánh định lượng (ANLS, EM, xử lý tiếng Việt).
>    - Slide 3: Case study định tính thực tế trên hóa đơn Highlands Coffee (khắc phục lỗi sai dấu thanh và lỗi câu trả lời lan man).
>    Anh xem qua giúp nhóm em xem bố cục này đã đủ chặt chẽ và làm nổi bật được giá trị đóng góp của nhóm chưa ạ?
> 
> **Nhóm em cảm ơn anh rất nhiều ạ!**
