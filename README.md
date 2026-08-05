# Document Visual Question Answering System Using OCR and Multimodal Architecture for Automated Document Processing
## Authors

* [Nguyễn Bá Quốc Long](https://github.com/nbquoclong-bit)
* [Lê Minh Sang](https://github.com/LeMinhSang241)
* [Nguyễn Văn Nhật Nam](https://github.com/uynd)
* [Trần Hoàng Minh Thiên](https://github.com/TranHoangMinhThien)
* [Trịnh Minh Đức Hoàng](https://github.com/Emiya2902)

## System Pipeline

```mermaid
flowchart TD
    A[📄 Input File: JPG / PNG / PDF] --> B[🖼️ Image Preprocessing\nOpenCV / Pillow]
    B --> C[🔍 PaddleOCR\nText Detection & Recognition]
    
    C -->|Text + Bounding Boxes| D[🤖 LayoutLMv3 / Multimodal AI\nFeature Extraction & KIE]
    
    D --> E1[🏷️ Key Information Extraction\nVendor, Date, Total, Tax ID]
    D --> E2[💬 Visual Question Answering\nQwen2-VL / Qwen2.5]
    
    E1 --> F[🎯 Highlight Evidence & Format JSON]
    E2 --> F
    
    F --> G[⚡ FastAPI Backend]
    G --> H[🖥️ React Web UI Dashboard]
```
## Dataset
