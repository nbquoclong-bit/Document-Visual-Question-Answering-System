import os
import sys
import time
import re

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["KAGGLE_API_TOKEN"] = "KGAT_543b77ae9398d7062e33f1934b10c69d"

from kaggle.api.kaggle_api_extended import KaggleApi

def get_live_gradio_link():
    api = KaggleApi()
    api.authenticate()
    kernel_slug = "qwen2-vl-finetuned-test"
    kernel_id = f"lminhsang241/{kernel_slug}"
    
    print("=" * 80)
    print("⏳ ĐANG CHỜ GPU TESLA T4 KHỞI TẠO MÔ HÌNH VÀ SINH ĐƯỜNG LINK WEB DEMO...")
    print("=" * 80)
    
    for step in range(40):
        time.sleep(10)
        try:
            status_res = api.kernels_status(kernel_id)
            status = status_res.get("status", "unknown").upper()
            
            with api.build_kaggle_client() as kaggle:
                from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
                req = ApiListKernelSessionOutputRequest()
                req.user_name = "lminhsang241"
                req.kernel_slug = kernel_slug
                resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
                
                if resp.log:
                    match = re.search(r"https://[a-zA-Z0-9-]+\.gradio\.live", resp.log)
                    if match:
                        url = match.group(0)
                        print("\n" + "=" * 80)
                        print("🎉 ĐÃ KHỞI CHẠY THÀNH CÔNG WEB DEMO TRÊN GPU TESLA T4!")
                        print(f"👉 ĐƯỜNG LINK WEB CỦA BẠN: {url}")
                        print("=" * 80)
                        return url
            print(f"[{time.strftime('%H:%M:%S')}] Trạng thái: {status} (Đang nạp weights lên VRAM...)")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Đang chờ: {e}")

if __name__ == "__main__":
    get_live_gradio_link()
