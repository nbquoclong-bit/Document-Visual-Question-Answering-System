import os
import sys
import json
import time
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["KAGGLE_API_TOKEN"] = "KGAT_c165b4251cf4050d1bc1bd1fd5b67156"

from kaggle.api.kaggle_api_extended import KaggleApi

def monitor_kernel():
    api = KaggleApi()
    api.authenticate()
    kernel_id = "lminhsang241/qwen2-vl-receipt-vqa-golden"
    target_dir = Path("d:/STUDY/MLIoT/project/model/stage1_vlm/output")
    target_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = target_dir / "lora_adapters"
    lora_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📡 [Kaggle Agent] Đang kiểm tra tiến trình kernel: {kernel_id}...")
    
    status_info = api.kernels_status(kernel_id)
    status = getattr(status_info, "status", str(status_info))
    if hasattr(status, "name"):
        status_str = status.name
    else:
        status_str = str(status).upper()
        
    print(f"📊 Trạng thái Kernel: {status_str}")
    
    if "COMPLETE" in status_str:
        print("🎉 [Kaggle Agent] HUẤN LUYỆN ĐÃ HOÀN TẤT THÀNH CÔNG 100% TRÊN GPU KAGGLE!")
        print("📦 Đang tải file trọng số LoRA về máy tính...")
        
        with api.build_kaggle_client() as kaggle:
            from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
            import requests
            req = ApiListKernelSessionOutputRequest()
            req.user_name = "lminhsang241"
            req.kernel_slug = "qwen2-vl-receipt-vqa-golden"
            response = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            
            print(f"📂 Tìm thấy {len(response.files or [])} tệp trong output:")
            zip_item = None
            for item in response.files or []:
                print(f"  - {item.file_name}")
                if "zip" in item.file_name:
                    zip_item = item
            
            if zip_item and zip_item.url:
                zip_path = target_dir / zip_item.file_name
                print(f"⬇️ Đang tải {zip_item.file_name}...")
                resp = requests.get(zip_item.url, stream=True)
                with open(zip_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                print(f"✅ Đã tải file zip về ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")
                
                print(f"📂 Đang giải nén vào {lora_dir}...")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(lora_dir)
                print(f"🎯 ĐÃ ĐỒNG BỘ TRỌNG SỐ LORA THÀNH CÔNG VÀO: {lora_dir}")
                print(f"📋 Các tệp trong lora_adapters: {list(p.name for p in lora_dir.glob('*'))}")
            else:
                # Tải tất cả file đơn lẻ nếu không có zip
                for item in response.files or []:
                    if item.url:
                        outfile = lora_dir / item.file_name
                        print(f"⬇️ Đang tải {item.file_name}...")
                        resp = requests.get(item.url)
                        with open(outfile, "wb") as f:
                            f.write(resp.content)
                print(f"🎯 ĐÃ TẢI CÁC TỆP THÀNH CÔNG VÀO: {lora_dir}")
    else:
        print(f"⏳ Kernel vẫn đang chạy hoặc ở trạng thái: {status_str}")

if __name__ == "__main__":
    monitor_kernel()

