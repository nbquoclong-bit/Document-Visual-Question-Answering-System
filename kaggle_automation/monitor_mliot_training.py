import os
import sys
import json
import time
import zipfile
import requests
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["KAGGLE_API_TOKEN"] = "KGAT_543b77ae9398d7062e33f1934b10c69d"

from kaggle.api.kaggle_api_extended import KaggleApi

def monitor_mliot_project_training():
    api = KaggleApi()
    api.authenticate()
    
    kernel_slug = "mliot-project-cu-i-k"
    kernel_id = f"lminhsang241/{kernel_slug}"
    
    target_output_dir = Path("model/output")
    target_adapter_dir = Path("model/stage1_vlm/output/lora_adapters")
    target_output_dir.mkdir(parents=True, exist_ok=True)
    target_adapter_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 85)
    print(f"⏳ ĐANG THEO DÕI TIẾN TRÌNH HUẤN LUYỆN TRÊN KAGGLE GPU: {kernel_id}")
    print(f"🔗 Link notebook: https://www.kaggle.com/code/{kernel_id}")
    print("=" * 85)
    
    for step in range(120):
        time.sleep(30)
        try:
            status_res = api.kernels_status(kernel_id)
            status = status_res.get("status", "unknown").upper()
            print(f"[{time.strftime('%H:%M:%S')}] Trạng thái Kernel: {status}")
            
            if status == "COMPLETE":
                print("\n🎉 TIẾN TRÌNH HUẤN LUYỆN & ĐÁNH GIÁ ĐÃ HOÀN TẤT THÀNH CÔNG TRÊN GPU TESLA T4!")
                print("📥 Đang tải các tệp artifact kết quả về hệ thống...")
                
                with api.build_kaggle_client() as kaggle:
                    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
                    req = ApiListKernelSessionOutputRequest()
                    req.user_name = "lminhsang241"
                    req.kernel_slug = kernel_slug
                    resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
                    
                    for item in (resp.files or []):
                        fname = item.file_name
                        r = requests.get(item.url)
                        if fname == "evaluation_report.json":
                            save_p = target_output_dir / "evaluation_report.json"
                            with open(save_p, "wb") as f:
                                f.write(r.content)
                            print(f"✅ Đã tải về báo cáo đánh giá LoRA mới: {save_p}")
                        elif fname.endswith(".zip") and "lora" in fname.lower():
                            zip_p = target_output_dir / fname
                            with open(zip_p, "wb") as f:
                                f.write(r.content)
                            print(f"✅ Đã tải về tệp nén LoRA: {zip_p}")
                            try:
                                with zipfile.ZipFile(zip_p, 'r') as zf:
                                    zf.extractall(target_adapter_dir.parent)
                                print(f"✅ Đã giải nén LoRA Adapter vào: {target_adapter_dir}")
                            except Exception as e:
                                print(f"   Lỗi giải nén: {e}")
                print("\n🎯 TOÀN BỘ TIẾN TRÌNH ĐỒNG BỘ ĐÃ HOÀN TẤT!")
                break
            elif status == "ERROR":
                print(f"❌ Kernel gặp lỗi. Vui lòng xem log chi tiết tại: https://www.kaggle.com/code/{kernel_id}")
                break
        except Exception as exc:
            print(f"[{time.strftime('%H:%M:%S')}] Đang chờ: {exc}")

if __name__ == "__main__":
    monitor_mliot_project_training()
