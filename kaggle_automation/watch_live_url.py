import os
import sys
import time
import re

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["KAGGLE_API_TOKEN"] = "KGAT_543b77ae9398d7062e33f1934b10c69d"

from kaggle.api.kaggle_api_extended import KaggleApi

def watch():
    api = KaggleApi()
    api.authenticate()
    kernel_slug = "qwen2-vl-finetuned-test"
    kernel_id = f"lminhsang241/{kernel_slug}"
    
    for i in range(40):
        time.sleep(15)
        try:
            with api.build_kaggle_client() as kaggle:
                from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest, ApiGetKernelSessionStatusRequest
                st_req = ApiGetKernelSessionStatusRequest()
                st_req.user_name = "lminhsang241"
                st_req.kernel_slug = kernel_slug
                st_resp = kaggle.kernels.kernels_api_client.get_kernel_session_status(st_req)
                status = str(st_resp.status)
                
                req = ApiListKernelSessionOutputRequest()
                req.user_name = "lminhsang241"
                req.kernel_slug = kernel_slug
                resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
                if resp.log:
                    urls = re.findall(r"https://[a-zA-Z0-9-]+\.gradio\.live", resp.log)
                    if urls:
                        print("\n" + "=" * 80)
                        print("🎉 GRADIO PUBLIC URL ĐÃ SẴN SÀNG:")
                        print(urls[-1])
                        print("=" * 80)
                        return
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] Status: {status} (Đang nạp mô hình lên GPU...)")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Status: {status} (Đang chạy...)")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Đang chờ: {e}")

if __name__ == "__main__":
    watch()
