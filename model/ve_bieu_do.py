import matplotlib.pyplot as plt
import numpy as np

# --- 1. DỮ LIỆU TỪ BẢNG BENCHMARK ---
# Tên 2 phiên bản mô hình
models = ['Qwen2-VL-2B\n(Base Zero-Shot)', 'Qwen2-VL-2B + QLoRA\n(Bản Fine-tune của nhóm)']

# Điểm số quy đổi ra phần trăm (%)
anls_scores = [64.12, 93.45] 
em_scores = [48.00, 88.50]

# --- 2. THIẾT LẬP THÔNG SỐ BIỂU ĐỒ ---
x = np.arange(len(models))  # Vị trí các nhóm cột
width = 0.35  # Độ rộng của từng cột

fig, ax = plt.subplots(figsize=(10, 6))

# Vẽ 2 nhóm cột (ANLS màu xanh dương, EM màu cam nhạt)
rects1 = ax.bar(x - width/2, anls_scores, width, label='ANLS Score (%)', color='#4C72B0', edgecolor='black')
rects2 = ax.bar(x + width/2, em_scores, width, label='Exact Match - EM (%)', color='#DD8452', edgecolor='black')

# --- 3. TRANG TRÍ & HIỂN THỊ THÔNG TIN ---
ax.set_ylabel('Điểm số (%)', fontsize=12, fontweight='bold')
ax.set_title('So sánh Hiệu năng Mô hình KIE trước và sau khi Fine-tune', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, fontweight='bold')
ax.set_ylim(0, 115) # Kéo cao trục Y lên một chút để có chỗ ghi số
ax.legend(loc='upper left', fontsize=11)

# Hàm tự động ghi con số phần trăm lên đầu mỗi cột
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # Dịch lên 5 points so với đỉnh cột
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# Thêm lưới kẻ ngang cho dễ nhìn
ax.grid(axis='y', linestyle='--', alpha=0.7)

# --- 4. LƯU & XUẤT ẢNH ---
plt.tight_layout()
output_filename = 'benchmark_comparison_chart.png'
plt.savefig(output_filename, dpi=300) # dpi=300 giúp ảnh cực kỳ sắc nét khi in PDF
print(f"🎉 Đã lưu biểu đồ thành công: {output_filename}")

plt.show()