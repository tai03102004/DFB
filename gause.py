import cv2
import os
from PIL import Image

# Tạo thư mục nếu chưa tồn tại
output_dir = "./images"
os.makedirs(output_dir, exist_ok=True)

# Đọc ảnh gốc (ảnh xám)
img = cv2.imread('./images/lena_clean.png', cv2.IMREAD_GRAYSCALE)

# Resize về kích thước 3x4 inch ở 300 DPI => 900 x 1200 pixel
resized_img = cv2.resize(img, (900, 1200))

# Chuyển sang ảnh PIL và lưu với DPI
final_img = Image.fromarray(resized_img)
output_path = os.path.join(output_dir, "lena_3x4_300dpi.png")
final_img.save(output_path, dpi=(300, 300))

print(f"Ảnh đã được lưu tại: {output_path}")
