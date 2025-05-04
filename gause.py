import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh gốc (ảnh xám hoặc RGB)
img = cv2.imread('./images/Gause.jpg', cv2.IMREAD_GRAYSCALE)  # hoặc cv2.IMREAD_COLOR nếu ảnh màu
img = img.astype(np.float32) / 255.0  # chuẩn hóa về [0, 1]

# Thêm nhiễu Gaussian trắng
mean = 0
std_dev = 0.05  # độ lệch chuẩn của nhiễu, bạn có thể điều chỉnh
gaussian_noise = np.random.normal(mean, std_dev, img.shape)

noisy_img = img + gaussian_noise
noisy_img = np.clip(noisy_img, 0, 1)  # giới hạn lại giá trị pixel sau khi thêm nhiễu

# Hiển thị ảnh
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img, cmap='gray')

plt.subplot(1,2,2)
plt.title("AWGN")
plt.imshow(noisy_img, cmap='gray')

plt.show()
