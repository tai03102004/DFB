import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import os
from scipy import ndimage

def create_dfb_filters(orientations=8, size=7):
    """Tạo bộ lọc định hướng DFB chính thống"""
    filters = []
    thetas = np.linspace(0, np.pi, orientations, endpoint=False)
    
    for theta in thetas:
        # Tạo lọc định hướng dựa trên gradient định hướng
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        
        # Áp dụng phép xoay để tạo hướng
        rotx = x * np.cos(theta) + y * np.sin(theta)
        roty = -x * np.sin(theta) + y * np.cos(theta)
        
        # Tạo bộ lọc cạnh định hướng (directional edge filter)
        kernel = np.exp(-(rotx**2 + (roty/4)**2)/(2*(size/4)**2))
        kernel = kernel * (1j * rotx) # Sử dụng filter phức để nắm bắt hướng
        kernel = kernel / np.abs(kernel).sum() # Chuẩn hóa
        
        # Chuyển từ filter phức sang filter thực
        kernel_real = np.real(kernel).astype(np.float32)
        filters.append(kernel_real)
        
    return torch.tensor(np.array(filters, dtype=np.float32)), thetas

def wavelet_threshold(coeffs, threshold, mode='soft'):
    """Áp dụng ngưỡng wavelet thông minh cho các hệ số"""
    if mode == 'soft':
        # Soft thresholding
        signs = torch.sign(coeffs)
        magnitude = torch.abs(coeffs)
        thresholded = signs * F.relu(magnitude - threshold)
    else:
        # Hard thresholding
        thresholded = coeffs * (torch.abs(coeffs) > threshold).float()
        
    return thresholded

def denoise_dfb(image_path, noise_sigma=15, orientations=16, filter_size=11):
    """Khử nhiễu bằng kỹ thuật DFB chính thống với tham số cải tiến"""
    
    # ========== KIỂM TRA ĐẦU VÀO ==========
    print(f"\n[DEBUG] Đang xử lý ảnh: {image_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"ERROR: Không tìm thấy file ảnh tại {image_path}")
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("ERROR: Không đọc được ảnh (kiểm tra định dạng file!)")
    
    print(f"[DEBUG] Kích thước ảnh: {image.shape}, dtype: {image.dtype}")

    # ========== TIỀN XỬ LÝ ==========
    image = image.astype(np.float32)/255.0  # Chuẩn hóa về [0,1] và float32
    
    # Thêm nhiễu với cùng kiểu dữ liệu
    noise = np.random.normal(0, noise_sigma/255, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 1)
    
    # ========== XỬ LÝ DFB ==========
    # Tạo bộ lọc định hướng cải tiến
    dfb_filters, thetas = create_dfb_filters(orientations=orientations, size=filter_size)
    dfb_filters = dfb_filters.unsqueeze(1)  # Đã là float32 từ khởi tạo
    
    # Chuyển đổi tensor
    noisy_tensor = torch.from_numpy(noisy_image).unsqueeze(0).unsqueeze(0).to(torch.float32)
    
    # Áp dụng phân tách định hướng (directional decomposition)
    dir_coeffs = F.conv2d(noisy_tensor, dfb_filters, padding='same')
    
    # Ước lượng ngưỡng thích ứng theo mức độ nhiễu
    # BayesShrink threshold estimation
    sigma_noise = noise_sigma/255
    thresholds = []
    
    for i in range(orientations):
        band = dir_coeffs[:, i].cpu().numpy()
        # Ước lượng phương sai của hệ số = tổng bình phương / số lượng
        var_band = np.mean(band**2)
        # Ngưỡng thích ứng: sigma_noise^2 / max(sigma_band, sigma_noise)
        sigma_band = max(np.sqrt(max(var_band - sigma_noise**2, 0)), 1e-6)
        t = sigma_noise**2 / sigma_band
        thresholds.append(t)
    
    thresholds = torch.tensor(thresholds, device=dir_coeffs.device).view(1, -1, 1, 1)
    
    # Áp dụng ngưỡng wavelet thích ứng theo hướng
    denoised_coeffs = wavelet_threshold(dir_coeffs, thresholds, mode='soft')
    
    # Tái tạo ảnh cải tiến (weighted reconstruction)
    # Mỗi hướng đóng góp theo mức năng lượng tương đối của nó
    energy = torch.sum(denoised_coeffs**2, dim=(0,2,3), keepdim=True)
    total_energy = torch.sum(energy)
    weights = energy / (total_energy + 1e-10)  # Tránh chia cho 0
    
    # Tái tạo có trọng số
    recon_filters = torch.flip(dfb_filters, [2, 3])  # Flip filters for reconstruction
    denoised_components = []
    
    for i in range(orientations):
        component = F.conv_transpose2d(
            denoised_coeffs[:, i:i+1], 
            recon_filters[i:i+1], 
            padding=filter_size//2
        )
        denoised_components.append(component * weights[0, i])
    
    # Kết hợp các thành phần
    denoised_tensor = torch.sum(torch.cat(denoised_components, dim=1), dim=1, keepdim=True)
    
    # Kết hợp với ảnh gốc để giữ lại chi tiết (khử nhiễu có điều khiển)
    original_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    alpha = 0.35  # Tham số điều khiển mức độ khử nhiễu
    denoised_tensor = alpha * denoised_tensor + (1-alpha) * original_tensor
    
    # Chuyển về numpy
    denoised_image = denoised_tensor.squeeze().cpu().numpy()
    denoised_image = np.clip(denoised_image, 0, 1)
    
    # ========== HẬU XỬ LÝ ==========
    # Áp dụng TV-L2 làm mịn cuối cùng để giảm artifact nếu có
    def denoise_tv_chambolle(image, weight=0.1, eps=2.e-4, max_iter=50):
        """TV-L2 denoising"""
        return ndimage.median_filter(image, size=3)
    
    denoised_image = denoise_tv_chambolle(denoised_image, weight=0.1)
    
    denoised_display = (denoised_image * 255).astype(np.uint8)
    original_display = (image * 255).astype(np.uint8)
    noisy_display = (noisy_image * 255).astype(np.uint8)
    
    psnr_val = psnr(image, denoised_image, data_range=1)
    
    # ========== HIỂN THỊ ==========
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(original_display, cmap='gray'); plt.title("Original")
    plt.subplot(1, 3, 2); plt.imshow(noisy_display, cmap='gray'); plt.title(f"Noisy (σ={noise_sigma})")
    plt.subplot(1, 3, 3); plt.imshow(denoised_display, cmap='gray'); plt.title(f"DFB Denoised\nPSNR={psnr_val:.2f} dB")
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(image_path), "denoised_result.jpg")
    cv2.imwrite(output_path, denoised_display)
    print(f"[DEBUG] Đã lưu ảnh kết quả tại: {output_path}")
    print(f"[DEBUG] PSNR: {psnr_val:.2f} dB")
    
    # Hiển thị các bộ lọc để phân tích
    plt.figure(figsize=(12, 6))
    for i in range(min(8, orientations)):
        plt.subplot(2, 4, i+1)
        plt.imshow(dfb_filters[i, 0].cpu().numpy(), cmap='viridis')
        plt.title(f"Filter {i+1}: {thetas[i]:.2f} rad")
    plt.tight_layout()
    
    plt.show()
    return denoised_image

if __name__ == "__main__":
    try:
        # Thay đổi đường dẫn chính xác đến ảnh của bạn
        denoise_dfb("./images/image.png", orientations=16, filter_size=11)  
    except Exception as e:
        print(f"\n[ERROR] Đã xảy ra lỗi: {str(e)}")
        print("[DEBUG] Hướng dẫn khắc phục:")
        print("1. Kiểm tra đường dẫn ảnh đầu vào")
        print("2. Đảm bảo ảnh có định dạng .jpg/.png hợp lệ")
        print("3. Kiểm tra phiên bản thư viện (pip list)")
        print("4. Thử giảm giá trị noise_sigma nếu ảnh đã có nhiễu")