import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr, mean_squared_error as mse
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import ndimage
import pywt

def add_noise(image, noise_sigma=15):
    """Thêm nhiễu Gaussian vào ảnh"""
    noise = np.random.normal(0, noise_sigma/255, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image

# ============= DFB METHOD =============
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

def denoise_dfb(noisy_image, noise_sigma=15, orientations=8, filter_size=9):
    """Khử nhiễu bằng kỹ thuật Directional Filter Bank (DFB)"""
    
    # Chuyển đổi tensor
    noisy_tensor = torch.from_numpy(noisy_image).unsqueeze(0).unsqueeze(0).to(torch.float32)
    
    # Tạo bộ lọc định hướng
    dfb_filters, thetas = create_dfb_filters(orientations=orientations, size=filter_size)
    dfb_filters = dfb_filters.unsqueeze(1)
    
    # Áp dụng phân tách định hướng
    dir_coeffs = F.conv2d(noisy_tensor, dfb_filters, padding='same')
    
    # Ước lượng ngưỡng thích ứng theo mức độ nhiễu
    sigma_noise = noise_sigma/255
    thresholds = []
    
    for i in range(orientations):
        band = dir_coeffs[:, i].cpu().numpy()
        var_band = np.mean(band**2)
        sigma_band = max(np.sqrt(max(var_band - sigma_noise**2, 0)), 1e-6)
        t = sigma_noise**2 / sigma_band
        thresholds.append(t)
    
    thresholds = torch.tensor(thresholds, device=dir_coeffs.device).view(1, -1, 1, 1)
    
    # Áp dụng ngưỡng wavelet thích ứng theo hướng
    denoised_coeffs = wavelet_threshold(dir_coeffs, thresholds, mode='soft')
    
    # Tái tạo ảnh cải tiến (weighted reconstruction)
    energy = torch.sum(denoised_coeffs**2, dim=(0,2,3), keepdim=True)
    total_energy = torch.sum(energy)
    weights = energy / (total_energy + 1e-10)
    
    # Tái tạo có trọng số
    recon_filters = torch.flip(dfb_filters, [2, 3])
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
    original_tensor = torch.from_numpy(noisy_image).unsqueeze(0).unsqueeze(0)
    alpha = 0.28  # Tham số điều khiển mức độ khử nhiễu
    denoised_tensor = alpha * denoised_tensor + (1-alpha) * original_tensor
    
    # Chuyển về numpy
    denoised_image = denoised_tensor.squeeze().cpu().numpy()
    denoised_image = np.clip(denoised_image, 0, 1)
    
    # Áp dụng lọc trung vị để làm mịn kết quả cuối cùng
    denoised_image = ndimage.median_filter(denoised_image, size=3)
    
    return denoised_image

# ============= DWT-IDEAL METHOD =============
def denoise_dwt_ideal(noisy_image, noise_sigma=15, wavelet='db4', level=2):
    """
    Khử nhiễu bằng DWT với ngưỡng lý tưởng
    Ngưỡng lý tưởng là ngưỡng biết trước nhiễu sigma
    """
    # DWT transform
    coeffs = pywt.wavedec2(noisy_image, wavelet, level=level)
    # Ước lượng ngưỡng theo mức nhiễu đã biết
    sigma = noise_sigma/255.0
    threshold = sigma * np.sqrt(2 * np.log(noisy_image.size)) * 1.12  # Hệ số điều chỉnh 1.12
    
    # Áp dụng ngưỡng cho các chi tiết wavelet (giữ nguyên approximation)
    new_coeffs = [coeffs[0]]  # Keep approximation coefficients
    
    for i in range(1, len(coeffs)):
        # Áp dụng ngưỡng cho các chi tiết theo hướng (H, V, D)
        details = []
        for detail in coeffs[i]:
            # Soft thresholding
            thresholded = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
            details.append(thresholded)
        new_coeffs.append(tuple(details))
    
    # Tái tạo ảnh từ các hệ số đã lọc
    denoised_image = pywt.waverec2(new_coeffs, wavelet)
    
    # Cắt về kích thước ban đầu và giới hạn phạm vi [0,1]
    denoised_image = denoised_image[:noisy_image.shape[0], :noisy_image.shape[1]]
    denoised_image = np.clip(denoised_image, 0, 1)
    
    return denoised_image

# ============= DWT-GCV METHOD =============
def gcv_function(threshold, wavelet_coeffs, n):
    """Hàm GCV (Generalized Cross-Validation) để tìm ngưỡng tối ưu"""
    # Số lượng hệ số nhỏ hơn ngưỡng
    m = np.sum(np.abs(wavelet_coeffs) <= threshold)
    
    # Áp dụng ngưỡng mềm để ước lượng sai số
    thresholded = np.sign(wavelet_coeffs) * np.maximum(np.abs(wavelet_coeffs) - threshold, 0)
    residual = np.sum((wavelet_coeffs - thresholded)**2)
    
    # Công thức GCV
    if m == n:  # Tránh chia cho 0
        return float('inf')
    
    gcv_score = residual / (n * (1 - m/n)**2)
    return gcv_score

def find_optimal_threshold_gcv(wavelet_coeffs):
    """Tìm ngưỡng tối ưu bằng phương pháp GCV"""
    n = wavelet_coeffs.size
    
    # Các giá trị ngưỡng ứng viên (dựa trên phân bố của các hệ số)
    max_coeff = np.max(np.abs(wavelet_coeffs))
    thresholds = np.linspace(0, max_coeff/2, 30)  # 30 giá trị ngưỡng khác nhau
    
    # Tính GCV cho mỗi ngưỡng
    gcv_scores = [gcv_function(t, wavelet_coeffs, n) for t in thresholds]
    
    # Ngưỡng tối ưu là ngưỡng có GCV nhỏ nhất
    optimal_idx = np.argmin(gcv_scores)
    return thresholds[optimal_idx]

def denoise_dwt_gcv(noisy_image, wavelet='sym8', level=2):
    """
    Khử nhiễu bằng DWT với ngưỡng tự động GCV
    GCV tự động tìm ngưỡng tối ưu mà không cần biết trước mức nhiễu
    """
    # DWT transform
    coeffs = pywt.wavedec2(noisy_image, wavelet, level=level)
    
    # Giữ nguyên approximation
    new_coeffs = [coeffs[0]]
    
    # Áp dụng GCV cho mỗi level và hướng detail
    for i in range(1, len(coeffs)):
        details = []
        for detail in coeffs[i]:
            # Tìm ngưỡng tối ưu cho mỗi band
            optimal_threshold = find_optimal_threshold_gcv(detail.flatten())
            
            # Soft thresholding với ngưỡng tối ưu
            thresholded = np.sign(detail) * np.maximum(np.abs(detail) - optimal_threshold, 0)
            details.append(thresholded)
        new_coeffs.append(tuple(details))
    
    # Tái tạo ảnh từ các hệ số đã lọc
    denoised_image = pywt.waverec2(new_coeffs, wavelet)
    
    # Cắt về kích thước ban đầu và giới hạn phạm vi [0,1]
    denoised_image = denoised_image[:noisy_image.shape[0], :noisy_image.shape[1]]
    denoised_image = np.clip(denoised_image, 0, 1)
    
    return denoised_image

# ============= MAIN EVALUATION FUNCTION =============
def compare_denoising_methods(image_path, noise_sigma=15):
    """So sánh các phương pháp khử nhiễu khác nhau"""
    print(f"\n[INFO] Đang xử lý ảnh: {image_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"ERROR: Không tìm thấy file ảnh tại {image_path}")
    
    # Đọc ảnh
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("ERROR: Không đọc được ảnh (kiểm tra định dạng file!)")
    
    # Chuẩn hóa về [0,1]
    image = image.astype(np.float32)/255.0
    
    # Thêm nhiễu
    noisy_image = add_noise(image, noise_sigma)
    
    print(f"[INFO] Đang áp dụng các phương pháp khử nhiễu (sigma={noise_sigma})...")
    
    # Áp dụng các phương pháp khử nhiễu
    denoised_dfb = denoise_dfb(noisy_image, noise_sigma, orientations=16, filter_size=11)
    denoised_dwt_ideal = denoise_dwt_ideal(noisy_image, noise_sigma, wavelet='db4', level=3)
    denoised_dwt_gcv = denoise_dwt_gcv(noisy_image, wavelet='db4', level=3)
    
    # Tính các độ đo đánh giá
    results = {
        'Method': ['Noisy Image', 'DWT-Ideal', 'DWT-GCV', 'DFB'],
        'MSE': [
            mse(image, noisy_image),
            mse(image, denoised_dwt_ideal),
            mse(image, denoised_dwt_gcv),
            mse(image, denoised_dfb)
        ],
        'PSNR': [
            psnr(image, noisy_image, data_range=1),
            psnr(image, denoised_dwt_ideal, data_range=1),
            psnr(image, denoised_dwt_gcv, data_range=1),
            psnr(image, denoised_dfb, data_range=1)
        ]
    }
    
    # Hiển thị bảng kết quả
    df = pd.DataFrame(results)
    print("\n===== KẾT QUẢ SO SÁNH =====")
    print(df.to_string(index=False))
    
    # Chuẩn bị hiển thị kết quả trực quan
    original_display = (image * 255).astype(np.uint8)
    noisy_display = (noisy_image * 255).astype(np.uint8)
    denoised_dfb_display = (denoised_dfb * 255).astype(np.uint8)
    denoised_dwt_ideal_display = (denoised_dwt_ideal * 255).astype(np.uint8)
    denoised_dwt_gcv_display = (denoised_dwt_gcv * 255).astype(np.uint8)
    
    # Hiển thị kết quả
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 3, 1)
    plt.imshow(original_display, cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(noisy_display, cmap='gray')
    plt.title(f"Noisy (σ={noise_sigma})\nMSE: {results['MSE'][0]:.4f}, PSNR: {results['PSNR'][0]:.2f} dB")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(denoised_dwt_ideal_display, cmap='gray')
    plt.title(f"DWT-Ideal\nMSE: {results['MSE'][1]:.4f}, PSNR: {results['PSNR'][1]:.2f} dB")
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(denoised_dwt_gcv_display, cmap='gray')
    plt.title(f"DWT-GCV\nMSE: {results['MSE'][2]:.4f}, PSNR: {results['PSNR'][2]:.2f} dB")
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(denoised_dfb_display, cmap='gray')
    plt.title(f"DFB\nMSE: {results['MSE'][3]:.4f}, PSNR: {results['PSNR'][3]:.2f} dB")
    plt.axis('off')
    
    # Thêm bảng kết quả vào hình
    plt.subplot(2, 3, 6)
    plt.axis('off')
    table_data = [[method, f"{mse_val:.4f}", f"{psnr_val:.2f}"] 
                  for method, mse_val, psnr_val in zip(results['Method'], results['MSE'], results['PSNR'])]
    plt.table(cellText=table_data, colLabels=['Method', 'MSE', 'PSNR (dB)'], 
              loc='center', cellLoc='center')
    plt.title("Comparison Results")
    
    plt.tight_layout()
    
    # Lưu kết quả
    output_dir = os.path.dirname(image_path)
    output_path = os.path.join(output_dir, f"denoising_comparison_sigma{noise_sigma}.png")
    plt.savefig(output_path, dpi=300)
    print(f"[INFO] Đã lưu kết quả so sánh tại: {output_path}")
    
    plt.show()
    
    return {
        'noisy': noisy_image,
        'dfb': denoised_dfb,
        'dwt_ideal': denoised_dwt_ideal,
        'dwt_gcv': denoised_dwt_gcv,
        'metrics': df
    }

# ============= MAIN EXECUTION =============
if __name__ == "__main__":
    try:
        # Thay đổi đường dẫn chính xác đến ảnh của bạn
        image_path = "./images/original.png"
        
        # Chạy với các mức nhiễu khác nhau
        noise_levels = [10, 15, 25]
        results = {}
        
        for sigma in noise_levels:
            print(f"\n==== TESTING NOISE LEVEL σ={sigma} ====")
            results[sigma] = compare_denoising_methods(image_path, noise_sigma=sigma)
            
        # So sánh tất cả mức nhiễu
        methods = ['Noisy Image', 'DWT-Ideal', 'DWT-GCV', 'DFB']
        mse_data = pd.DataFrame({
            'Method': methods,
            'σ=10': [results[10]['metrics']['MSE'][i] for i in range(4)],
            'σ=15': [results[15]['metrics']['MSE'][i] for i in range(4)],
            'σ=25': [results[25]['metrics']['MSE'][i] for i in range(4)]
        })
        
        print("\n===== BẢNG SO SÁNH MSE THEO MỨC NHIỄU =====")
        print(mse_data.to_string(index=False))
        
        # Vẽ biểu đồ so sánh MSE
        plt.figure(figsize=(10, 6))
        x = np.arange(len(methods))
        width = 0.25
        
        plt.bar(x - width, mse_data['σ=10'], width, label='σ=10')
        plt.bar(x, mse_data['σ=15'], width, label='σ=15')
        plt.bar(x + width, mse_data['σ=25'], width, label='σ=25')
        
        plt.xlabel('Phương pháp')
        plt.ylabel('MSE (Mean Squared Error)')
        plt.title('So sánh MSE của các phương pháp khử nhiễu theo mức nhiễu')
        plt.xticks(x, methods, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        output_bar_path = os.path.join(os.path.dirname(image_path), "mse_comparison_chart.png")
        plt.savefig(output_bar_path, dpi=300)
        print(f"[INFO] Đã lưu biểu đồ so sánh MSE tại: {output_bar_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"\n[ERROR] Đã xảy ra lỗi: {str(e)}")
        print("[DEBUG] Hướng dẫn khắc phục:")
        print("1. Kiểm tra đường dẫn ảnh đầu vào")
        print("2. Đảm bảo ảnh có định dạng .jpg/.png hợp lệ")
        print("3. Kiểm tra đã cài đặt đầy đủ thư viện (numpy, torch, pywt, pandas, etc.)")
        print("4. Thử giảm giá trị noise_sigma nếu ảnh đã có nhiễu")