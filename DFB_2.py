import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from skimage import io, color
from scipy.ndimage import rotate
from skimage.transform import resize

def create_low_pass_filter(length=17, cutoff=0.25):
    """
    Create a separable low-pass linear phase filter as mentioned in the paper.
    The paper mentions using a separable low-pass linear phase filter of length 17
    and cut-off frequency at π/4.
    """
    # Design a 1D FIR lowpass filter
    h = signal.firwin(length, cutoff)
    # Return the 1D filter for separable 2D filtering
    return h

def apply_separable_filter(image, filter_1d):
    """Apply a separable 2D filter to an image"""
    # Apply filter along rows
    filtered_rows = np.zeros_like(image, dtype=float)
    for i in range(image.shape[0]):
        filtered_rows[i, :] = signal.convolve(image[i, :], filter_1d, mode='same')
    
    # Apply filter along columns
    filtered_image = np.zeros_like(filtered_rows, dtype=float)
    for j in range(image.shape[1]):
        filtered_image[:, j] = signal.convolve(filtered_rows[:, j], filter_1d, mode='same')
    
    return filtered_image

def quincunx_downsample(image):
    """
    Quincunx downsampling Q0: giữ pixel tại (i+j)%2==0.
    Với ảnh H×W, sẽ chọn được H*(W/2) mẫu và reshape về (H, W//2).
    """
    H, W = image.shape
    mask = (np.indices((H, W)).sum(axis=0) % 2 == 0)
    selected = image[mask]               # 1D mảng độ dài H*(W/2)
    return selected.reshape((H, W // 2))

def quincunx_upsample(image, original_shape):
    """
    Quincunx upsampling Q0 ngược lại:
    Đưa mảng H×(W/2) lên (H, W), đặt giá trị vào (i+j)%2==0.
    """
    H, W = original_shape
    up = np.zeros((H, W), dtype=subband.dtype)
    # mỗi hàng i có W//2 giá trị, đặt vào j sao cho (i+j)%2==0
    for i in range(H):
        # chỉ số cột j: j % 2 == i % 2
        cols = np.arange(i % 2, W, 2)
        up[i, cols] = subband[i, :]
    return up

def create_directional_filter(h, angle):
    """Rotate separable filter to given angle (in degrees)"""
    H2D = np.outer(h, h)
    return rotate(H2D, angle, reshape=False)

def fan_filter_bank(image, h):
    filt_45 = create_directional_filter(h, 45)
    filt_135 = create_directional_filter(h, -45)

    subband0 = signal.convolve2d(image, filt_45, mode='same', boundary='symm')
    subband1 = signal.convolve2d(image, filt_135, mode='same', boundary='symm')

    return quincunx_downsample(subband0), quincunx_downsample(subband1)


def diamond_filter_bank(image, h):
    """
    Approximated 2-channel diamond filter bank using separable filters and
    quincunx downsampling. This mimics the prefiltering process in DTCWT.
    """
    # Construct separable filter
    H2D = np.outer(h, h)

    # Convolve with separable lowpass and bandpass filters
    # The following are approximations; DTCWT uses ladder structures
    filtered0 = signal.convolve2d(image, H2D, mode='same', boundary='symm')  # Lowpass approx
    filtered1 = signal.convolve2d(image, np.flipud(np.fliplr(H2D)), mode='same', boundary='symm')  # Highpass approx

    # Downsample
    subband0 = quincunx_downsample(filtered0)
    subband1 = quincunx_downsample(filtered1)

    return subband0, subband1

def directional_filter_bank(image, h, num_bands=8):
    """
    Approximate an 8-band directional filter bank using 3-level decomposition.
    - Level 1: Fan filters (2 subbands)
    - Level 2: Diamond filters (4 subbands)
    - Level 3: Diamond filters again (8 subbands)
    :param image: input 2D image
    :param h: 1D filter used to build separable directional filters
    :param num_bands: must be 8 for this approximation
    :return: list of 8 directional subbands
    """
    if num_bands != 8:
        raise ValueError("Only 8-band directional filter bank supported.")

    # Level 1 - Fan filter bank
    s0, s1 = fan_filter_bank(image, h)

    # Level 2 - Diamond decomposition on fan outputs
    s00, s01 = diamond_filter_bank(s0, h)
    s10, s11 = diamond_filter_bank(s1, h)

    # Level 3 - Diamond again on each of the 4 branches
    subbands = []
    for s in [s00, s01, s10, s11]:
        d0, d1 = diamond_filter_bank(s, h)
        subbands.append(d0)
        subbands.append(d1)

    return subbands

def soft_threshold(x, t):
    """
    Soft-thresholding function as used in wavelet denoising.
    Shrinks coefficients by threshold t.
    """
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)

def sure_threshold(coeffs, sigma):
    """
    Compute optimal threshold based on Stein’s Unbiased Risk Estimate (SURE).
    Often used for adaptive denoising in wavelet transform domain.
    """
    x = coeffs.ravel()
    n = len(x)
    absx = np.abs(x)
    sorted_x = np.sort(absx)

    risks = []
    for t in sorted_x:
        num_small = np.sum(absx <= t)
        risk = np.sum(np.minimum(absx**2, t**2)) + 2 * sigma**2 * num_small - n * sigma**2
        risks.append(risk)

    return sorted_x[np.argmin(risks)]

def visushrink_threshold(sigma, n):
    """
    Compute universal threshold using VisuShrink:
    T = sigma * sqrt(2 * log(n))
    """
    return sigma * np.sqrt(2 * np.log(n))


def estimate_noise_variance(subband):
    """
    Ước lượng phương sai nhiễu trong một subband bằng MAD (Median Absolute Deviation).
    """
    mad = np.median(np.abs(subband - np.median(subband)))
    sigma = mad / 0.6745  # Ước lượng sigma giả sử nhiễu Gauss trắng
    return sigma**2

def inverse_directional_filter_bank(subbands, original_shape):
    """
    Hàm khôi phục ảnh từ các subbands của DFB
    subbands: list các ảnh subband đã xử lý
    original_shape: kích thước ảnh gốc để resize đúng
    """
    reconstructed = np.zeros(original_shape)
    for sb in subbands:
        upsampled = resize_to_shape(sb, original_shape)
        reconstructed += upsampled
    return reconstructed / len(subbands)  # trung bình hóa để giữ cường độ hợp lý

def resize_to_shape(image, target_shape):
    return resize(image, target_shape, mode='reflect', anti_aliasing=True)

def dfb_denoise(image, noise_std):
    """
    Khử nhiễu ảnh sử dụng Directional Filter Bank (DFB).
    Bao gồm tách low/high-frequency, phân tích high bằng DFB,
    ngưỡng hóa từng subband, rồi tái cấu trúc lại ảnh.
    """
    # --- Step 1: Tách ảnh thành thành phần tần số thấp và cao ---
    lp_filter = create_low_pass_filter()  # Hàm lọc low-pass (ví dụ Gaussian)
    xl = apply_separable_filter(image, lp_filter)  # Thành phần tần số thấp
    xh = image - xl  # Thành phần tần số cao (residual)

    # --- Step 2: Phân tích directional filter bank ---
    subbands = directional_filter_bank(xh, lp_filter)  # giả sử phân tích được 8 subbands

    # --- Step 3: Ngưỡng hóa từng subband theo phương pháp bài báo ---
    denoised_subbands = []
    for subband in subbands:
        var_i = estimate_noise_variance(subband)
        threshold_i = 1.4 * np.sqrt(var_i)  # Hệ số 1.4 là do bài báo đề xuất
        denoised_subband = soft_threshold(subband, threshold_i)
        denoised_subbands.append(denoised_subband)

    # --- Step 4: Tái cấu trúc lại ảnh từ DFB ---
    reconstructed_xh = inverse_directional_filter_bank(denoised_subbands, xh.shape)


    # --- Step 5: Kết hợp lại với thành phần tần số thấp để khôi phục ảnh ---
    denoised_image = xl + reconstructed_xh
    return denoised_image


def add_gaussian_noise(image, std):
    """
    Thêm nhiễu Gaussian trắng với độ lệch chuẩn std.
    """
    noise = np.random.normal(0, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255)

def mse(original, denoised):
    """
    Mean Squared Error giữa ảnh gốc và ảnh khử nhiễu
    """
    return np.mean((original - denoised) ** 2)

def psnr(original, denoised):
    return np.mean((original - denoised) ** 2)
    # mse_val = np.mean((original - denoised) ** 2)
    # max_pixel = 255.0
    # return 10 * np.log10((max_pixel ** 2) / mse_val) if mse_val != 0 else float('inf')


def display_comparison(original, noisy, denoised, title="Image Denoising Comparison"):
    """
    Hiển thị ảnh gốc, ảnh có nhiễu và ảnh sau khử nhiễu, cùng với PSNR.
    Phù hợp với trình bày trong bài báo khoa học.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    images = [original, noisy, denoised]
    titles = [
        "Original",
        f"Noisy\nPSNR: {psnr(original, noisy):.2f} dB",
        f"Denoised\nPSNR: {psnr(original, denoised):.2f} dB"
    ]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_directional_subbands(subbands, normalize=True):
    """
    Hiển thị 8 subbands hướng của DFB theo bố cục 2x4, có thể dùng trong bài báo.
    Normalize để hình ảnh rõ ràng hơn nếu subbands có biên độ nhỏ.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    axes = axes.flatten()

    for i, subband in enumerate(subbands):
        if normalize:
            sub = subband - np.min(subband)
            sub = sub / np.max(sub) if np.max(sub) != 0 else sub
        else:
            sub = subband
        
        axes[i].imshow(sub, cmap='gray')
        axes[i].set_title(f'Subband {i+1}', fontsize=10)
        axes[i].axis('off')

    plt.suptitle('Directional Subbands (DFB Decomposition)', fontsize=14)
    plt.tight_layout()
    plt.show()

def load_and_prepare_image(image_path, size=512):
    """
    Load and preprocess the image: grayscale + resize + scale to 0–255.
    If loading fails, fall back to a synthetic test pattern.
    """
    try:
        img = io.imread(image_path)

        # Convert to grayscale if necessary
        if img.ndim == 3:
            img = color.rgb2gray(img)

        # Resize to fixed size
        img = resize(img, (size, size), anti_aliasing=True)

        # Convert to 0–255 range if in [0, 1]
        if img.max() <= 1.0:
            img = img * 255.0

        return img.astype(np.float64)

    except Exception as e:
        print(f"[Warning] Failed to load image: {e}")
        print("Using synthetic test pattern instead.")
        return create_test_pattern(size)

def create_test_pattern(size=512):
    """
    Generate a synthetic image with structures: checkerboard + edges + circle.
    """
    img = np.zeros((size, size), dtype=np.float64)

    # Checkerboard pattern
    block_size = size // 8
    for i in range(size):
        for j in range(size):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                img[i, j] = 200

    # Add diagonal cross
    for i in range(size):
        if 0 <= i < size:
            img[i, i] = 230
            img[i, size - i - 1] = 230

    # Add a circular ring at center
    center = size // 2
    radius = size // 4
    for i in range(size):
        for j in range(size):
            dist = (i - center) ** 2 + (j - center) ** 2
            if (radius - 3) ** 2 < dist < (radius + 3) ** 2:
                img[i, j] = 255

    return img

def dwt_ideal_denoise(image, sigma, wavelet='sym4', level=6):
    """
    Denoise image using 2D DWT with fixed (ideal) threshold.
    Only high-frequency coefficients are thresholded; the approximation (low-frequency) is kept intact.
    """
    # Perform 2D wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    cA, details = coeffs[0], coeffs[1:]
    # Threshold value (fixed sigma)
    thr = sigma
    # Apply soft-thresholding to each detail subband
    new_details = []
    for (cH, cV, cD) in details:
        cH_th = pywt.threshold(cH, thr, mode='soft')
        cV_th = pywt.threshold(cV, thr, mode='soft')
        cD_th = pywt.threshold(cD, thr, mode='soft')
        new_details.append((cH_th, cV_th, cD_th))
    # Reconstruct image from original approximation and thresholded details
    coeffs_thresh = [cA] + new_details
    denoised = pywt.waverec2(coeffs_thresh, wavelet)
    return denoised

def gcv_threshold(coefs):
    """
    Estimate threshold via Generalized Cross-Validation (GCV)/SURE.
    coefs should be a 1D numpy array of wavelet detail coefficients.
    """
    # Estimate noise level sigma using median absolute deviation
    sigma = np.median(np.abs(coefs)) / 0.6745
    coefs = np.sort(np.abs(coefs))
    n = len(coefs)
    if n == 0:
        return 0.0
    # Prefix sum of squares for efficient SURE computation
    prefix_sq = np.concatenate(([0.0], np.cumsum(coefs**2)))
    prefix_sq_up_to_i = prefix_sq[1:]  # sum of squares up to each index
    vals_sq = coefs**2
    # Number of coefficients larger than threshold at each index
    counts_large = n - np.arange(1, n+1)
    # Compute SURE for each potential threshold = coefs[i]
    # SURE(T) = sum(min(coef^2, T^2)) + 2*sigma^2 * (#coefs > T) - n*sigma^2
    sure = prefix_sq_up_to_i + vals_sq * counts_large + 2*(sigma**2) * counts_large - n*(sigma**2)
    # Find threshold that minimizes SURE
    idx_opt = np.argmin(sure)
    return coefs[idx_opt]

def dwt_gcv_denoise(image, wavelet='sym4', level=4):
    """
    Denoise image using 2D DWT with threshold selected by GCV (Generalized Cross-Validation).
    Only high-frequency coefficients are thresholded; the approximation is kept intact.
    """
    # Perform 2D wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    cA, details = coeffs[0], coeffs[1:]
    new_details = []
    # Apply GCV-based threshold to each detail subband
    for (cH, cV, cD) in details:
        # Combine subbands to compute threshold
        stacked = np.concatenate([cH.ravel(), cV.ravel(), cD.ravel()])
        thr = gcv_threshold(stacked)
        # Apply soft-thresholding with the found threshold
        cH_th = pywt.threshold(cH, thr, mode='soft')
        cV_th = pywt.threshold(cV, thr, mode='soft')
        cD_th = pywt.threshold(cD, thr, mode='soft')
        new_details.append((cH_th, cV_th, cD_th))
    # Reconstruct image from original approximation and thresholded details
    coeffs_thresh = [cA] + new_details
    denoised = pywt.waverec2(coeffs_thresh, wavelet)
    return denoised

def compare_methods(original, noisy, noise_std):
    """
    So sánh 3 phương pháp khử nhiễu:
        - DWT-ideal (ngưỡng = noise_std)
        - DWT-GCV   (ngưỡng ước lượng GCV)
        - DFB       (Direction Filter Bank)
    """
    # 1) DWT-ideal
    deno_ideal = dwt_ideal_denoise(noisy, noise_std)
    # 2) DWT-GCV
    deno_gcv   = dwt_gcv_denoise(noisy)
    # 3) DFB
    deno_dfb   = dfb_denoise(noisy, noise_std)
    
    # Đảm bảo cùng kích thước
    h, w = original.shape
    for img in (deno_ideal, deno_gcv, deno_dfb):
        if img.shape != original.shape:
            img[:] = img[:h, :w]
    
    # Vẽ so sánh
    titles = [
        f"Noisy\nPSNR={psnr(original, noisy):.2f} dB",
        f"DWT-Ideal\nPSNR={psnr(original, deno_ideal):.2f} dB",
        f"DWT-GCV\nPSNR={psnr(original, deno_gcv):.2f} dB",
        f"DFB\nPSNR={psnr(original, deno_dfb):.2f} dB"
    ]
    images = [noisy, deno_ideal, deno_gcv, deno_dfb]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, img, t in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(t)
        ax.axis('off')
    plt.suptitle("So sánh Denoising: DWT-Ideal vs DWT-GCV vs DFB")
    plt.tight_layout()
    plt.show()

    return deno_ideal, deno_gcv, deno_dfb

def main(image_path=None):
    np.random.seed(42)

    if image_path:
        original = load_and_prepare_image(image_path)
    else:
        print("No image provided, creating test pattern...")
        original = create_test_pattern()

    noise_levels = [10, 15, 20]
    for noise_std in noise_levels:
        print(f"\nNoise σ = {noise_std}")
        noisy = add_gaussian_noise(original, noise_std)

        # Compare all three methods
        print("Comparing methods...")
        deno_i, deno_g, deno_d = compare_methods(original, noisy, noise_std)

        if noise_std == 15:
            lp_filter = create_low_pass_filter()
            xl = apply_separable_filter(noisy, lp_filter)
            xh = noisy - xl
            print("Extracting directional subbands...")
            subbands = directional_filter_bank(xh, lp_filter)
            visualize_directional_subbands(subbands)

if __name__ == "__main__":
    # You can either provide an image path or let it create a test pattern
    # Example: main("path/to/your/image.jpg")
    # image_path = "./images/original.png"
    main("./images/vetinh.webp")
    
    # Alternatively, you can use this command to create a test pattern
    # main()
