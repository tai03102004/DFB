import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from skimage import io, color
from scipy.ndimage import rotate
from skimage.transform import resize

def create_low_pass_filter(length=17, cutoff=0.25):
    h = signal.firwin(length, cutoff)
    return h

def apply_separable_filter(image, filter_1d):
    filtered_rows = np.zeros_like(image, dtype=float)
    for i in range(image.shape[0]):
        filtered_rows[i, :] = signal.convolve(image[i, :], filter_1d, mode='same')

    filtered_image = np.zeros_like(filtered_rows, dtype=float)
    for j in range(image.shape[1]):
        filtered_image[:, j] = signal.convolve(filtered_rows[:, j], filter_1d, mode='same')
    
    return filtered_image

def quincunx_downsample(image):
    H, W = image.shape
    mask = (np.indices((H, W)).sum(axis=0) % 2 == 0)
    selected = image[mask]
    return selected.reshape((H, W // 2))

def quincunx_upsample(image, original_shape):
    H, W = original_shape
    up = np.zeros((H, W), dtype=subband.dtype)
    for i in range(H):
        cols = np.arange(i % 2, W, 2)
        up[i, cols] = subband[i, :]
    return up

def create_directional_filter(h, angle):
    H2D = np.outer(h, h)
    return rotate(H2D, angle, reshape=False)

def fan_filter_bank(image, h):
    filt_45 = create_directional_filter(h, 45)
    filt_135 = create_directional_filter(h, -45)

    subband0 = signal.convolve2d(image, filt_45, mode='same', boundary='symm')
    subband1 = signal.convolve2d(image, filt_135, mode='same', boundary='symm')

    return quincunx_downsample(subband0), quincunx_downsample(subband1)


def diamond_filter_bank(image, h):
    H2D = np.outer(h, h)

    filtered0 = signal.convolve2d(image, H2D, mode='same', boundary='symm')
    filtered1 = signal.convolve2d(image, np.flipud(np.fliplr(H2D)), mode='same', boundary='symm')

    subband0 = quincunx_downsample(filtered0)
    subband1 = quincunx_downsample(filtered1)

    return subband0, subband1

def directional_filter_bank(image, h, num_bands=8):
    if num_bands != 8:
        raise ValueError("Only 8-band directional filter bank supported.")

    s0, s1 = fan_filter_bank(image, h)

    s00, s01 = diamond_filter_bank(s0, h)
    s10, s11 = diamond_filter_bank(s1, h)

    subbands = []
    for s in [s00, s01, s10, s11]:
        d0, d1 = diamond_filter_bank(s, h)
        subbands.append(d0)
        subbands.append(d1)

    return subbands

def soft_threshold(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)

def sure_threshold(coeffs, sigma):
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
    return sigma * np.sqrt(2 * np.log(n))


def estimate_noise_variance(subband):
    mad = np.median(np.abs(subband - np.median(subband)))
    sigma = mad / 0.6745
    return sigma**2

def inverse_directional_filter_bank(subbands, original_shape):
    reconstructed = np.zeros(original_shape)
    for sb in subbands:
        upsampled = resize_to_shape(sb, original_shape)
        reconstructed += upsampled
    return reconstructed / len(subbands)

def resize_to_shape(image, target_shape):
    return resize(image, target_shape, mode='reflect', anti_aliasing=True)

def dfb_denoise(image, noise_std):
    lp_filter = create_low_pass_filter()
    xl = apply_separable_filter(image, lp_filter)
    xh = image - xl

    subbands = directional_filter_bank(xh, lp_filter)

    denoised_subbands = []
    for subband in subbands:
        var_i = estimate_noise_variance(subband)
        threshold_i = 1.4 * np.sqrt(var_i)
        denoised_subband = soft_threshold(subband, threshold_i)
        denoised_subbands.append(denoised_subband)

    reconstructed_xh = inverse_directional_filter_bank(denoised_subbands, xh.shape)

    denoised_image = xl + reconstructed_xh
    return denoised_image


def add_gaussian_noise(image, std):
    noise = np.random.normal(0, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255)

def mse(original, denoised):
    return np.mean((original - denoised) ** 2)

def display_comparison(original, noisy, denoised, title="Image Denoising Comparison"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    images = [original, noisy, denoised]
    titles = [
        "Original",
        f"Noisy\nMSE: {mse(original, noisy):.2f} dB",
        f"Denoised\nMSE: {mse(original, denoised):.2f} dB"
    ]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_directional_subbands(subbands, normalize=True):
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
    try:
        img = io.imread(image_path)

        if img.ndim == 3:
            img = color.rgb2gray(img)

        img = resize(img, (size, size), anti_aliasing=True)

        if img.max() <= 1.0:
            img = img * 255.0

        return img.astype(np.float64)

    except Exception as e:
        print(f"[Warning] Failed to load image: {e}")
        print("Using synthetic test pattern instead.")
        return create_test_pattern(size)

def create_test_pattern(size=512):
    img = np.zeros((size, size), dtype=np.float64)

    block_size = size // 8
    for i in range(size):
        for j in range(size):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                img[i, j] = 200

    for i in range(size):
        if 0 <= i < size:
            img[i, i] = 230
            img[i, size - i - 1] = 230

    center = size // 2
    radius = size // 4
    for i in range(size):
        for j in range(size):
            dist = (i - center) ** 2 + (j - center) ** 2
            if (radius - 3) ** 2 < dist < (radius + 3) ** 2:
                img[i, j] = 255

    return img

def dwt_ideal_denoise(image, sigma, wavelet='sym4', level=6):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    cA, details = coeffs[0], coeffs[1:]
    thr = sigma
    new_details = []
    for (cH, cV, cD) in details:
        cH_th = pywt.threshold(cH, thr, mode='soft')
        cV_th = pywt.threshold(cV, thr, mode='soft')
        cD_th = pywt.threshold(cD, thr, mode='soft')
        new_details.append((cH_th, cV_th, cD_th))
    coeffs_thresh = [cA] + new_details
    denoised = pywt.waverec2(coeffs_thresh, wavelet)
    return denoised

def gcv_threshold(coefs):
    sigma = np.median(np.abs(coefs)) / 0.6745
    coefs = np.sort(np.abs(coefs))
    n = len(coefs)
    if n == 0:
        return 0.0
    
    prefix_sq = np.concatenate(([0.0], np.cumsum(coefs**2)))
    prefix_sq_up_to_i = prefix_sq[1:]
    vals_sq = coefs**2
    
    counts_large = n - np.arange(1, n+1)
    sure = prefix_sq_up_to_i + vals_sq * counts_large + 2*(sigma**2) * counts_large - n*(sigma**2)
    
    idx_opt = np.argmin(sure)
    return coefs[idx_opt]

def dwt_gcv_denoise(image, wavelet='sym4', level=4):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    cA, details = coeffs[0], coeffs[1:]
    new_details = []
    
    for (cH, cV, cD) in details:
        stacked = np.concatenate([cH.ravel(), cV.ravel(), cD.ravel()])
        thr = gcv_threshold(stacked)
        cH_th = pywt.threshold(cH, thr, mode='soft')
        cV_th = pywt.threshold(cV, thr, mode='soft')
        cD_th = pywt.threshold(cD, thr, mode='soft')
        new_details.append((cH_th, cV_th, cD_th))
    coeffs_thresh = [cA] + new_details
    denoised = pywt.waverec2(coeffs_thresh, wavelet)
    return denoised

def compare_methods(original, noisy, noise_std):
    deno_ideal = dwt_ideal_denoise(noisy, noise_std)
    deno_gcv   = dwt_gcv_denoise(noisy)
    deno_dfb   = dfb_denoise(noisy, noise_std)

    h, w = original.shape
    for img in (deno_ideal, deno_gcv, deno_dfb):
        if img.shape != original.shape:
            img[:] = img[:h, :w]
    
    titles = [
        f"Noisy\nMSE={mse(original, noisy):.2f} dB",
        f"DWT-Ideal\nMSE={mse(original, deno_ideal):.2f} dB",
        f"DWT-GCV\nMSE={mse(original, deno_gcv):.2f} dB",
        f"DFB\nMSE={mse(original, deno_dfb):.2f} dB"
    ]
    images = [noisy, deno_ideal, deno_gcv, deno_dfb]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for ax, img, t in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(t)
        ax.axis('off')

    plt.suptitle("So sánh Denoising: DWT-Ideal vs DWT-GCV vs DFB với σ=" + str(noise_std), fontsize=12)
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
    main("./images/lena_clean.png")
