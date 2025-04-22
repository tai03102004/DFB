import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from skimage import io, color
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
    Apply quincunx downsampling with matrix M = [1, 1; 1, -1]
    This effectively rotates the data by 45 degrees
    """
    rows, cols = image.shape
    # Initialize downsampled image - quincunx pattern means half the size
    downsampled = np.zeros((rows//2, cols), dtype=image.dtype)
    
    # Implement quincunx pattern: keep alternate samples in a checkerboard pattern
    for i in range(0, rows, 2):
        for j in range(cols):
            if (i + j) % 2 == 0:  # This creates the checkerboard pattern
                downsampled[i//2, j] = image[i, j]
    
    return downsampled

def quincunx_upsample(image, original_shape):
    """
    Apply quincunx upsampling, inverse of quincunx_downsample
    """
    rows, cols = image.shape
    upsampled = np.zeros(original_shape, dtype=image.dtype)
    
    # Place samples back in checkerboard pattern
    for i in range(rows):
        for j in range(cols):
            if (2*i + j) % 2 == 0:
                upsampled[2*i, j] = image[i, j]
    
    return upsampled

def fan_filter_bank(image):
    """
    Implement a two-channel fan filter bank as described in the paper
    Returns two subbands corresponding to the fan filters
    """
    # For simplicity, we'll use predefined filters that approximate fan filters
    # In practice, proper fan filters should be designed as described in the paper
    
    # Create approximated fan filters (this is simplified)
    h0 = np.array([0.1, 0.3, 0.4, 0.3, 0.1])  # Example lowpass
    h1 = np.array([-0.1, -0.3, 0.8, -0.3, -0.1])  # Example highpass
    
    # Apply filters
    subband0 = signal.convolve2d(image, h0[:, np.newaxis] * h0[np.newaxis, :], mode='same')
    subband1 = signal.convolve2d(image, h1[:, np.newaxis] * h1[np.newaxis, :], mode='same')
    
    # Downsample using quincunx pattern
    subband0_down = quincunx_downsample(subband0)
    subband1_down = quincunx_downsample(subband1)
    
    return subband0_down, subband1_down

def diamond_filter_bank(image):
    """
    Implementation of a diamond filter bank using polyphase structure
    """
    # This is a simplified implementation
    # Create polyphase filters P0 and P1 (simplified)
    p0 = np.array([[0.25, 0.25], [0.25, 0.25]])  # Example
    p1 = np.array([[0.25, -0.25], [-0.25, 0.25]])  # Example
    
    # Apply polyphase filtering and quincunx downsampling
    rows, cols = image.shape
    
    # Downsample first
    downsampled = np.zeros((rows//2, cols//2), dtype=image.dtype)
    for i in range(0, rows, 2):
        for j in range(0, cols, 2):
            downsampled[i//2, j//2] = image[i, j]
    
    # Apply polyphase filters
    subband0 = signal.convolve2d(downsampled, p0, mode='same')
    subband1 = signal.convolve2d(downsampled, p1, mode='same')
    
    return subband0, subband1

def directional_filter_bank(image, num_bands=8):
    """
    Implementation of the directional filter bank as described in the paper
    Returns num_bands directional subbands
    """
    if num_bands != 8:
        raise ValueError("Currently only 8-band DFB is implemented")
        
    # Storage for subbands
    subbands = []
    
    # First level: split into 2 subbands
    s0, s1 = fan_filter_bank(image)
    
    # Second level: split each subband into 2 more
    s00, s01 = fan_filter_bank(s0)
    s10, s11 = fan_filter_bank(s1)
    
    # Third level: final split into 8 directional subbands
    subbands.append(fan_filter_bank(s00)[0])
    subbands.append(fan_filter_bank(s00)[1])
    subbands.append(fan_filter_bank(s01)[0])
    subbands.append(fan_filter_bank(s01)[1])
    subbands.append(fan_filter_bank(s10)[0])
    subbands.append(fan_filter_bank(s10)[1])
    subbands.append(fan_filter_bank(s11)[0])
    subbands.append(fan_filter_bank(s11)[1])
    
    return subbands

def soft_threshold(x, threshold):
    """
    Apply soft thresholding as defined in the paper:
    qs(x, λ) = sgn(x)(|x| - λ)+
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def estimate_noise_variance(subband):
    """
    Estimate noise variance in a subband using MAD (Median Absolute Deviation)
    This is a common method for robust estimation of noise in wavelet domains
    """
    # MAD estimator
    mad = np.median(np.abs(subband - np.median(subband)))
    # Convert MAD to standard deviation (assuming normal distribution)
    sigma = mad / 0.6745
    return sigma**2

def dfb_denoise(image, noise_std):
    """
    Implement image denoising using directional filter bank as described in the paper
    
    Args:
        image: Input noisy image
        noise_std: Standard deviation of the noise
        
    Returns:
        Denoised image
    """
    # Step 1: Separate into low and high frequency components
    lp_filter = create_low_pass_filter()
    xl = apply_separable_filter(image, lp_filter)  # Low frequency component
    xh = image - xl  # High frequency component by differencing
    
    # Step 2: Apply directional decomposition to high frequency component
    subbands = directional_filter_bank(xh)
    
    # Step 3: Apply soft thresholding to directional coefficients
    denoised_subbands = []
    for i, subband in enumerate(subbands):
        # Estimate noise variance in the subband
        var_i = estimate_noise_variance(subband)
        # Calculate threshold as mentioned in the paper: λi = 1.4 · σ̂i
        threshold_i = 1.4 * np.sqrt(var_i)
        # Apply soft thresholding
        denoised_subband = soft_threshold(subband, threshold_i)
        denoised_subbands.append(denoised_subband)
    
    # Step 4: Reconstruct high frequency component from denoised subbands
    # This is a simplified reconstruction - full implementation would require
    # proper inverse DFB which is more complex
    reconstructed_xh = np.zeros_like(xh)
    # Simple averaging of denoised subbands (not accurate but for demonstration)
    for subband in denoised_subbands:
        # Upsample and add
        upsampled = np.repeat(np.repeat(subband, 2, axis=0), 2, axis=1)
        # Resize to match original high frequency component size if needed
        # Resize to match original high frequency component size
        upsampled_resized = resize(upsampled, xh.shape, mode='reflect', anti_aliasing=True)
        reconstructed_xh += upsampled_resized / len(denoised_subbands)
    
    # Step 5: Combine with low frequency component
    denoised_image = xl + reconstructed_xh
    
    return denoised_image

def add_gaussian_noise(image, std):
    """Add Gaussian noise to an image"""
    noise = np.random.normal(0, std, image.shape)
    noisy_image = image + noise
    # Clip to valid range
    return np.clip(noisy_image, 0, 255)

def psnr(original, noisy):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((original - noisy) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def display_comparison(original, noisy, denoised, title="Image Denoising Comparison"):
    """Display original, noisy and denoised images side by side"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2.imshow(noisy, cmap='gray')
    ax2.set_title(f'Noisy (PSNR: {psnr(original, noisy):.2f} dB)')
    ax2.axis('off')
    
    ax3.imshow(denoised, cmap='gray')
    ax3.set_title(f'Denoised (PSNR: {psnr(original, denoised):.2f} dB)')
    ax3.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('denoising_comparison.png')  # Save the figure
    plt.show()

def visualize_directional_subbands(subbands):
    """Visualize the 8 directional subbands"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, subband in enumerate(subbands):
        # Normalize for better visualization
        subband_norm = (subband - np.min(subband)) / (np.max(subband) - np.min(subband))
        axes[i].imshow(subband_norm, cmap='gray')
        axes[i].set_title(f'Direction {i+1}')
        axes[i].axis('off')
    
    plt.suptitle('Directional Subbands')
    plt.tight_layout()
    plt.savefig('directional_subbands.png')  # Save the figure
    plt.show()

def load_and_prepare_image(image_path, size=512):
    """
    Load an image, convert to grayscale if needed, and resize
    """
    # Load image
    try:
        img = io.imread(image_path)
        
        # Convert to grayscale if image is RGB/RGBA
        if len(img.shape) > 2:
            img = color.rgb2gray(img)
        
        # Resize to desired size
        img = resize(img, (size, size), anti_aliasing=True)
        
        # Scale to [0, 255] range if not already
        if img.max() <= 1.0:
            img = img * 255
            
        return img.astype(np.float64)
    except Exception as e:
        print(f"Error loading image: {e}")
        # Create a test pattern instead
        print("Creating a test pattern instead...")
        return create_test_pattern(size)

def create_test_pattern(size=512):
    """Create a test pattern with lines and edges"""
    img = np.zeros((size, size))
    
    # Add horizontal and vertical stripes
    for i in range(size):
        for j in range(size):
            if (i//64 + j//64) % 2 == 0:
                img[i, j] = 200
    
    # Add diagonal lines
    for i in range(size):
        for j in range(size):
            if abs(i - j) < 5 or abs(i + j - size) < 5:
                img[i, j] = 230
    
    # Add a circle
    center = size // 2
    radius = size // 4
    for i in range(size):
        for j in range(size):
            if (i - center)**2 + (j - center)**2 < radius**2 and \
               (i - center)**2 + (j - center)**2 > (radius-5)**2:
                img[i, j] = 255
    
    return img

def compare_with_wavelet(original, noisy, noise_std):
    """Compare DFB denoising with wavelet denoising"""
    # DFB denoising
    dfb_denoised = dfb_denoise(noisy, noise_std)
    
    # Wavelet denoising using PyWavelets
    # Using Symlet with 4 vanishing moments as in the paper
    wavelet = 'sym4'  
    coeffs = pywt.wavedec2(noisy, wavelet, level=4)
    
    # Threshold calculation (similar to what was used in the paper)
    threshold = noise_std * np.sqrt(2 * np.log(noisy.size))
    
    # Apply thresholding
    new_coeffs = []
    new_coeffs.append(coeffs[0])  # Approximation coefficients
    
    for i in range(1, len(coeffs)):
        # Detail coefficients (horizontal, vertical, diagonal)
        detail_coeffs = []
        for detail in coeffs[i]:
            detail_coeffs.append(pywt.threshold(detail, threshold, mode='soft'))
        new_coeffs.append(tuple(detail_coeffs))
    
    # Reconstruct image
    wavelet_denoised = pywt.waverec2(new_coeffs, wavelet)
    
    # Make sure the size matches original
    if wavelet_denoised.shape != original.shape:
        wavelet_denoised = wavelet_denoised[:original.shape[0], :original.shape[1]]
    
    # Display comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(noisy, cmap='gray')
    axes[0].set_title(f'Noisy (PSNR: {psnr(original, noisy):.2f} dB)')
    axes[0].axis('off')
    
    axes[1].imshow(dfb_denoised, cmap='gray')
    axes[1].set_title(f'DFB (PSNR: {psnr(original, dfb_denoised):.2f} dB)')
    axes[1].axis('off')
    
    axes[2].imshow(wavelet_denoised, cmap='gray')
    axes[2].set_title(f'Wavelet (PSNR: {psnr(original, wavelet_denoised):.2f} dB)')
    axes[2].axis('off')
    
    plt.suptitle('DFB vs Wavelet Denoising')
    plt.tight_layout()
    plt.savefig('dfb_vs_wavelet.png')  # Save the figure
    plt.show()
    
    return dfb_denoised, wavelet_denoised

def main(image_path=None):
    """
    Main function to demonstrate DFB denoising
    If image_path is provided, loads that image, otherwise creates a test pattern
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load image or create test pattern
    if image_path:
        original = load_and_prepare_image(image_path)
    else:
        print("No image provided, creating test pattern...")
        original = create_test_pattern()
    
    # Add Gaussian noise with different standard deviations
    noise_levels = [10, 15, 20]
    
    for noise_std in noise_levels:
        print(f"\nProcessing noise level: σ = {noise_std}")
        
        # Add noise
        noisy = add_gaussian_noise(original, noise_std)
        
        # Denoise using DFB
        print("Applying DFB denoising...")
        denoised = dfb_denoise(noisy, noise_std)
        
        # Display results
        display_comparison(original, noisy, denoised, 
            title=f"DFB Denoising (Noise σ = {noise_std})")
        
        # Compare with wavelet denoising
        print("Comparing with wavelet denoising...")
        dfb_result, wavelet_result = compare_with_wavelet(original, noisy, noise_std)
        
        # Lưu ảnh gốc, nhiễu và khử nhiễu
        io.imsave(f'original_image_sigma_{noise_std}.png', original.astype(np.uint8))
        io.imsave(f'noisy_image_sigma_{noise_std}.png', noisy.astype(np.uint8))
        io.imsave(f'denoised_image_sigma_{noise_std}.png', denoised.astype(np.uint8))
        
        # For σ = 15, also visualize the directional subbands
        if noise_std == 15:
            # Get the high frequency component
            lp_filter = create_low_pass_filter()
            xl = apply_separable_filter(noisy, lp_filter)
            xh = noisy - xl
            
            # Get directional subbands
            print("Extracting directional subbands...")
            subbands = directional_filter_bank(xh)
            
            # Visualize subbands
            visualize_directional_subbands(subbands)

if __name__ == "__main__":
    # You can either provide an image path or let it create a test pattern
    # Example: main("path/to/your/image.jpg")
    # image_path = "./images/original.png"
    main("./images/denoised_result.jpg")
    
    # Alternatively, you can use this command to create a test pattern
    # main()