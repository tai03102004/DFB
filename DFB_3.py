import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal, ndimage
import cv2

def generate_lowpass_filter(size=17, cutoff=0.25):
    """
    Generate a separable low-pass filter with given size and cutoff frequency.
    """
    # Generate 1D filter
    n = np.arange(size) - (size-1)/2
    h = np.sinc(2 * cutoff * n)  # Ideal low-pass filter (sinc)
    # Apply window for better frequency response
    window = np.hamming(size)
    h = h * window
    # Normalize
    h = h / np.sum(h)
    return h

def apply_separable_filter(image, filter_1d):
    """Apply a separable filter to an image."""
    # Apply horizontally
    temp = ndimage.convolve1d(image, filter_1d, axis=1, mode='reflect')
    # Apply vertically
    return ndimage.convolve1d(temp, filter_1d, axis=0, mode='reflect')

def quincunx_downsample(image):
    """Downsample image using quincunx pattern."""
    # Create quincunx pattern (checkerboard)
    pattern = np.zeros(image.shape, dtype=bool)
    pattern[0::2, 0::2] = True
    pattern[1::2, 1::2] = True
    
    # Extract values at quincunx positions
    values = image[pattern]
    
    # Reshape to form downsampled image
    m, n = image.shape
    downsampled = values.reshape((m+1)//2, (n+1)//2)
    return downsampled

def quincunx_upsample(image, target_shape):
    """Upsample image using quincunx pattern."""
    m, n = target_shape
    
    # Create upsampled image
    upsampled = np.zeros(target_shape)
    
    # Create quincunx pattern
    pattern = np.zeros(target_shape, dtype=bool)
    pattern[0::2, 0::2] = True
    pattern[1::2, 1::2] = True
    
    # Reshape downsampled image to fit pattern
    upsampled[pattern] = image.flatten()
    
    return upsampled

def fan_filter_pair(size=17):
    """
    Generate a pair of fan filters (for two-channel structure).
    Returns h0 and h1 filters.
    """
    # Generate basic lowpass filter
    lp_filter = generate_lowpass_filter(size, 0.25)
    
    # Create 2D filter
    h0_2d = np.outer(lp_filter, lp_filter)
    
    # Create complementary filter (for the second channel)
    # In the ideal case, H1(z) = H0(-z) - Using modulation to achieve this
    modulation = np.ones((size, size))
    for i in range(size):
        for j in range(size):
            modulation[i, j] = (-1)**(i+j)
    
    h1_2d = h0_2d * modulation
    
    return h0_2d, h1_2d

def implement_dfb_level(image, level=3):
    """
    Implement one level of the directional filter bank.
    For an 8-band DFB, level should be 3.
    """
    # Number of bands at this level
    num_bands = 2**level
    
    # Generate fan filters
    h0, h1 = fan_filter_pair()
    
    # Apply fan filters to get initial two channels
    y0 = signal.convolve2d(image, h0, mode='same')
    y1 = signal.convolve2d(image, h1, mode='same')
    
    # Downsample using quincunx pattern
    y0_down = quincunx_downsample(y0)
    y1_down = quincunx_downsample(y1)
    
    # If this is the first level, return the two channels
    if level == 1:
        return [y0_down, y1_down]
    
    # Otherwise, continue decomposition recursively
    subbands0 = implement_dfb_level(y0_down, level-1)
    subbands1 = implement_dfb_level(y1_down, level-1)
    
    # Combine results
    return subbands0 + subbands1

def dfb_decomposition(image, level=3):
    """
    Perform directional filter bank decomposition.
    For an 8-band DFB, level should be 3.
    """
    # Pre-processing may be needed based on specific implementation details
    # For simplicity, we use the image directly
    return implement_dfb_level(image, level)

def dfb_reconstruction(subbands, image_shape, level=3):
    """
    Reconstruct image from DFB subbands.
    For an 8-band DFB, level should be 3.
    """
    # Base case
    if level == 1:
        # We have two subbands
        y0_down, y1_down = subbands
        
        # Upsample
        y0 = quincunx_upsample(y0_down, image_shape)
        y1 = quincunx_upsample(y1_down, image_shape)
        
        # Generate synthesis filters (in this simple case, using the same filters)
        h0, h1 = fan_filter_pair()
        
        # Apply synthesis filters
        z0 = signal.convolve2d(y0, h0, mode='same')
        z1 = signal.convolve2d(y1, h1, mode='same')
        
        # Combine results
        return z0 + z1
    
    # Split subbands for recursive reconstruction
    num_bands = 2**level
    mid = num_bands // 2
    
    # Calculate intermediate shape for reconstruction
    # This would depend on the specific implementation details
    # For simplicity, we use a fixed shape calculation
    if level > 1:
        # Estimate intermediate shape
        intermed_shape = (image_shape[0]//2**(level-1), image_shape[1]//2**(level-1))
    else:
        intermed_shape = image_shape
        
    # Recursively reconstruct
    y0 = dfb_reconstruction(subbands[:mid], intermed_shape, level-1)
    y1 = dfb_reconstruction(subbands[mid:], intermed_shape, level-1)
    
    # Combine results for final reconstruction
    # This would depend on the specific tree structure and filters
    # For a simple case, we concatenate the results
    reconstructed = np.zeros(image_shape)
    reconstructed[:intermed_shape[0], :intermed_shape[1]] = y0
    reconstructed[:intermed_shape[0], intermed_shape[1]:] = y1
    
    return reconstructed

def soft_threshold(x, threshold):
    """Apply soft thresholding to x."""
    sign = np.sign(x)
    return sign * np.maximum(np.abs(x) - threshold, 0)

def estimate_noise_variance(subband):
    """Estimate noise variance using MAD."""
    # Use median absolute deviation (MAD) to estimate noise variance
    mad = np.median(np.abs(subband - np.median(subband)))
    # For Gaussian noise, sigma = MAD / 0.6745
    return (mad / 0.6745) ** 2

def dfb_denoise(image, sigma):
    """
    Denoise image using the Directional Filter Bank approach.
    
    Parameters:
    -----------
    image : 2D array
        Input noisy image
    sigma : float
        Standard deviation of the noise
    
    Returns:
    --------
    denoised : 2D array
        Denoised image
    """
    # Step 1: Separate low and high frequency components
    lp_filter = generate_lowpass_filter(size=17, cutoff=0.25)
    x_low = apply_separable_filter(image, lp_filter)
    x_high = image - x_low
    
    # Step 2: Apply directional decomposition to high frequency component
    subbands = dfb_decomposition(x_high, level=3)  # 8-band DFB
    
    # Step 3: Apply thresholding to each subband
    thresholded_subbands = []
    for i, subband in enumerate(subbands):
        # Estimate noise variance for this subband
        noise_var = estimate_noise_variance(subband)
        
        # Set threshold as per paper recommendation (1.4 * sigma)
        threshold = 1.4 * np.sqrt(noise_var)
        
        # Apply soft thresholding
        thresholded = soft_threshold(subband, threshold)
        thresholded_subbands.append(thresholded)
    
    # Step 4: Reconstruct high frequency component
    x_high_denoised = dfb_reconstruction(thresholded_subbands, x_high.shape, level=3)
    
    # Step 5: Combine with low frequency component
    denoised = x_low + x_high_denoised
    
    return denoised

def dwt_ideal_denoise(image, sigma, wavelet='sym4', level=6):
    """
    Denoise image using 2D DWT with fixed (ideal) threshold.
    
    Parameters:
    -----------
    image : 2D array
        Input noisy image
    sigma : float
        Standard deviation of the noise
    wavelet : str
        Wavelet to use (default: 'sym4' - symmlet with 4 vanishing moments)
    level : int
        Number of decomposition levels
    
    Returns:
    --------
    denoised : 2D array
        Denoised image
    """
    # Perform 2D wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    # Extract approximation coefficients (low frequency)
    cA, details = coeffs[0], coeffs[1:]
    
    # Threshold value - ideal threshold that minimizes MSE
    threshold = sigma  # As mentioned in the paper for comparison
    
    # Apply soft-thresholding to each detail subband
    new_details = []
    for level_detail in details:
        cH, cV, cD = level_detail
        cH_thresh = soft_threshold(cH, threshold)
        cV_thresh = soft_threshold(cV, threshold)
        cD_thresh = soft_threshold(cD, threshold)
        new_details.append((cH_thresh, cV_thresh, cD_thresh))
    
    # Reconstruct image from original approximation and thresholded details
    coeffs_thresh = [cA] + new_details
    denoised = pywt.waverec2(coeffs_thresh, wavelet)
    
    return denoised

def gcv_threshold(coeffs):
    """
    Estimate threshold via Generalized Cross-Validation (GCV).
    
    Parameters:
    -----------
    coeffs : 1D array
        Wavelet detail coefficients
    
    Returns:
    --------
    threshold : float
        Estimated threshold value
    """
    # Estimate noise level sigma using median absolute deviation
    mad = np.median(np.abs(coeffs - np.median(coeffs)))
    sigma = mad / 0.6745
    
    n = len(coeffs)
    if n == 0:
        return 0.0
    
    # Sort coefficients by magnitude
    sorted_coeffs = np.sort(np.abs(coeffs))
    
    # Calculate risk for each potential threshold value
    risks = np.zeros(n)
    for i in range(n):
        threshold = sorted_coeffs[i]
        
        # Count coefficients above threshold
        num_above = np.sum(np.abs(coeffs) > threshold)
        
        # Calculate risk using GCV formula
        term1 = np.sum(np.minimum(coeffs**2, threshold**2))
        term2 = 2 * sigma**2 * num_above
        term3 = n * sigma**2
        
        risks[i] = term1 + term2 - term3
    
    # Find threshold that minimizes risk
    min_idx = np.argmin(risks)
    return sorted_coeffs[min_idx]

def dwt_gcv_denoise(image, wavelet='sym4', level=4):
    """
    Denoise image using 2D DWT with threshold selected by GCV.
    
    Parameters:
    -----------
    image : 2D array
        Input noisy image
    wavelet : str
        Wavelet to use (default: 'sym4' - symmlet with 4 vanishing moments)
    level : int
        Number of decomposition levels
    
    Returns:
    --------
    denoised : 2D array
        Denoised image
    """
    # Perform 2D wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    # Extract approximation coefficients (low frequency)
    cA, details = coeffs[0], coeffs[1:]
    
    # Apply GCV-based threshold to each detail subband
    new_details = []
    for level_detail in details:
        cH, cV, cD = level_detail
        
        # Combine all coefficients from this level to compute threshold
        all_coeffs = np.concatenate([cH.flatten(), cV.flatten(), cD.flatten()])
        threshold = gcv_threshold(all_coeffs)
        
        # Apply soft-thresholding with the computed threshold
        cH_thresh = soft_threshold(cH, threshold)
        cV_thresh = soft_threshold(cV, threshold)
        cD_thresh = soft_threshold(cD, threshold)
        
        new_details.append((cH_thresh, cV_thresh, cD_thresh))
    
    # Reconstruct image from original approximation and thresholded details
    coeffs_thresh = [cA] + new_details
    denoised = pywt.waverec2(coeffs_thresh, wavelet)
    
    return denoised

def add_gaussian_noise(image, sigma):
    """Add Gaussian noise to an image."""
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255)  # Clip to valid range

def calculate_mse(original, processed):
    """Calculate Mean Squared Error between two images."""
    return np.mean((original - processed)**2)

def compare_denoising_methods(image_path, sigma_values=[10, 15, 20]):
    """
    Compare three denoising methods:
    1. DFB (Directional Filter Bank)
    2. DWT with ideal threshold
    3. DWT with GCV threshold
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    sigma_values : list
        List of noise standard deviations to test
    """
    # Load the image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Ensure image is of type float
    original_image = original_image.astype(np.float64)
    
    results = {}
    
    for sigma in sigma_values:
        print(f"Processing with sigma = {sigma}...")
        
        # Add Gaussian noise
        noisy_image = add_gaussian_noise(original_image, sigma)
        
        # Apply the three denoising methods
        dfb_result = dfb_denoise(noisy_image, sigma)
        dwt_ideal_result = dwt_ideal_denoise(noisy_image, sigma)
        dwt_gcv_result = dwt_gcv_denoise(noisy_image)
        
        # Calculate MSE for each method
        mse_dfb = calculate_mse(original_image, dfb_result)
        mse_dwt_ideal = calculate_mse(original_image, dwt_ideal_result)
        mse_dwt_gcv = calculate_mse(original_image, dwt_gcv_result)
        
        # Store results
        results[sigma] = {
            'MSE_DFB': mse_dfb,
            'MSE_DWT_Ideal': mse_dwt_ideal,
            'MSE_DWT_GCV': mse_dwt_gcv
        }
        
        # Display results
        print(f"MSE for sigma = {sigma}:")
        print(f"  DFB: {mse_dfb:.2f}")
        print(f"  DWT-Ideal: {mse_dwt_ideal:.2f}")
        print(f"  DWT-GCV: {mse_dwt_gcv:.2f}")
        
        # Visualize results for sigma = 15 (as in the paper)
        if sigma == 15:
            plt.figure(figsize=(12, 10))
            
            plt.subplot(2, 2, 1)
            plt.imshow(noisy_image, cmap='gray')
            plt.title('Noisy Image')
            plt.axis('off')
            
            plt.subplot(2, 2, 2)
            plt.imshow(dwt_ideal_result, cmap='gray')
            plt.title(f'DWT-Ideal Threshold (MSE: {mse_dwt_ideal:.2f})')
            plt.axis('off')
            
            plt.subplot(2, 2, 3)
            plt.imshow(dwt_gcv_result, cmap='gray')
            plt.title(f'DWT-GCV (MSE: {mse_dwt_gcv:.2f})')
            plt.axis('off')
            
            plt.subplot(2, 2, 4)
            plt.imshow(dfb_result, cmap='gray')
            plt.title(f'DFB (MSE: {mse_dfb:.2f})')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig('denoising_comparison_sigma15.png')
            plt.show()
    
    # Print table of results
    print("\nSummary of MSE Results:")
    print("----------------------")
    print("Sigma | DFB     | DWT-Ideal | DWT-GCV")
    print("-------------------------------------")
    for sigma in sigma_values:
        print(f"{sigma:5d} | {results[sigma]['MSE_DFB']:.2f} | {results[sigma]['MSE_DWT_Ideal']:.2f} | {results[sigma]['MSE_DWT_GCV']:.2f}")

# Example usage
if __name__ == "__main__":
    # Path to your test image (e.g., Lenna)
    image_path = "./images/lena_clean.png"
    
    # Compare methods with different noise levels
    compare_denoising_methods(image_path, sigma_values=[10, 15, 20])