# Khử Nhiễu Ảnh Bằng Phương Pháp Bộ Lọc Định Hướng (Directional Filter Bank - DFB)

Mã nguồn này triển khai các kỹ thuật bộ lọc định hướng cho việc khử nhiễu ảnh, cung cấp cả phương pháp sử dụng biến đổi wavelet truyền thống và phương pháp lọc định hướng nâng cao. Mã này minh họa cách tận dụng thông tin định hướng trong ảnh để bảo toàn tốt hơn các cạnh và kết cấu trong khi loại bỏ nhiễu.

## Mục Lục
- [Cài Đặt](#cài-đặt)
- [Cách Sử Dụng](#cách-sử-dụng)
- [Tổng Quan Phương Pháp](#tổng-quan-phương-pháp)
- [Tài Liệu Hàm](#tài-liệu-hàm)
  - [Hàm Tạo Bộ Lọc](#hàm-tạo-bộ-lọc)
  - [Hàm Lọc và Phân Tách](#hàm-lọc-và-phân-tách)
  - [Hàm Ngưỡng Hóa](#hàm-ngưỡng-hóa)
  - [Hàm Thuật Toán Khử Nhiễu](#hàm-thuật-toán-khử-nhiễu)
  - [Hàm Tiện Ích](#hàm-tiện-ích)
  - [Hàm Hiển Thị](#hàm-hiển-thị)

## Cài Đặt

### Bước 1: Tạo và kích hoạt môi trường ảo Python
```bash
# Tạo môi trường ảo
python3 -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows
venv\Scripts\activate
# Trên macOS/Linux
source venv/bin/activate
```

### Bước 2: Cài đặt các thư viện cần thiết
```bash
pip install numpy matplotlib scipy pywavelets scikit-image opencv-python pandas torch
```

### Bước 3: Chạy chương trình
```bash
python DFB.py
```

## Cách Sử Dụng

Mã nguồn cho phép bạn khử nhiễu ảnh bằng phương pháp bộ lọc định hướng (DFB). Bạn có thể sử dụng hàm `main()` với đường dẫn đến ảnh cần xử lý:

```python
# Chỉ định đường dẫn đến ảnh
main("./images/ten_anh.jpg")

# Hoặc sử dụng ảnh mẫu tự tạo
main()
```

## Tổng Quan Phương Pháp

Mã nguồn này triển khai ba phương pháp khử nhiễu và so sánh hiệu quả của chúng:

1. **DWT-Ideal**: Sử dụng biến đổi wavelet rời rạc với ngưỡng cố định biết trước (ngưỡng = độ lệch chuẩn nhiễu).
2. **DWT-GCV**: Sử dụng biến đổi wavelet với ngưỡng được ước lượng bằng phương pháp Generalized Cross-Validation (GCV).
3. **DFB**: Phương pháp đề xuất sử dụng bộ lọc định hướng, phân tách tín hiệu thành các subband định hướng và ngưỡng hóa từng subband.

## Tài Liệu Hàm

### Hàm Tạo Bộ Lọc

#### `create_low_pass_filter(length=17, cutoff=0.25)`
Tạo bộ lọc thông thấp tuyến tính có thể tách được như đề cập trong bài báo. Bộ lọc này có độ dài 17 và tần số cắt tại π/4.

- **Tham số**:
  - `length`: Độ dài của bộ lọc (mặc định là 17)
  - `cutoff`: Tần số cắt (mặc định là 0.25)
- **Trả về**: Bộ lọc 1D cho việc lọc 2D có thể tách được

#### `create_directional_filter(h, angle)`
Xoay bộ lọc có thể tách được để tạo bộ lọc theo góc chỉ định (đơn vị độ).

- **Tham số**:
  - `h`: Bộ lọc 1D
  - `angle`: Góc xoay (đơn vị độ)
- **Trả về**: Bộ lọc 2D đã xoay

### Hàm Lọc và Phân Tách

#### `apply_separable_filter(image, filter_1d)`
Áp dụng bộ lọc 2D có thể tách được lên ảnh.

- **Tham số**:
  - `image`: Ảnh đầu vào
  - `filter_1d`: Bộ lọc 1D
- **Trả về**: Ảnh đã được lọc

#### `quincunx_downsample(image)`
Lấy mẫu quincunx Q0, giữ lại các pixel tại vị trí (i+j)%2==0.

- **Tham số**:
  - `image`: Ảnh đầu vào kích thước H×W
- **Trả về**: Ảnh downsample có kích thước H×(W/2)

#### `quincunx_upsample(image, original_shape)`
Lấy mẫu quincunx ngược lại, đưa mảng H×(W/2) lên kích thước (H, W).

- **Tham số**:
  - `image`: Ảnh đầu vào đã được downsample
  - `original_shape`: Kích thước gốc muốn upsample lên
- **Trả về**: Ảnh đã được upsample

#### `fan_filter_bank(image, h)`
Phân tách ảnh bằng bộ lọc quạt để tạo 2 subband định hướng 45° và 135°.

- **Tham số**:
  - `image`: Ảnh đầu vào
  - `h`: Bộ lọc 1D
- **Trả về**: Hai subband đã lấy mẫu quincunx

#### `diamond_filter_bank(image, h)`
Bộ lọc kim cương 2 kênh xấp xỉ sử dụng bộ lọc có thể tách được và lấy mẫu quincunx.

- **Tham số**:
  - `image`: Ảnh đầu vào
  - `h`: Bộ lọc 1D
- **Trả về**: Hai subband đã lấy mẫu quincunx

#### `directional_filter_bank(image, h, num_bands=8)`
Xấp xỉ bộ lọc định hướng 8 băng tần sử dụng phân tách 3 mức.

- **Tham số**:
  - `image`: Ảnh đầu vào 2D
  - `h`: Bộ lọc 1D
  - `num_bands`: Số băng tần (hiện tại chỉ hỗ trợ 8)
- **Trả về**: Danh sách 8 subband định hướng

#### `inverse_directional_filter_bank(subbands, original_shape)`
Khôi phục ảnh từ các subband của DFB.

- **Tham số**:
  - `subbands`: Danh sách các ảnh subband đã xử lý
  - `original_shape`: Kích thước ảnh gốc để resize đúng
- **Trả về**: Ảnh đã tái cấu trúc

### Hàm Ngưỡng Hóa

#### `soft_threshold(x, t)`
Hàm ngưỡng hóa mềm dùng trong khử nhiễu wavelet. Co các hệ số bằng ngưỡng t.

- **Tham số**:
  - `x`: Dữ liệu đầu vào
  - `t`: Giá trị ngưỡng
- **Trả về**: Dữ liệu đã ngưỡng hóa

#### `sure_threshold(coeffs, sigma)`
Tính ngưỡng tối ưu dựa trên Stein's Unbiased Risk Estimate (SURE).

- **Tham số**:
  - `coeffs`: Hệ số wavelet
  - `sigma`: Độ lệch chuẩn nhiễu
- **Trả về**: Giá trị ngưỡng tối ưu

#### `visushrink_threshold(sigma, n)`
Tính ngưỡng phổ quát sử dụng VisuShrink: T = sigma * sqrt(2 * log(n)).

- **Tham số**:
  - `sigma`: Độ lệch chuẩn nhiễu
  - `n`: Số lượng mẫu
- **Trả về**: Giá trị ngưỡng phổ quát

#### `gcv_threshold(coefs)`
Ước lượng ngưỡng thông qua Generalized Cross-Validation (GCV)/SURE.

- **Tham số**:
  - `coefs`: Mảng 1D của hệ số chi tiết wavelet
- **Trả về**: Giá trị ngưỡng được ước lượng

### Hàm Thuật Toán Khử Nhiễu

#### `dfb_denoise(image, noise_std)`
Khử nhiễu ảnh sử dụng Directional Filter Bank (DFB). Bao gồm tách tần số thấp/cao, phân tích tần số cao bằng DFB, ngưỡng hóa từng subband, rồi tái cấu trúc lại ảnh.

- **Tham số**:
  - `image`: Ảnh đầu vào
  - `noise_std`: Độ lệch chuẩn của nhiễu
- **Trả về**: Ảnh đã khử nhiễu

#### `dwt_ideal_denoise(image, sigma, wavelet='sym4', level=6)`
Khử nhiễu ảnh sử dụng biến đổi wavelet 2D với ngưỡng cố định (lý tưởng).

- **Tham số**:
  - `image`: Ảnh đầu vào
  - `sigma`: Độ lệch chuẩn nhiễu (dùng cho ngưỡng)
  - `wavelet`: Loại wavelet sử dụng
  - `level`: Số mức phân tách
- **Trả về**: Ảnh đã khử nhiễu

#### `dwt_gcv_denoise(image, wavelet='sym4', level=4)`
Khử nhiễu ảnh sử dụng biến đổi wavelet 2D với ngưỡng được chọn bởi GCV.

- **Tham số**:
  - `image`: Ảnh đầu vào
  - `wavelet`: Loại wavelet sử dụng
  - `level`: Số mức phân tách
- **Trả về**: Ảnh đã khử nhiễu

#### `compare_methods(original, noisy, noise_std)`
So sánh ba phương pháp khử nhiễu: DWT-ideal, DWT-GCV và DFB.

- **Tham số**:
  - `original`: Ảnh gốc
  - `noisy`: Ảnh có nhiễu
  - `noise_std`: Độ lệch chuẩn nhiễu
- **Trả về**: Ba ảnh đã khử nhiễu tương ứng với ba phương pháp

### Hàm Tiện Ích

#### `estimate_noise_variance(subband)`
Ước lượng phương sai nhiễu trong một subband bằng MAD (Median Absolute Deviation).

- **Tham số**:
  - `subband`: Subband đầu vào
- **Trả về**: Phương sai nhiễu ước lượng

#### `add_gaussian_noise(image, std)`
Thêm nhiễu Gaussian trắng với độ lệch chuẩn std vào ảnh.

- **Tham số**:
  - `image`: Ảnh đầu vào
  - `std`: Độ lệch chuẩn của nhiễu
- **Trả về**: Ảnh đã thêm nhiễu

#### `mse(original, denoised)`
Tính Mean Squared Error giữa ảnh gốc và ảnh khử nhiễu.

- **Tham số**:
  - `original`: Ảnh gốc
  - `denoised`: Ảnh đã khử nhiễu
- **Trả về**: Giá trị MSE

#### `psnr(original, denoised)`
Tính Peak Signal-to-Noise Ratio giữa ảnh gốc và ảnh khử nhiễu.

- **Tham số**:
  - `original`: Ảnh gốc
  - `denoised`: Ảnh đã khử nhiễu
- **Trả về**: Giá trị PSNR (dB)

#### `load_and_prepare_image(image_path, size=512)`
Tải và tiền xử lý ảnh: chuyển sang grayscale, resize và scale về 0-255.

- **Tham số**:
  - `image_path`: Đường dẫn đến ảnh
  - `size`: Kích thước mục tiêu
- **Trả về**: Ảnh đã tiền xử lý

#### `create_test_pattern(size=512)`
Tạo ảnh mẫu tổng hợp với các cấu trúc: bàn cờ, cạnh và vòng tròn.

- **Tham số**:
  - `size`: Kích thước ảnh mẫu
- **Trả về**: Ảnh mẫu đã tạo

#### `resize_to_shape(image, target_shape)`
Thay đổi kích thước ảnh theo kích thước mục tiêu.

- **Tham số**:
  - `image`: Ảnh đầu vào
  - `target_shape`: Kích thước mục tiêu
- **Trả về**: Ảnh đã được resize

### Hàm Hiển Thị

#### `display_comparison(original, noisy, denoised, title="Image Denoising Comparison")`
Hiển thị ảnh gốc, ảnh có nhiễu và ảnh sau khử nhiễu, cùng với PSNR.

- **Tham số**:
  - `original`: Ảnh gốc
  - `noisy`: Ảnh có nhiễu
  - `denoised`: Ảnh đã khử nhiễu
  - `title`: Tiêu đề của biểu đồ

#### `visualize_directional_subbands(subbands, normalize=True)`
Hiển thị 8 subband hướng của DFB theo bố cục 2x4.

- **Tham số**:
  - `subbands`: Danh sách 8 subband định hướng
  - `normalize`: Chuẩn hóa giá trị để hiển thị tốt hơn
