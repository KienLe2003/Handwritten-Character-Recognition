
# Image Processing App

Ứng dụng này cung cấp khả năng tải lên ảnh và thực hiện phân tích ảnh bằng cách sử dụng hai loại mô hình học máy: **CNN Model** và **Softmax Model**. Nó hỗ trợ dự đoán số trên ảnh sau khi áp dụng một số phương pháp xử lý ảnh, bao gồm biến đổi hình ảnh và nhận diện chữ số.

## Các tính năng chính:
- **Tải ảnh**: Người dùng có thể tải ảnh từ máy tính của mình.
- **Chọn mô hình**: Ứng dụng hỗ trợ hai loại mô hình học máy: **CNN Model** và **Softmax Model**.
- **Xử lý ảnh**: Ứng dụng xử lý ảnh tải lên, bao gồm chuyển đổi sang ảnh xám, làm mờ ảnh, tìm kiếm các đường viền, và sau đó phân loại các số trên ảnh.
- **Dự đoán**: Sau khi xử lý ảnh, ứng dụng sử dụng mô hình đã chọn để dự đoán chữ số trên ảnh.

## Các bước sử dụng:
1. **Tải ảnh**:
   - Nhấn nút **Tải ảnh** để chọn một ảnh từ máy tính.
   
2. **Chọn mô hình**:
   - Nhấn nút **Chọn model** để chọn mô hình học máy (CNN hoặc Softmax). Sau khi mô hình được tải thành công, tên mô hình sẽ xuất hiện trên nút.

3. **Xử lý và dự đoán**:
   - Sau khi tải ảnh và chọn mô hình, nhấn nút **Dự đoán** để ứng dụng thực hiện phân tích và dự đoán số trên ảnh.

4. **Reset**:
   - Nhấn nút **Reset** để xóa ảnh hiện tại, mô hình đã chọn và kết quả dự đoán.

## Các mô hình:
- **CNN Model**: Đây là mô hình học sâu CNN (Convolutional Neural Network) được huấn luyện để nhận diện các chữ số trên ảnh.
- **Softmax Model**: Mô hình này sử dụng thuật toán phân loại Softmax, phù hợp cho các bài toán phân loại đơn giản.

## Các yêu cầu cài đặt:
Ứng dụng yêu cầu các thư viện Python sau:
- `tkinter` (được tích hợp sẵn trong Python)
- `tensorflow` (dành cho mô hình CNN)
- `numpy` (cho các phép toán số học)
- `opencv-python` (để xử lý ảnh)
- `Pillow` (để hiển thị ảnh trong giao diện Tkinter)

Bạn có thể cài đặt các thư viện còn lại bằng cách sử dụng pip:
```bash
pip install tensorflow numpy opencv-python pillow
```

## Cách chạy ứng dụng:
1. Cài đặt các thư viện phụ thuộc như đã mô tả ở trên.
2. Tải mô hình đã huấn luyện (CNN hoặc Softmax) vào thư mục chứa mã nguồn.
3. Chạy tệp Python chứa mã nguồn của ứng dụng.

Ứng dụng sẽ mở một cửa sổ GUI, nơi bạn có thể thực hiện các thao tác tải ảnh, chọn mô hình và xem kết quả dự đoán.

## Lưu ý:
- Ứng dụng yêu cầu hình ảnh đầu vào có định dạng `.jpg`, `.png`, hoặc `.jpeg`.
- Các mô hình cần phải được huấn luyện và lưu lại trước khi sử dụng trong ứng dụng. Nếu mô hình không tồn tại tại đường dẫn được chỉ định, ứng dụng sẽ báo lỗi.

.
