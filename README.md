**1. Tổng quan về dự án:**
Dự án xây dựng hệ thống nhận diện và tính tiền món ăn thông qua hình ảnh khay cơm. Hệ thống sử dụng mô hình YOLO để phát hiện các vùng chứa món ăn, sau đó dùng mô hình CNN (huấn luyện riêng) để phân loại món ăn. Sau khi nhận diện, hệ thống tự động tính tổng hóa đơn theo bảng giá có sẵn và tạo mã QR dẫn đến trang thanh toán giả lập.
Mục đích của dự án là tự động hoá quá trình nhận diện món ăn và thanh toán tại các nhà ăn, căng tin hoặc hệ thống phục vụ ăn uống, giúp tiết kiệm thời gian, giảm nhân lực và nâng cao trải nghiệm người dùng.

**2. Hướng dẫn cài đặt:**

Bước 1: Cài đặt Python 3.8 trở lên từ trang chính thức https://www.python.org/downloads/

Bước 2: Cài đặt Visual Studio Code (VSCode)

Bước 3: Tải mã nguồn dự án từ GitHub hoặc copy vào máy.

Bước 4: Mở thư mục dự án bằng VSCode.

Bước 5: Tạo môi trường ảo (virtual environment):
•	Mở terminal và chạy lệnh:
python -m venv venv
•	Kích hoạt môi trường:
o	Windows: .\venv\Scripts\activate
o	macOS/Linux: source venv/bin/activate

Bước 6: Cài đặt các thư viện cần thiết bằng lệnh:
pip install -r requirements.txt

**3. Hướng dẫn sử dụng:**

Bước 1: Đảm bảo mô hình CNN đã được huấn luyện và lưu với tên model.h5 trong thư mục dự án.

Bước 2: Chạy chương trình bằng lệnh:
python app.py

Bước 3: Giao diện xuất hiện, cho phép chọn ảnh khay cơm từ máy.

Bước 4: Hệ thống thực hiện các bước:
•	Phát hiện món ăn bằng YOLO.
•	Phân loại từng món bằng CNN.
•	Hiển thị danh sách món ăn, tổng tiền và mã QR.

Bước 5: Quét mã QR để truy cập trang thanh toán giả lập (có thể mở bằng trình duyệt hoặc ứng dụng quét QR).

**4. Các phần phụ thuộc:**
Các thư viện cần thiết đã được liệt kê trong file requirements.txt:

•	opencv-python: xử lý ảnh và video

•	numpy: xử lý ma trận

•	ultralytics: hỗ trợ mô hình YOLO

•	tensorflow: huấn luyện và chạy mô hình CNN

•	Pillow: xử lý ảnh với thư viện Image

Các thư viện mặc định đi kèm Python (không cần cài thêm):

•	tkinter: giao diện người dùng

•	tkinter.ttk: giao diện nâng cao

•	threading: hỗ trợ xử lý đa luồng

**5. Chất lượng chương trình:**

•	Dự án được tổ chức rõ ràng với các thư mục tách biệt: Com Trang/,Thit kho/,Thit kho trung/,Trung chien/,Ca hu kho/,Dau hu sot ca/,Canh cai/,Canh chua/,Rau muong xao toi/,Ga chien/
•	Giao diện người dùng được xây dựng bằng Tkinter thân thiện và trực quan.
•	Ứng dụng sử dụng đa luồng để tránh giao diện bị treo khi xử lý ảnh.
•	Dễ dàng mở rộng với các món ăn mới, cập nhật giá, hoặc tích hợp camera trực tiếp.

