import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
import pickle
from PIL import Image, ImageTk
import numpy as np
import cv2
import numpy as np

class App:
    def __init__(self, window):
        self.window = window
        self.window.title("Image Processing App")
        self.current_image = None
        self.model = None  # Biến lưu mô hình
        self.model_type = None  # Loại mô hình (CNN hoặc Softmax)
        # Khởi tạo các thuộc tính khác
        self.softmax_weights = None
        self.load_softmax_weights('softmax_weight.pkl')

        # Các thành phần giao diện
        self.panel = tk.Label(self.window)
        self.panel.grid(row=0, column=0, padx=10, pady=10)

        self.load_button = tk.Button(self.window, text="Tải ảnh", command=self.load_image)
        self.load_button.grid(row=1, column=0, padx=10, pady=10)

        self.model_button = tk.Button(self.window, text="Chọn model", command=self.select_model)
        self.model_button.grid(row=2, column=0, padx=10, pady=10)

        self.reset_button = tk.Button(self.window, text="Reset", command=self.reset)
        self.reset_button.grid(row=3, column=0, padx=10, pady=10)

        

        # Danh sách các mô hình có sẵn (bao gồm CNN và Softmax)
        self.models = ["CNN Model", "Softmax Model"]  # Danh sách mô hình sẽ hiển thị

        # Đường dẫn đến các mô hình đã lưu
        self.model_paths = {
            "CNN Model": "./cnn_model_weights.keras",  # Đường dẫn đến mô hình CNN
            "Softmax Model": "./softmax_weight.pkl"  # Đường dẫn đến mô hình Softmax
        }

    def load_image(self):
        # Chọn ảnh từ file
        file_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        
        # Tải ảnh từ đường dẫn
        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Lỗi", "Không thể tải ảnh. Vui lòng chọn lại ảnh hợp lệ.")
            return
        # Lưu ảnh vào biến self.current_image
        self.current_image = image
        self.current_image_path = file_path

        print(f"Đã tải ảnh: {self.current_image_path}")
        print(f"Kích thước ảnh: {self.current_image.shape}")  # In ra kích thước của ảnh
        self.show_image(self.current_image)

    def show_image(self, image):
        # Kiểm tra nếu image là None (khi reset hoặc không có ảnh)
        if image is None:
            self.panel.config(image=None)
            self.panel.image = None
            return

    #   Kiểm tra nếu image là mảng NumPy (OpenCV sử dụng NumPy)
        if isinstance(image, np.ndarray):
        # Chuyển từ BGR (OpenCV) sang RGB (PIL)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        # Chuyển từ NumPy array sang PIL Image
            image = Image.fromarray(image)

        # Resize ảnh
        image = image.resize((250, 250), Image.Resampling.LANCZOS)

        # Chuyển đổi sang dạng có thể hiển thị trên Tkinter
        self.image_tk = ImageTk.PhotoImage(image)

        # Cập nhật ảnh vào giao diện Tkinter
        self.panel.config(image=self.image_tk)
        self.panel.image = self.image_tk

    def select_model(self):
        # Tạo cửa sổ con để chọn model
        def on_select_model(model_name):
            try:
                
                if model_name == "CNN Model":
                    self.model = load_model(self.model_paths[model_name])  # Tải mô hình CNN
                    self.model_type = "CNN"
                elif model_name == "Softmax Model":
                    with open(self.model_paths[model_name], 'rb') as f:
                        self.model = pickle.load(f)  # Tải mô hình Softmax
                    self.model_type = "Softmax"
                
                # Cập nhật nút chọn model thành tên mô hình đã chọn
                self.model_button.config(text=model_name)

                messagebox.showinfo("Thông báo", f"Mô hình {model_name} đã được tải thành công.")
                model_selection_window.destroy()  # Đóng cửa sổ chọn model
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải mô hình: {e}")
                model_selection_window.destroy()

        # Tạo cửa sổ con
        model_selection_window = tk.Toplevel(self.window)
        model_selection_window.title("Chọn Mô Hình")

        label = tk.Label(model_selection_window, text="Chọn mô hình để tải:")
        label.grid(row=0, column=0, padx=10, pady=10)

        # Tạo nút để chọn mô hình
        row = 1
        for model_name in self.models:
            button = tk.Button(model_selection_window, text=model_name, command=lambda name=model_name: on_select_model(name))
            button.grid(row=row, column=0, padx=10, pady=5)
            row += 1

    def reset(self):
        self.current_image = None  # Xóa ảnh hiện tại
        self.model = None  # Xóa mô hình đã tải
        self.model_type = None  # Xóa loại mô hình đã chọn
        
        self.image_tk = None
    
        # Xóa ảnh trên giao diện
        self.show_image(None)  # Gọi hàm show_image với tham số None để xóa ảnh
    
        self.result_text.config(text="")  # Xóa kết quả dự đoán trên giao diện
        self.result_label.config(text="Kết quả:")  # Đặt lại nhãn kết quả về trạng thái ban đầu
        self.model_button.config(text="Chọn model")  # Đặt lại tên nút chọn model

    print("Giao diện đã được reset.")

        

    def load_softmax_weights(self, file_path):
        # Tải trọng số mô hình Softmax từ file
        try:
            with open(file_path, 'rb') as f:
                self.softmax_weights = pickle.load(f)  # Gán trọng số vào thuộc tính self.softmax_weights
                print("Trọng số mô hình đã được tải thành công.")
        except Exception as e:
            print(f"Error loading softmax weights: {e}")

    def process_image(self):
        # Kiểm tra xem đã chọn mô hình và ảnh hay chưa
        if self.model is None:
            messagebox.showerror("Lỗi", "Vui lòng chọn mô hình trước.")
            return
        
        if self.current_image is None:
            messagebox.showerror("Lỗi", "Vui lòng tải ảnh lên trước!")
            return
    
        # Kiểm tra xem ảnh có phải là mảng NumPy hợp lệ không
        if not isinstance(self.current_image, np.ndarray):
            messagebox.showerror("Lỗi", "Ảnh tải lên không hợp lệ!")
            return

        # Xử lý ảnh
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _,blurred = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        # Áp dụng phép dãn (dilation) để làm nổi bật các vùng chữ viết
        kernel = np.ones((5, 5), np.uint8)
        blurred = cv2.dilate(blurred, kernel, iterations=1) 
        edges = cv2.Canny(blurred, 50, 150)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detect = []
        predictions = []
        for i, contour in enumerate(contours):
            if hierarchy[0][i][2] == -1:
                x, y, w, h = cv2.boundingRect(contour)
                src_points = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
                dst_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
                M = cv2.getPerspectiveTransform(src_points, dst_points)
                warped_number = cv2.warpPerspective(self.current_image, M, (w, h))
                detect.append(warped_number)
                
        # Dự đoán với mô hình đã chọn
            if self.model_type == "Softmax":
                predicted_class = self.classify_with_softmax([warped_number])
            else:
                predicted_class = self.classification_withCNN([warped_number])

            predictions.append(predicted_class[0])

            # Vẽ contour và số dự đoán lên ảnh
            cv2.rectangle(self.current_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(self.current_image, str(predicted_class[0]), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                         

       # Hiển thị kết quả
        self.show_image(self.current_image)
        
        

    def classification_withCNN(self, detect):
        predictionsCNN = []
        for img in detect:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            two_color = cv2.inRange(gray, 75, 255)
            two_color = 255 - two_color
            img_resized = cv2.resize(two_color, (28, 28), interpolation=cv2.INTER_CUBIC)
            img_normalized = img_resized / 255.0
            img_expanded = np.expand_dims(img_normalized, axis=-1)
            img_expanded = np.expand_dims(img_expanded, axis=0)
            predictions_prob = self.model.predict(img_expanded)
            predicted_class = np.argmax(predictions_prob, axis=1)
            predictionsCNN.append(predicted_class[0])
        return predictionsCNN

    def classify_with_softmax(self, detect):
        predictions = []
        for img in detect:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            two_color = cv2.inRange(gray, 75, 255)
            two_color = 255 - two_color
            img_resized = cv2.resize(two_color, (56, 56), interpolation=cv2.INTER_CUBIC)
            img_normalized = img_resized / 255.0
            img_flattened = img_normalized.reshape(1, -1)
            y_pred = self.predict(img_flattened, self.softmax_weights)
            predicted_class = np.argmax(y_pred)
            predictions.append(predicted_class)
        return predictions

    def predict(self, x, w):
        h = np.dot(x, w)
        softmax = np.exp(h - np.max(h, axis=1, keepdims=True))
        y_pred = softmax / np.sum(softmax, axis=1, keepdims=True)
        return y_pred         


# Khởi tạo giao diện
root = tk.Tk()
app = App(root)

# Thêm nút để gọi hàm xử lý sau khi chọn mô hình
process_button = tk.Button(root, text="Dự đoán", command=app.process_image)
process_button.grid(row=6, column=0, padx=10, pady=10)

root.mainloop()
