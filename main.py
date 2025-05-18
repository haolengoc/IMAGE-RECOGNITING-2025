import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

# Load models
model1 = YOLO('D:/AI/yolo_tu_train.pt')
model2 = load_model('last_cnn.h5')

# Dishes and prices
dish_labels = ["Ca hu kho", "Canh cai", "Canh chua", "Com trang",
               "Dau hu sot ca", "Ga chien", "Rau muong xao toi",
               "Thit kho", "Thit kho trung", "Trung chien"]

dish_prices = {
    "Ca hu kho": 22000,
    "Canh cai": 9000,
    "Canh chua": 10000,
    "Com trang": 5000,
    "Dau hu sot ca": 16000,
    "Ga chien": 25000,
    "Rau muong xao toi": 8000,
    "Thit kho": 17000,
    "Thit kho trung": 18000,
    "Trung chien": 12000
}

# Global variables
captured_image = None
boxes = []
confidence_threshold = 0.3
recognized_dishes = []
webcam_running = True

# Detect food regions
def detect_food_regions(model, image, conf=0.3):
    results = model(image, conf=conf)
    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)
    confs = results[0].boxes.conf.cpu().numpy()

    boxes = []
    for box, c in zip(boxes_xyxy, confs):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if w * h > 1000:
            boxes.append((x1, y1, w, h))
    return boxes

# Classify cropped dish image
def classify_dish(model, crop_img):
    img = cv2.resize(crop_img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0]
    label_index = np.argmax(pred)
    return dish_labels[label_index]

# Update webcam preview
def update_webcam():
    if not webcam_running:
        return
    ret, frame = cap.read()
    if ret:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        canvas.imgtk = imgtk
        canvas.create_image(0, 0, anchor='nw', image=imgtk)
    canvas.after(30, update_webcam)

# Capture image from webcam
def capture_image():
    global captured_image, boxes, webcam_running
    webcam_running = False
    ret, frame = cap.read()
    if ret:
        captured_image = frame.copy()
        show_image_with_boxes()

# Update confidence threshold
def update_confidence(val):
    global confidence_threshold
    confidence_threshold = float(val)
    if captured_image is not None:
        show_image_with_boxes()

# Show image with bounding boxes
def show_image_with_boxes():
    global boxes
    image_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
    boxes = detect_food_regions(model1, image_rgb, conf=confidence_threshold)
    img_copy = image_rgb.copy()

    for (x, y, w, h) in boxes:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    img_pil = Image.fromarray(img_copy)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    canvas.imgtk = imgtk
    canvas.create_image(0, 0, anchor='nw', image=imgtk)

def process_payment():
    global recognized_dishes
    if captured_image is None:
        return

    image_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
    recognized_dishes.clear()

    for (x, y, w, h) in boxes:
        crop = image_rgb[y:y+h, x:x+w]
        label = classify_dish(model2, crop)
        recognized_dishes.append(label)

    dish_list.delete(0, tk.END)
    total = 0
    for dish in recognized_dishes:
        price = dish_prices.get(dish, 0)
        dish_list.insert(tk.END, f"{dish}: {price:,} VND")
        total += price
    total_label.config(text=f"Tổng cộng: {total:,} VND")

    # Fix here: use pack instead of grid
    qr_label.pack(pady=10)
    total_label.pack(pady=0)


# GUI Setup
root = tk.Tk()
root.title("Nhận diện món ăn từ khay cơm")
root.geometry("1920x1080")

# Left frame for webcam and buttons
left_frame = tk.Frame(root)
left_frame.pack(side='left', padx=20, pady=20)

canvas = tk.Canvas(left_frame, width=640, height=480)
canvas.pack()

btn_frame = tk.Frame(left_frame)
btn_frame.pack(fill='x', pady=10)

capture_btn = tk.Button(btn_frame, text="📸 Chụp hình", command=capture_image)
capture_btn.pack(side='left', expand=True, fill='x', padx=5)

pay_btn = tk.Button(btn_frame, text="💵 Thanh toán", command=process_payment)
pay_btn.pack(side='left', expand=True, fill='x', padx=5)

conf_label = tk.Label(left_frame, text="Confidence YOLO")
conf_label.pack(anchor='w')

conf_slider = tk.Scale(left_frame, from_=0.1, to=1.0, resolution=0.05,
                       orient=tk.HORIZONTAL, command=update_confidence, length=300)
conf_slider.set(confidence_threshold)
conf_slider.pack()

# Right frame for dish list and QR
right_frame = tk.Frame(root)
right_frame.pack(side='left', fill='both', expand=True, padx=20, pady=20)

list_frame = tk.Frame(right_frame)
list_frame.pack(anchor='n', fill='x')

dish_list = tk.Listbox(list_frame, width=50, height=10)
dish_list.pack(pady=10)

qr_img = Image.open("qr.jpg")
qr_img = qr_img.resize((220, 220))
qr_imgtk = ImageTk.PhotoImage(qr_img)
qr_label = tk.Label(right_frame, image=qr_imgtk)

total_label = tk.Label(right_frame, text="Tổng cộng: 0 VND", font=('Arial', 14, 'bold'))

# Start webcam
cap = cv2.VideoCapture(0)
update_webcam()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
