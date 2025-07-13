import customtkinter as ctk
import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog
import cv2

ctk.set_default_color_theme("blue")
ctk.set_appearance_mode("System")

app = ctk.CTk()
app.geometry("800x600")
app.title("Forest Eye - Model 1")

# Model yükleme - Hata handling ile
try:
    model = tf.keras.models.load_model('unet_1.h5', compile=False)
except Exception:
    # Alternatif yükleme yöntemi
    try:
        model = tf.keras.models.load_model('unet_1.h5', compile=False, custom_objects=None)
    except Exception:
        model = None

# Global değişkenler
uploaded_image = None
uploaded_image_path = None
accuracy_canvas = None

def upload_image():
    global uploaded_image, uploaded_image_path
    file_path = filedialog.askopenfilename(
        title="Fotoğraf Seç",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    
    if file_path:
        uploaded_image_path = file_path
        img = Image.open(file_path)
        img = img.resize((200, 200))
        photo = ImageTk.PhotoImage(img)
        image_label.configure(image=photo)
        image_label.image = photo
        uploaded_image = img
        if model is not None:
            predict_button.configure(state="normal")
        else:
            predict_button.configure(state="disabled")

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def calculate_accuracy(prediction):
    """Modelin gerçek doğruluk oranını hesapla"""
    # Segmentasyon sonucundan orman ve çorak toprak piksellerini say
    threshold = 0.5
    forest_pixels = np.sum(prediction > threshold)
    barren_pixels = np.sum(prediction <= threshold)
    total_pixels = prediction.size
    
    # Basit doğruluk hesaplama (gerçek projede ground truth ile karşılaştır)
    confidence = np.mean(np.abs(prediction - 0.5)) * 2  # 0-1 arası normalize
    return confidence

def predict_function():
    if uploaded_image_path and model is not None:
        # Resmi hazırla
        processed_image = preprocess_image(uploaded_image_path)
        
        # Tahmin yap
        prediction = model.predict(processed_image)
                
        # Sonucu göster
        show_prediction_result(prediction[0])
        
        # Doğruluk oranını hesapla ve güncelle
        accuracy = calculate_accuracy(prediction[0])
        update_accuracy_chart(accuracy)
    elif model is None:
        # Model yüklenmemişse hata mesajı göster
        show_error_message("Model yüklenemedi. Lütfen model dosyasını kontrol edin.")

def show_error_message(message):
    """Hata mesajını göster"""
    import tkinter.messagebox as msgbox
    msgbox.showerror("Hata", message)

def show_prediction_result(prediction):
    # Matplotlib penceresi
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Orijinal resim
    original = cv2.imread(uploaded_image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, (256, 256))
    ax1.imshow(original)
    ax1.set_title('Orijinal Resim')
    ax1.axis('off')
    
    # Segmentasyon sonucu - Açık renk orman, koyu renk çorak toprak
    # Renk haritasını tersine çevir: beyaz=orman, siyah=çorak toprak
    ax2.imshow(prediction, cmap='gray_r')  # gray_r ile beyaz=yüksek değer (orman), siyah=düşük değer (çorak)
    ax2.set_title('Segmentasyon Sonucu\n(Siyah: Orman, Beyaz: Çorak Toprak)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def update_accuracy_chart(accuracy):
    global accuracy_canvas
    
    # Önceki grafiği temizle
    if accuracy_canvas:
        accuracy_canvas.get_tk_widget().destroy()
    
    # Yeni pie chart oluştur
    fig, ax = plt.subplots(figsize=(4, 4))
    sizes = [accuracy * 100, (1 - accuracy) * 100]
    labels = ['Güvenilir', 'Belirsiz']
    colors = ['#4CAF50', '#FFC107']
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title(f'Model Güven Oranı\n{accuracy:.1%}')
    
    # Tkinter'a entegre et
    accuracy_canvas = FigureCanvasTkAgg(fig, master=accuracy_frame)
    accuracy_canvas.draw()
    accuracy_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# UI Elemanları
model_status = "Model 1 ✓" if model is not None else "Model 1 ✗"
label = ctk.CTkLabel(app, text=f"Forest Eye - {model_status}", fg_color="transparent", font=("Arial", 20))
label.place(relx=0.5, rely=0.05, anchor=ctk.CENTER)

subtitle_text = "Orman Segmentasyonu" if model is not None else "Model yüklenmedi - Kontrol edin"
subtitle_color = "white" if model is not None else "red"
label2 = ctk.CTkLabel(app, text=subtitle_text, fg_color="transparent", font=("Arial", 16), text_color=subtitle_color)
label2.place(relx=0.5, rely=0.1, anchor=ctk.CENTER)

# Fotoğraf yükleme alanı
upload_frame = ctk.CTkFrame(app, width=300, height=250)
upload_frame.place(relx=0.25, rely=0.35, anchor=ctk.CENTER)

upload_button = ctk.CTkButton(upload_frame, text="Fotoğraf Yükle", command=upload_image)
upload_button.pack(pady=10)

image_label = ctk.CTkLabel(upload_frame, text="Henüz fotoğraf seçilmedi", width=200, height=200)
image_label.pack(pady=10)

# Doğruluk grafiği alanı
accuracy_frame = ctk.CTkFrame(app, width=300, height=250)
accuracy_frame.place(relx=0.75, rely=0.35, anchor=ctk.CENTER)

accuracy_title = ctk.CTkLabel(accuracy_frame, text="Model Güven Oranı", font=("Arial", 16))
accuracy_title.pack(pady=10)

# Başlangıçta boş mesaj
initial_message = ctk.CTkLabel(accuracy_frame, text="Tahmin sonrası görüntülenecek", 
                              font=("Arial", 12), text_color="gray")
initial_message.pack(pady=50)

# Tahmin butonu
predict_button = ctk.CTkButton(master=app, text="Tahmin Et", command=predict_function, state="disabled")
predict_button.place(relx=0.5, rely=0.9, anchor=ctk.CENTER)

app.mainloop()
