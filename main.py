import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import mlflow
import mlflow.keras

# Load model yang sudah dilatih
model = load_model('best_densenet_model_001.h5')

# File CSV yang berisi path gambar dan label
csv_file = 'test-preprocessed2.csv'

# Ukuran input gambar
image_size = (256, 256)

def classify_image(image_path):
    # Load dan preprocess gambar
    image = load_img(image_path, target_size=image_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Tambahkan batch dimension
    
    # Prediksi kelas gambar
    prediction = model.predict(image)
    
    # Menentukan kelas dengan nilai probabilitas tertinggi
    predicted_class = np.argmax(prediction, axis=1)
    
    return predicted_class[0]

def classify_images_from_csv(csv_file):
    # Membaca CSV ke dalam DataFrame
    df = pd.read_csv(csv_file)
    
    results = {}
    
    # Memulai run di MLFlow
    with mlflow.start_run():
        # Log parameter seperti ukuran input gambar
        mlflow.log_param("image_size", image_size)

        # Iterasi melalui setiap baris dalam CSV
        for index, row in df.iterrows():
            image_path = row['path']
            actual_label = row['class']
            
            if os.path.exists(image_path):
                predicted_class = classify_image(image_path)
                
                # Simpan hasil prediksi dan label sebenarnya
                results[image_path] = {
                    'predicted_class': predicted_class,
                    'actual_label': actual_label
                }
                
                print(f"Image: {image_path}, Predicted Class: {predicted_class}, Actual Label: {actual_label}")
                
                # Log hasil prediksi sebagai metric
                mlflow.log_metric(f"predicted_class_{index}", predicted_class)
                mlflow.log_metric(f"actual_label_{index}", actual_label)
            else:
                print(f"Image path {image_path} does not exist.")
    
    return results

# Bagian di bawah ini ditempatkan di bagian paling akhir script
if __name__ == '__main__':
    # Tentukan tracking URI untuk MLFlow server di localhost
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Melakukan klasifikasi untuk semua gambar berdasarkan data di CSV
    classification_results = classify_images_from_csv(csv_file)
