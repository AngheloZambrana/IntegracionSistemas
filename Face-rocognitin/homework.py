import os
import time
from PIL import Image
import numpy as np
import face_recognition

# Configuración de carpetas
INPUT_FOLDER = "input_images"
OUTPUT_FOLDER = "output_faces"
POLL_INTERVAL = 5  # En segundos

# Crear carpetas si no existen
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def process_images():
    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(INPUT_FOLDER, filename)

            try:
                image = face_recognition.load_image_file(input_path)

                face_locations = face_recognition.face_locations(image)

                if not face_locations:
                    print(f"No se encontraron rostros en {filename}")
                    continue

                for i, face_location in enumerate(face_locations):
                    top, right, bottom, left = face_location
                    face_image = image[top:bottom, left:right]

                    pil_image = Image.fromarray(face_image)
                    output_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}_face_{i + 1}.jpg")
                    pil_image.save(output_path)

                print(f"Procesado: {filename} - {len(face_locations)} rostros extraídos.")

                os.remove(input_path)

            except Exception as e:
                print(f"Error al procesar {filename}: {e}")


if __name__ == "__main__":
    print("Servicio de extracción de rostros iniciado...")

    while True:
        process_images()
        time.sleep(POLL_INTERVAL)
