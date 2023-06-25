import cv2
import uuid
import numpy as np
import os
import time

# Definir el nombre base para las imágenes
nombre_base = input("Ingrese el nombre del gesto: ")

# Definir el número de imágenes que se tomarán para cada nombre
num_imagenes_por_nombre = 5

# Definir el número de nombres diferentes que se utilizarán
num_nombres_diferentes = 1 #27

# Inicializar la cámara
camara = cv2.VideoCapture(0)

# Iterar a través de los nombres diferentes
for i in range(num_nombres_diferentes):
    
    # Iterar a través del número de imágenes por nombre
    for j in range(num_imagenes_por_nombre):
        print("taking photo #{}", j)
        # Esperar 1 segundo antes de tomar la siguiente foto
        time.sleep(1)
        
        # Capturar un cuadro de video
        ret, cuadro = camara.read()
        image_np = cv2.flip(np.array(cuadro), 1)

        # Generar un UUID para la imagen
        id_unica = str(uuid.uuid4())

        path = f"images/espacio/"

        if not os.path.exists(path):
            os.makedirs(path)

        # Guardar la imagen con el nombre formato <nombre.uuid>
        cv2.imwrite(f"{path}/{nombre_base}.{id_unica}.jpg", cuadro)

    # Cambiar el nombre base para el siguiente conjunto de imágenes
    nombre_base = input("Ingrese el nombre del gesto: ")
    
# Liberar la cámara y cerrar todas las ventanas
camara.release()
cv2.destroyAllWindows()
