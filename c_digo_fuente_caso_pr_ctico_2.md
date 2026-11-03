"""
Caso Práctico 2: Procesamiento y Convolución 2D de Imágenes Satelitales
Materia: Cómputo de Alto Rendimiento

Este script evalúa el impacto de la localidad de memoria y el control de concurrencia
en el procesamiento de una matriz 2D (simulación de imagen satelital).
Se aplica la metodología PCAM para la paralelización.
"""

import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# CONFIGURACIÓN GENERAL Y VARIABLES GLOBALES
# ==========================================
WIDTH = 2000
HEIGHT = 2000
NUM_THREADS = 4

# Generación de la imagen satelital simulada (Matriz 2D)
print(f"Generando imagen simulada de {WIDTH}x{HEIGHT}...")
image_in = np.random.rand(HEIGHT, WIDTH)
image_out = np.zeros((HEIGHT, WIDTH))

# Variables compartidas para reducción
global_brightness = 0.0
brightness_lock = threading.Lock()

def benchmark(name):
    """Decorador para medir el tiempo de ejecución y reiniciar el estado global."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            global image_out, global_brightness
            image_out = np.zeros((HEIGHT, WIDTH)) # Limpieza del buffer
            global_brightness = 0.0
            
            start = time.time()
            func(*args, **kwargs)
            end = time.time()
            
            print(f"[{name}] Tiempo: {end - start:.4f} s | Brillo: {global_brightness:.2f}")
        return wrapper
    return decorator

# ==========================================
# 1. CÓDIGO BASE SECUENCIAL (INEFICIENTE)
# ==========================================
@benchmark("1. Base Secuencial (Por Columnas)")
def process_col_major():
    """
    Problema de Memoria: Recorre la matriz iterando primero por el eje X y luego Y.
    Esto genera continuos fallos de caché (Cache Misses) al no aprovechar el 
    almacenamiento en memoria contigua (Row-Major Order de NumPy/C).
    """
    global global_brightness
    for x in range(1, WIDTH - 1):
        for y in range(1, HEIGHT - 1):
            val = np.sum(image_in[y-1:y+2, x-1:x+2]) / 9.0
            image_out[y, x] = val
            global_brightness += val

# ==========================================
# 2. LOCALIDAD DE MEMORIA OPTIMIZADA
# ==========================================
@benchmark("2. Localidad Optimizada (Por Filas)")
def process_row_major():
    """
    Optimización de Memoria (Localidad Espacial):
    Se invierte el anidamiento. Al recorrer primero Y (filas) y luego X (columnas),
    se lee la memoria de forma secuencial. El procesador aprovecha el Cache Line 
    Prefetching cargando bloques de 64 bytes a la caché L1/L2.
    """
    global global_brightness
    for y in range(1, HEIGHT - 1):
        for x in range(1, WIDTH - 1):
            val = np.sum(image_in[y-1:y+2, x-1:x+2]) / 9.0
            image_out[y, x] = val
            global_brightness += val

# ==========================================
# 3. CÓDIGO CONCURRENTE CON BLOQUEO INGENUO
# ==========================================
@benchmark("3. Bloqueo Ingenuo (Global Lock)")
def process_naive_lock():
    """
    Problema de Concurrencia (Lock Contention):
    Aunque particiona el trabajo (PCAM), utiliza un cerrojo (Lock) global por CADA
    píxel calculado. Esto serializa la ejecución y el Lock Overhead ahoga el CPU,
    haciendo que el algoritmo sea mucho más lento que el secuencial.
    """
    def process_strip(start_y, end_y):
        global global_brightness
        for y in range(start_y, end_y):
            for x in range(1, WIDTH - 1):
                val = np.sum(image_in[y-1:y+2, x-1:x+2]) / 9.0
                image_out[y, x] = val
                
                # Cuello de botella severo:
                with brightness_lock:
                    global_brightness += val

    chunk = (HEIGHT - 2) // NUM_THREADS
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as exec:
        for i in range(NUM_THREADS):
            start = 1 + i * chunk
            end = HEIGHT - 1 if i == NUM_THREADS - 1 else start + chunk
            exec.submit(process_strip, start, end)

# ==========================================
# 4. SOLUCIÓN OPTIMIZADA FINAL (PCAM)
# ==========================================
@benchmark("4. Solución Final Optimizada")
def process_optimized():
    """
    Implementación correcta de PCAM y Exclusión Mutua:
    - Particionamiento/Agrupamiento: La imagen se divide en franjas horizontales.
    - Mapeo: Cada franja va a un hilo distinto.
    - Prevención de Condiciones de Carrera: Uso de variables locales privadas
      por hilo (local_brightness) para evitar colisiones.
    - Exclusión Mutua Oportuna: El candado global solo se adquiere 1 vez por hilo
      al finalizar toda su carga de trabajo (Reducción).
    """
    def process_strip_opt(start_y, end_y):
        global global_brightness
        local_brightness = 0.0 # Variable privada en registro/memoria local
        
        # Localidad de memoria preservada (Y -> X)
        for y in range(start_y, end_y):
            for x in range(1, WIDTH - 1):
                val = np.sum(image_in[y-1:y+2, x-1:x+2]) / 9.0
                image_out[y, x] = val
                local_brightness += val
                
        # Consolidación de métricas globales de forma segura
        with brightness_lock:
            global_brightness += local_brightness

    chunk = (HEIGHT - 2) // NUM_THREADS
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as exec:
        for i in range(NUM_THREADS):
            start = 1 + i * chunk
            end = HEIGHT - 1 if i == NUM_THREADS - 1 else start + chunk
            exec.submit(process_strip_opt, start, end)


if __name__ == "__main__":
    print("\n--- INICIANDO BENCHMARKS ---")
    process_col_major()
    process_row_major()
    
    # ATENCIÓN: Se comenta el escenario 3 porque su diseño ineficiente hace
    # que tarde demasiado tiempo en procesar matrices grandes.
    # Descomentar bajo propio riesgo para visualizar el efecto en la matriz de rendimiento.
    # process_naive_lock() 
    
    process_optimized()
    print("--- BENCHMARKS FINALIZADOS ---")