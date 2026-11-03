import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# CONFIGURACIÓN DEL CASO PRÁCTICO 2
# ==========================================
WIDTH = 2000
HEIGHT = 2000
NUM_THREADS = 4
# Simulando la imagen satelital de entrada
image_in = np.random.rand(HEIGHT, WIDTH)
image_out = np.zeros((HEIGHT, WIDTH))
global_brightness = 0.0
brightness_lock = threading.Lock()

def benchmark(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            global image_out, global_brightness
            image_out = np.zeros((HEIGHT, WIDTH)) # Limpiar el buffer de salida
            global_brightness = 0.0
            start = time.time()
            func(*args, **kwargs)
            end = time.time()
            print(f"[{name}] Tiempo de ejecución: {end - start:.4f} segundos")
        return wrapper
    return decorator

# ---------------------------------------------------------
# ESCENARIO 1: Código Base Secuencial (Ineficiente)
# ---------------------------------------------------------
@benchmark("1. Código Secuencial Ineficiente (Por Columnas)")
def process_col_major():
    # Rompe la localidad espacial saltando por memoria (Cache Misses)
    global global_brightness
    for x in range(1, WIDTH - 1):
        for y in range(1, HEIGHT - 1):
            val = np.sum(image_in[y-1:y+2, x-1:x+2]) / 9.0
            image_out[y, x] = val
            global_brightness += val

# ---------------------------------------------------------
# ESCENARIO 2: Localidad Optimizada
# ---------------------------------------------------------
@benchmark("2. Localidad de Memoria Optimizada (Por Filas)")
def process_row_major():
    # Recorrido secuencial contiguo (Aprovecha la caché L1/L2)
    global global_brightness
    for y in range(1, HEIGHT - 1):
        for x in range(1, WIDTH - 1):
            val = np.sum(image_in[y-1:y+2, x-1:x+2]) / 9.0
            image_out[y, x] = val
            global_brightness += val

# ---------------------------------------------------------
# ESCENARIO 3: Concurrente con Bloqueo Ingenuo
# ---------------------------------------------------------
@benchmark("3. Concurrente con Bloqueo Ingenuo")
def process_naive_lock():
    def process_strip(start_y, end_y):
        global global_brightness
        for y in range(start_y, end_y):
            for x in range(1, WIDTH - 1):
                val = np.sum(image_in[y-1:y+2, x-1:x+2]) / 9.0
                image_out[y, x] = val
                # Cuello de botella: Sincronización continua en variable compartida
                with brightness_lock:
                    global_brightness += val

    chunk = (HEIGHT - 2) // NUM_THREADS
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as exec:
        for i in range(NUM_THREADS):
            start = 1 + i * chunk
            end = HEIGHT - 1 if i == NUM_THREADS - 1 else start + chunk
            exec.submit(process_strip, start, end)

# ---------------------------------------------------------
# ESCENARIO 4: Solución Optimizada Final
# ---------------------------------------------------------
@benchmark("4. Solución Optimizada Final (Franjas + Reducción)")
def process_optimized():
    def process_strip_opt(start_y, end_y):
        global global_brightness
        local_brightness = 0.0 # Variable local privada por hilo
        
        for y in range(start_y, end_y):
            for x in range(1, WIDTH - 1):
                val = np.sum(image_in[y-1:y+2, x-1:x+2]) / 9.0
                image_out[y, x] = val
                local_brightness += val
                
        # Exclusión mutua únicamente al final de la franja (Reduce Lock Overhead)
        with brightness_lock:
            global_brightness += local_brightness

    chunk = (HEIGHT - 2) // NUM_THREADS
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as exec:
        for i in range(NUM_THREADS):
            start = 1 + i * chunk
            end = HEIGHT - 1 if i == NUM_THREADS - 1 else start + chunk
            exec.submit(process_strip_opt, start, end)

if __name__ == "__main__":
    print("Iniciando Benchmarks del Caso Práctico 2 (Imágenes)...")
    process_col_major()
    process_row_major()
    
    # PRECAUCIÓN: El escenario 3 se comenta porque puede tardar MUCHÍSIMO en ejecutarse.
    # Si quieres medirlo para tu tabla, descoméntalo borrando el '#' al inicio de la siguiente línea:
    # process_naive_lock() 
    
    process_optimized()
