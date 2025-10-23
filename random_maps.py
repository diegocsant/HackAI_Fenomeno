import random
import os

# ========== PARÁMETROS ==========
DIMENSION = 250             # Tamaño del mapa (MxM)
NUM_CASAS = 5000             # Número de casas (O)
NUM_CARRETERAS = 1000       # Número de carreteras (X)
# ================================

def generar_mapa(dimension: int, num_casas: int, num_carreteras: int) -> list:
    """
    Genera un mapa aleatorio con la dimensión y elementos especificados.
    
    Args:
        dimension: Tamaño del mapa (MxM)
        num_casas: Número de casas a colocar
        num_carreteras: Número de carreteras a colocar
    
    Returns:
        Lista de listas representando el mapa
    """
    # Validación
    total_celdas = dimension * dimension
    if num_casas + num_carreteras > total_celdas:
        raise ValueError(
            f"Error: No caben {num_casas} casas y {num_carreteras} carreteras "
            f"en un mapa {dimension}x{dimension} ({total_celdas} celdas)"
        )
    
    # Inicializar mapa vacío
    mapa = [['-' for _ in range(dimension)] for _ in range(dimension)]
    
    # Generar todas las posiciones disponibles
    posiciones = [(i, j) for i in range(dimension) for j in range(dimension)]
    random.shuffle(posiciones)
    
    # Colocar casas
    for i in range(num_casas):
        fila, col = posiciones.pop()
        mapa[fila][col] = 'O'
    
    # Colocar carreteras
    for i in range(num_carreteras):
        fila, col = posiciones.pop()
        mapa[fila][col] = 'X'
    
    return mapa


def guardar_mapa(mapa: list, nombre_archivo: str = "mapa_{DIMENSION}_{NUM_CARRETERAS}_{NUM_CASAS}.txt", carpeta: str = "entradas"):
    """Guarda el mapa en un archivo de texto"""
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, nombre_archivo)
    
    with open(ruta, 'w') as f:
        for fila in mapa:
            f.write(''.join(fila) + '\n')
    
    return ruta


def main():
    print("\n" + "="*50)
    print(" GENERADOR DE MAPAS ALEATORIOS")
    print("="*50)
    print(f" Dimensión: {DIMENSION}x{DIMENSION}")
    print(f" Casas: {NUM_CASAS}")
    print(f" Carreteras: {NUM_CARRETERAS}")
    print("="*50 + "\n")
    
    # Generar mapa
    mapa = generar_mapa(DIMENSION, NUM_CASAS, NUM_CARRETERAS)
    
    # Guardar
    nombre_salida = f"1.txt"
    ruta = guardar_mapa(mapa, nombre_salida)
    
    print(f" Mapa generado exitosamente")
    print(f" Guardado en: {ruta}\n")
    
    # Mostrar vista previa si el mapa es pequeño
    if DIMENSION <= 30:
        print("  VISTA PREVIA:")
        print("-" * (DIMENSION + 2))
        for fila in mapa:
            print('|' + ''.join(fila) + '|')
        print("-" * (DIMENSION + 2))
    else:
        print(f" Mapa demasiado grande para mostrar en consola ({DIMENSION}x{DIMENSION})")
    
    # Estadísticas
    espacios_vacios = sum(fila.count('-') for fila in mapa)
    print(f"\n Total de celdas: {DIMENSION * DIMENSION}")
    print(f"   - Espacios vacíos: {espacios_vacios}")
    print(f"   - Casas: {NUM_CASAS}")
    print(f"   - Carreteras: {NUM_CARRETERAS}\n")


if __name__ == "__main__":
    main()
