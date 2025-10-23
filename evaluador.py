from collections import deque
import os

# --- VARIABLES GLOBALES ---
NOMBRE_MAPA = "1.txt"
CANTIDAD_C = 2  # Cantidad esperada de contenedores 'C'

# --- 1. Leer mapas original y generado ---
ruta_entrada = f"entradas/{NOMBRE_MAPA}"
ruta_salida = f"salidas/{NOMBRE_MAPA}"

with open(ruta_entrada, "r") as f:
    mapa_original = [list(line.strip()) for line in f.readlines()]

with open(ruta_salida, "r") as f:
    mapa_generado = [list(line.strip()) for line in f.readlines()]

# --- 2. Verificar que O y X están en las mismas posiciones ---
def verificar_posiciones(mapa_orig, mapa_gen):
    filas_orig = len(mapa_orig)
    columnas_orig = len(mapa_orig[0]) if filas_orig > 0 else 0
    filas_gen = len(mapa_gen)
    columnas_gen = len(mapa_gen[0]) if filas_gen > 0 else 0
    
    # Verificar dimensiones
    if filas_orig != filas_gen or columnas_orig != columnas_gen:
        print(f"Error: Los mapas tienen dimensiones diferentes")
        return False
    
    # Verificar posiciones de O y X
    for i in range(filas_orig):
        for j in range(columnas_orig):
            # Si en el original hay O o X, debe ser igual en el generado
            if mapa_orig[i][j] in ['O', 'X']:
                if mapa_gen[i][j] != mapa_orig[i][j]:
                    print(f"Error en posición ({i},{j}): Original '{mapa_orig[i][j]}' vs Generado '{mapa_gen[i][j]}'")
                    return False
            # Si en el generado hay O o X, debe ser igual en el original
            if mapa_gen[i][j] in ['O', 'X']:
                if mapa_orig[i][j] != mapa_gen[i][j]:
                    print(f"Error en posición ({i},{j}): Original '{mapa_orig[i][j]}' vs Generado '{mapa_gen[i][j]}'")
                    return False
    
    return True

# --- 3. Verificar cantidad de contenedores 'C' ---
def verificar_cantidad_contenedores(mapa, cantidad_esperada):
    contenedores = 0
    for fila in mapa:
        for celda in fila:
            if celda == 'C':
                contenedores += 1
    
    if contenedores != cantidad_esperada:
        print(f"Error: Se esperaban {cantidad_esperada} contenedores 'C', pero se encontraron {contenedores}")
        return False
    
    return True

# --- 3.5 Verificar que cada 'C' tiene al menos una 'X' adyacente ---
def verificar_c_adyacente_a_x(mapa):
    filas = len(mapa)
    cols = len(mapa[0]) if filas > 0 else 0
    
    # Direcciones de las 8 casillas adyacentes (incluye diagonales)
    direcciones = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    for i in range(filas):
        for j in range(cols):
            if mapa[i][j] == 'C':
                tiene_x_adyacente = False
                
                # Verificar las 8 casillas adyacentes
                for di, dj in direcciones:
                    ni, nj = i + di, j + dj
                    # Verificar que esté dentro de los límites
                    if 0 <= ni < filas and 0 <= nj < cols:
                        if mapa[ni][nj] == 'X':
                            tiene_x_adyacente = True
                            break
                
                if not tiene_x_adyacente:
                    print(f"Error: El contenedor 'C' en posición ({i},{j}) no tiene ninguna 'X' adyacente")
                    return False
    
    return True

# --- 4. Función para calcular distancia mínima ---
def distancia_minima(mapa, start, targets):
    filas, cols = len(mapa), len(mapa[0])
    visitado = [[False]*cols for _ in range(filas)]
    queue = deque([(start[0], start[1], 0)])
    
    while queue:
        i, j, d = queue.popleft()
        if (i, j) in targets:
            return d
        if visitado[i][j]:
            continue
        visitado[i][j] = True
        
        # Moverse a las 4 direcciones (arriba, abajo, izquierda, derecha)
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i + di, j + dj
            # Solo verificar que esté dentro de los límites
            # TODAS las casillas son transitables (-, X, O, C)
            if 0 <= ni < filas and 0 <= nj < cols:
                queue.append((ni, nj, d+1))
    
    return float('inf')

# --- 5. Función distancia_total ---
def distancia_total(mapa):
    filas, cols = len(mapa), len(mapa[0])
    casas = [(i,j) for i in range(filas) for j in range(cols) if mapa[i][j] == 'O']
    contenedores = [(i,j) for i in range(filas) for j in range(cols) if mapa[i][j] == 'C']
    
    total = 0
    for casa in casas:
        total += distancia_minima(mapa, casa, contenedores)
    return total

# --- 6. Ejecutar verificaciones y cálculo ---
if (verificar_posiciones(mapa_original, mapa_generado) and 
    verificar_cantidad_contenedores(mapa_generado, CANTIDAD_C) and
    verificar_c_adyacente_a_x(mapa_generado)):
    total_pasos = distancia_total(mapa_generado)
    print(total_pasos)
else:
    print("Error: El mapa generado no cumple con los requisitos")
