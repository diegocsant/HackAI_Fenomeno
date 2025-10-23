from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
import asyncio
import numpy as np
from typing import List, Tuple, Set, Dict
import os
import time
import random
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURACIÓN ---
NOMBRE_MAPA = "1.txt"
CANTIDAD_C = 3      ### NÚMERO DE CONTENEDORES <------- ESTO ES LO ÚNICO A MODIFICAR POR EL USUARIO
API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyAP12JKH0F4o6lugb7Ow3KNsaaon77CK9Q")
NUM_WORKERS = 4

local_llm = GoogleGenAI(model="gemini-2.5-flash", api_key=API_KEY)
Settings.llm = local_llm


class AgenteHibridoLLM:
    """
    Agente que combina LLM (análisis estratégico) + Algoritmos (optimización numérica)
    """

    def __init__(self, mapa: List[List[str]], n_contenedores: int):
        self.mapa = np.array(mapa)
        self.filas, self.cols = self.mapa.shape
        self.n_contenedores = n_contenedores
        self.llm = local_llm

        # Identificar elementos
        self.casas = self._encontrar_posiciones('O')
        self.carreteras = self._encontrar_posiciones('X')
        self.espacios_vacios = self._encontrar_posiciones('-')
        self.posiciones_validas = self._calcular_posiciones_validas()
        self.lista_posiciones_validas = list(self.posiciones_validas)

        # Pre-calcular matriz de distancias
        print("Pre-calculando matriz de distancias...")
        start = time.time()
        self.matriz_distancias = self._precalcular_distancias()
        print(f"Matriz calculada en {time.time() - start:.2f}s")

        print(f"\nMapa: {self.filas}×{self.cols}")
        print(f"Casas: {len(self.casas)}")
        print(f"Carreteras: {len(self.carreteras)}")
        print(f"Posiciones válidas: {len(self.posiciones_validas)}")

    def _encontrar_posiciones(self, simbolo: str) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(self.filas)
                for j in range(self.cols) if self.mapa[i, j] == simbolo]

    def _calcular_posiciones_validas(self) -> Set[Tuple[int, int]]:
        validas = set()
        direcciones = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

        for i, j in self.espacios_vacios:
            for di, dj in direcciones:
                ni, nj = i + di, j + dj
                if (0 <= ni < self.filas and 0 <= nj < self.cols
                    and self.mapa[ni, nj] == 'X'):
                    validas.add((i, j))
                    break

        return validas

    def _precalcular_distancias(self) -> np.ndarray:
        H = np.array(self.casas, dtype=np.int32)
        S = np.array(self.lista_posiciones_validas, dtype=np.int32)
        if H.size == 0 or S.size == 0:
            self.pos_a_idx = {pos: i for i, pos in enumerate(self.lista_posiciones_validas)}
            return np.zeros((len(H), len(S)), dtype=np.int32)
        self.pos_a_idx = {tuple(pos): i for i, pos in enumerate(self.lista_posiciones_validas)}
        Hr = H[:, [0]].astype(np.int32)
        Hc = H[:, [1]].astype(np.int32)
        Sr = S[:, 0][None, :].astype(np.int32)
        Sc = S[:, 1][None, :].astype(np.int32)
        D = np.abs(Hr - Sr) + np.abs(Hc - Sc)   # (m x k)
        return D.astype(np.int32)

    def _build_assignment(self, cont_idx: List[int]):
        Dsel = self.matriz_distancias[:, cont_idx]  # (m x p)
        best_arg = Dsel.argmin(axis=1)              # 0..p-1
        best_dist = Dsel[np.arange(Dsel.shape[0]), best_arg]
        # 2º mejor con partition (O(p))
        part = np.partition(Dsel, 1, axis=1)
        second_best = part[:, 1] if Dsel.shape[1] > 1 else np.full_like(best_dist, 1<<30)
        return best_arg.astype(np.int16), best_dist.astype(np.int64), second_best.astype(np.int64)

    def _lb1(self) -> int:
        """Cota inferior barata: para cada casa, distancia al candidato válido más cercano (como si hubiera ∞ contenedores)."""
        if self.matriz_distancias.shape[1] == 0:
            return int(1e18)
        return int(self.matriz_distancias.min(axis=1).astype(np.int64).sum())

    def calcular_distancia_total_rapido(self, contenedores_idx: List[int]) -> int:
        if not contenedores_idx:
            return float('inf')
        distancias_minimas = self.matriz_distancias[:, contenedores_idx].min(axis=1)
        return int(distancias_minimas.sum())

    def _median_snap_step(self, cont_idx: List[int]) -> Tuple[List[int], int, bool]:
        """Un paso de Lloyd para L1: reasignar, mover cada centro a la mediana de su cluster y 'snap' al candidato válido más cercano."""
        if len(cont_idx) == 0:
            return cont_idx, int(1e18), False
        best_arg, best_dist, _ = self._build_assignment(cont_idx)
        total = int(best_dist.sum())

        used = set(cont_idx)
        S = np.array(self.lista_posiciones_validas, dtype=np.int32)
        new_idx = cont_idx.copy()
        changed = False

        for j in range(len(cont_idx)):
            mask = (best_arg == j)
            if not np.any(mask):
                continue
            Hj = np.array(self.casas, dtype=np.int32)[mask]
            r_med = int(np.median(Hj[:, 0]))
            c_med = int(np.median(Hj[:, 1]))
            # snap: candidato más cercano en L1
            d = np.abs(S[:, 0] - r_med) + np.abs(S[:, 1] - c_med)
            order = np.argsort(d)
            # elige el primer no usado
            chosen = None
            for idx in order[:512]:  # cap de seguridad
                ci = int(idx)
                if ci not in used:
                    chosen = ci
                    break
            if chosen is None:
                continue
            if chosen != new_idx[j]:
                changed = True
                used.discard(new_idx[j])
                used.add(chosen)
                new_idx[j] = chosen

        if changed:
            # recalcular coste
            new_total = self.calcular_distancia_total_rapido(new_idx)
            return new_idx, new_total, new_total < total
        else:
            return new_idx, total, False

    def _best_1swap(self, cont_idx: List[int], pool: np.ndarray = None, first_improvement: bool = False):
        k = len(cont_idx)
        if k == 0:
            return cont_idx, int(1e18), False

        best_arg, best_dist, second_best = self._build_assignment(cont_idx)
        base_sum = int(best_dist.sum())

        all_cands = np.arange(self.matriz_distancias.shape[1], dtype=np.int32)
        mask_not = np.ones_like(all_cands, dtype=bool)
        mask_not[cont_idx] = False
        not_chosen = all_cands[mask_not]
        if pool is not None:
            not_chosen = np.intersect1d(not_chosen, pool, assume_unique=False)

        D = self.matriz_distancias  # (m x K)
        best_delta = 0
        best_move = None

        for out_pos in range(k):
            cand_try = not_chosen
            for add_c in cand_try:
                add_col = D[:, add_c]
                # si la casa estaba servida por 'out_pos', su reserva es second_best; si no, best_dist
                tentative = np.minimum(np.where(best_arg == out_pos, second_best, best_dist), add_col)
                delta = int(tentative.sum() - base_sum)
                if delta < best_delta:
                    best_delta = delta
                    best_move = (out_pos, int(add_c))
                    if first_improvement:
                        break
            if first_improvement and best_move is not None:
                break

        if best_move is None:
            return cont_idx, base_sum, False

        out_pos, add_c = best_move
        new_idx = cont_idx.copy()
        new_idx[out_pos] = add_c
        new_cost = base_sum + best_delta
        return new_idx, new_cost, True

    def _opt_local_radius(self, cont_idx: List[int], base_cost: int, base_radius: int = 2, rounds: int = 3):
        best_idx, best_cost = cont_idx, base_cost
        S = np.array(self.lista_posiciones_validas, dtype=np.int32)
        for r in [base_radius * (i+1) for i in range(rounds)]:
            improved = False
            for pos in range(len(best_idx)):
                cur = S[best_idx[pos]]
                # pool de vecinos Manhattan <= r
                d = np.abs(S[:,0] - cur[0]) + np.abs(S[:,1] - cur[1])
                pool = np.where(d <= r)[0]
                cand = np.setdiff1d(pool, np.array(best_idx, dtype=np.int32), assume_unique=False)
                if cand.size == 0:
                    continue
                trial_idx, trial_cost, ok = self._best_1swap(best_idx, pool=cand, first_improvement=True)
                if ok and trial_cost < best_cost:
                    best_idx, best_cost = trial_idx, trial_cost
                    improved = True
            if not improved:
                break
        return best_idx, best_cost

    def _two_swap_random(self, cont_idx: List[int], trials: int = 25):
        best_idx = cont_idx
        best_cost = self.calcular_distancia_total_rapido(best_idx)
        K = self.matriz_distancias.shape[1]
        all_cands = np.arange(K, dtype=np.int32)
        not_chosen = np.setdiff1d(all_cands, np.array(best_idx, dtype=np.int32), assume_unique=False)
        for _ in range(trials):
            if len(not_chosen) < 2:
                break
            out_pos = np.random.choice(len(best_idx), size=2, replace=False)
            add_c = np.random.choice(not_chosen, size=2, replace=False)
            trial = best_idx.copy()
            trial[out_pos[0]] = int(add_c[0])
            trial[out_pos[1]] = int(add_c[1])
            cost = self.calcular_distancia_total_rapido(trial)
            if cost < best_cost:
                best_idx, best_cost = trial, int(cost)
                # refrescar not_chosen
                not_chosen = np.setdiff1d(all_cands, np.array(best_idx, dtype=np.int32), assume_unique=False)
        return best_idx, best_cost

    def _shake_and_repair(self, cont_idx: List[int], rounds: int = 3):
        best_idx = cont_idx
        best_cost = self.calcular_distancia_total_rapido(best_idx)
        K = self.matriz_distancias.shape[1]
        all_cands = np.arange(K, dtype=np.int32)

        for _ in range(rounds):
            # shake: mover 1 centro a un candidato aleatorio no usado
            not_chosen = np.setdiff1d(all_cands, np.array(best_idx, dtype=np.int32), assume_unique=False)
            if not_chosen.size == 0:
                break
            j = np.random.randint(len(best_idx))
            new_idx = best_idx.copy()
            new_idx[j] = int(np.random.choice(not_chosen))
            # repair: median-snap y 1-swap
            new_idx, _, _ = self._median_snap_step(new_idx)
            new_idx, new_cost, _ = self._best_1swap(new_idx)
            if new_cost < best_cost:
                best_idx, best_cost = new_idx, int(new_cost)
        return best_idx, best_cost

    def optimizacion_multicapa(self, contenedores_idx: List[int], base_max_iter: int = 500, radio_busqueda: int = 3):
        # capa 0: tu búsqueda local existente (rápida)
        idx0, cost0 = self.optimizacion_local_rapida(contenedores_idx, max_iter=base_max_iter, radio_busqueda=radio_busqueda)

        # capa 1: median-snap (Lloyd L1)
        idx1, cost1, _ = self._median_snap_step(idx0)
        if cost1 < cost0:
            idx0, cost0 = idx1, cost1

        # capa 2: 1-swap best-improvement (global pool)
        idx2, cost2, _ = self._best_1swap(idx0)
        if cost2 < cost0:
            idx0, cost0 = idx2, cost2

        # capa 3: VNS (radio creciente)
        idx3, cost3 = self._opt_local_radius(idx0, cost0, base_radius=max(2, radio_busqueda), rounds=3)
        if cost3 < cost0:
            idx0, cost0 = idx3, cost3

        # capa 4: 2-swap aleatorio (pocos intentos)
        idx4, cost4 = self._two_swap_random(idx0, trials=30)
        if cost4 < cost0:
            idx0, cost0 = idx4, cost4

        # capa 5: shake + repair (3 sacudidas)
        idx5, cost5 = self._shake_and_repair(idx0, rounds=3)
        if cost5 < cost0:
            idx0, cost0 = idx5, cost5

        return idx0, cost0

    # ==================== FUNCIONES LLM ====================

    async def analizar_mapa_con_llm(self) -> Dict:
        """
        FASE 1: LLM analiza la topología del mapa
        """
        print("\n FASE LLM: Analizando topología del mapa...")

        # Crear representación visual simplificada para la LLM
        mapa_sample = self._crear_muestra_visual()

        prompt = f"""
Analiza este mapa de una ciudad para optimización de contenedores:

Características:
- Dimensiones: {self.filas}×{self.cols}
- Total de casas (O): {len(self.casas)}
- Total de carreteras (X): {len(self.carreteras)}
- Contenedores a colocar: {self.n_contenedores}

Muestra del mapa (primeras 30×30 celdas si es grande):
{mapa_sample}

Analiza y responde en JSON:
{{
    "patron_casas": "describir si están en clusters, dispersas uniformemente, o en líneas",
    "densidad_casas": "baja/media/alta",
    "patron_carreteras": "red densa/líneas principales/dispersas",
    "num_clusters_recomendado": "número estimado de clusters naturales de casas",
    "estrategia_recomendada": "kmeans/greedy/grid/density_based",
    "razon": "justificación de la estrategia"
}}

Responde SOLO el JSON, sin texto adicional.
"""

        try:
            response = await self.llm.acomplete(prompt)
            # Limpiar la respuesta
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            analisis = json.loads(response_text.strip())
            print(f"    Análisis LLM completado")
            print(f"    Patrón detectado: {analisis.get('patron_casas', 'N/A')}")
            print(f"    Estrategia recomendada: {analisis.get('estrategia_recomendada', 'N/A')}")
            return analisis
        except Exception as e:
            print(f"     Error en análisis LLM: {e}")
            # Fallback a estrategia por defecto
            return {
                "estrategia_recomendada": "kmeans",
                "razon": "Fallback por error en LLM"
            }

    def _crear_muestra_visual(self, max_size: int = 30) -> str:
        """Crea una representación visual del mapa para la LLM"""
        if self.filas <= max_size and self.cols <= max_size:
            return '\n'.join([''.join(map(str, fila)) for fila in self.mapa.tolist()])
        else:
            # Muestra solo una esquina del mapa
            muestra = self.mapa[:max_size, :max_size]
            return '\n'.join([''.join(map(str, fila)) for fila in muestra.tolist()]) + "\n...(mapa continúa)"

    async def determinar_hiperparametros_con_llm(self, analisis: Dict) -> Dict:
        """
        FASE 2: LLM decide hiperparámetros basándose en el análisis
        """
        print("\n FASE LLM: Determinando hiperparámetros...")

        prompt = f"""
Basándote en este análisis de mapa:
{json.dumps(analisis, indent=2)}

Y estas características:
- Mapa: {self.filas}×{self.cols} ({self.filas * self.cols} celdas)
- Casas: {len(self.casas)}
- Contenedores: {self.n_contenedores}

Determina los mejores hiperparámetros para la optimización:

{{
    "n_starts": 3-8,
    "radio_busqueda": 2-8,
    "max_iter": 300-1000,
    "usar_candidatos_prometedores": true/false,
    "justificacion": "por qué estos valores son óptimos para este mapa"
}}

Responde SOLO el JSON.
"""

        try:
            response = await self.llm.acomplete(prompt)
            response_text = response.text.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            hiperparametros = json.loads(response_text.strip())
            print(f"    Hiperparámetros determinados:")
            print(f"    Inicios paralelos: {hiperparametros.get('n_starts', 'N/A')}")
            print(f"    Radio búsqueda: {hiperparametros.get('radio_busqueda', 'N/A')}")
            return hiperparametros
        except Exception as e:
            print(f"    Error determinando hiperparámetros: {e}")
            # Valores por defecto adaptativos
            area = self.filas * self.cols
            return {
                "n_starts": 4 if area > 40000 else 6,
                "radio_busqueda": 3 if area > 40000 else 4,
                "max_iter": 500,
                "justificacion": "Valores por defecto adaptativos"
            }

    # ==================== OPTIMIZACIÓN ALGORÍTMICA ====================

    def colocacion_inicial_kmeans(self, seed: int = None) -> List[int]:
        """K-means para colocación inicial"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if not self.casas or not self.posiciones_validas:
            return []

        k = min(self.n_contenedores, len(self.casas))
        centroides = [self.casas[np.random.randint(0, len(self.casas))]]

        # K-means++
        for _ in range(k - 1):
            distancias = np.array([
                min(abs(casa[0] - c[0]) + abs(casa[1] - c[1]) for c in centroides)
                for casa in self.casas
            ])
            distancias = distancias ** 2
            probs = distancias / (distancias.sum() + 1e-10)
            idx = np.random.choice(len(self.casas), p=probs)
            centroides.append(self.casas[idx])

        # Iterar k-means
        for _ in range(10):
            clusters = [[] for _ in range(k)]
            for casa in self.casas:
                idx_cercano = min(range(k),
                                key=lambda i: abs(casa[0] - centroides[i][0]) +
                                              abs(casa[1] - centroides[i][1]))
                clusters[idx_cercano].append(casa)

            nuevos_centroides = []
            for cluster in clusters:
                if cluster:
                    centroide = (
                        int(np.mean([c[0] for c in cluster])),
                        int(np.mean([c[1] for c in cluster]))
                    )
                    nuevos_centroides.append(centroide)
                else:
                    nuevos_centroides.append(centroides[len(nuevos_centroides)])

            if nuevos_centroides == centroides:
                break
            centroides = nuevos_centroides

        # Mapear a posiciones válidas
        contenedores_idx = []
        posiciones_usadas = set()

        for centroide in centroides:
            mejor_idx = min(
                (idx for idx in range(len(self.lista_posiciones_validas))
                 if idx not in posiciones_usadas),
                key=lambda idx: abs(self.lista_posiciones_validas[idx][0] - centroide[0]) +
                               abs(self.lista_posiciones_validas[idx][1] - centroide[1])
            )
            contenedores_idx.append(mejor_idx)
            posiciones_usadas.add(mejor_idx)

        # Completar si faltan
        while len(contenedores_idx) < self.n_contenedores:
            disponibles = [i for i in range(len(self.lista_posiciones_validas))
                          if i not in posiciones_usadas]
            if not disponibles:
                break
            mejor_idx = max(disponibles,
                          key=lambda idx: min(
                              abs(self.lista_posiciones_validas[idx][0] - self.lista_posiciones_validas[c][0]) +
                              abs(self.lista_posiciones_validas[idx][1] - self.lista_posiciones_validas[c][1])
                              for c in contenedores_idx
                          ))
            contenedores_idx.append(mejor_idx)
            posiciones_usadas.add(mejor_idx)

        return contenedores_idx

    def optimizacion_local_rapida(self, contenedores_idx: List[int],
                                  max_iter: int = 500,
                                  radio_busqueda: int = 3) -> Tuple[List[int], int]:
        """Búsqueda local optimizada"""
        if not contenedores_idx:
            return [], float('inf')

        mejor_contenedores = contenedores_idx.copy()
        mejor_distancia = self.calcular_distancia_total_rapido(mejor_contenedores)

        sin_mejora = 0
        max_sin_mejora = 30

        for _ in range(max_iter):
            mejora_encontrada = False

            for idx_contenedor in range(len(mejor_contenedores)):
                pos_actual_idx = mejor_contenedores[idx_contenedor]
                pos_actual = self.lista_posiciones_validas[pos_actual_idx]

                candidatos = [
                    i for i, pos in enumerate(self.lista_posiciones_validas)
                    if (abs(pos[0] - pos_actual[0]) <= radio_busqueda and
                        abs(pos[1] - pos_actual[1]) <= radio_busqueda and
                        i not in mejor_contenedores)
                ]

                for nuevo_idx in candidatos:
                    contenedores_prueba = mejor_contenedores.copy()
                    contenedores_prueba[idx_contenedor] = nuevo_idx
                    distancia_prueba = self.calcular_distancia_total_rapido(contenedores_prueba)

                    if distancia_prueba < mejor_distancia:
                        mejor_contenedores = contenedores_prueba
                        mejor_distancia = distancia_prueba
                        mejora_encontrada = True
                        sin_mejora = 0
                        break

                if mejora_encontrada:
                    break

            if not mejora_encontrada:
                sin_mejora += 1
                if sin_mejora >= max_sin_mejora:
                    break

        return mejor_contenedores, int(mejor_distancia)

    def optimizar_desde_inicio(self, seed: int, hiperparametros: Dict) -> Tuple[List[int], int]:
        try:
            contenedores_idx = self.colocacion_inicial_kmeans(seed=seed)
            if not contenedores_idx:
                return [], float('inf')
            # multicapa (más tiempo, más calidad)
            cont_opt, dist_opt = self.optimizacion_multicapa(
                contenedores_idx,
                base_max_iter=hiperparametros.get('max_iter', 500),
                radio_busqueda=hiperparametros.get('radio_busqueda', 3),
            )
            if not cont_opt:
                return [], float('inf')
            return cont_opt, int(dist_opt)
        except Exception:
            return [], float('inf')

    async def resolver_con_llm_guiada(self) -> Tuple[List[int], int, Dict]:
        """
        Pipeline completo: LLM guía la estrategia, algoritmos ejecutan
        """
        print("\n" + "="*60)
        print("AGENTE HÍBRIDO: LLM + ALGORITMOS")
        print("="*60)

        inicio_total = time.time()

        # Comprobaciones rápidas de factibilidad
        if len(self.casas) == 0:
            print("No hay casas: coste 0.")
            return [], 0, {'analisis_llm': {}, 'hiperparametros': {}, 'tiempo_total': 0.0}
        if len(self.lista_posiciones_validas) < self.n_contenedores:
            raise RuntimeError(f"No hay suficientes posiciones válidas: "
                               f"{len(self.lista_posiciones_validas)} < {self.n_contenedores}")

        # FASE 1: LLM analiza el mapa
        analisis = await self.analizar_mapa_con_llm()

        # FASE 2: LLM determina hiperparámetros
        hiperparametros = await self.determinar_hiperparametros_con_llm(analisis)
        n_starts = int(hiperparametros.get('n_starts', 4))
        if n_starts < 1:
            n_starts = 1

        # FASE 3: Optimización algorítmica con los parámetros de la LLM
        print(f"\n  FASE ALGORÍTMICA: Optimización paralela")
        print(f"   Ejecutando {n_starts} optimizaciones en paralelo...")
        start_time = time.time()

        mejor_contenedores: List[int] = []
        mejor_distancia: float = float('inf')
        soluciones_validas = 0

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(self.optimizar_desde_inicio, seed, hiperparametros): seed
                for seed in range(n_starts)
            }

            for i, future in enumerate(as_completed(futures), 1):
                seed = futures[future]
                try:
                    contenedores, distancia = future.result()
                except Exception as e:
                    print(f"   ✗ Worker seed={seed} falló: {e}")
                    continue

                if not contenedores or np.isinf(distancia):
                    print(f"   ✗ Worker {i}/{n_starts}: sin solución válida")
                    continue

                soluciones_validas += 1
                print(f"   ✓ Worker {i}/{n_starts}: distancia = {distancia:,}")

                if distancia < mejor_distancia:
                    mejor_distancia = distancia
                    mejor_contenedores = contenedores
                    print(f"      🌟 Nueva mejor solución!")

        # Fallback si ningún worker produjo solución
        if soluciones_validas == 0:
            print("⚠️  Ningún inicio produjo solución válida. Activando fallback determinista…")
            H = np.array(self.casas, dtype=np.int32)
            r_med = int(np.median(H[:, 0])); c_med = int(np.median(H[:, 1]))
            S = np.array(self.lista_posiciones_validas, dtype=np.int32)
            d = np.abs(S[:, 0] - r_med) + np.abs(S[:, 1] - c_med)
            order = np.argsort(d)
            base_idx = [int(x) for x in order[:self.n_contenedores]]
            try:
                base_idx, base_cost = self.optimizacion_multicapa(base_idx)
            except Exception:
                base_cost = self.calcular_distancia_total_rapido(base_idx)
            mejor_contenedores, mejor_distancia = base_idx, base_cost

        tiempo_optimizacion = time.time() - start_time
        tiempo_total = time.time() - inicio_total

        # --- Cota inferior y GAP (AHORA que mejor_distancia existe) ---
        lb1 = self._lb1()
        if not np.isfinite(mejor_distancia):
            gap_txt = "N/A"
        else:
            denom = max(1, int(mejor_distancia))
            gap_val = max(0, (int(mejor_distancia) - int(lb1))) / denom
            gap_txt = f"{100.0 * gap_val:.2f}%"

        print(f"\n{'='*60}")
        print(f"RESULTADO FINAL")
        print(f"{'='*60}")
        print(f"Distancia total: {int(mejor_distancia):,}")
        print(f"LB1 (cota inferior): {int(lb1):,}  |  GAP: {gap_txt}")
        print(f"Tiempo análisis LLM: ~{tiempo_total - tiempo_optimizacion:.2f}s")
        print(f"Tiempo optimización: {tiempo_optimizacion:.2f}s")
        print(f"Tiempo total: {tiempo_total:.2f}s")
        print(f"{'='*60}")

        return mejor_contenedores, int(mejor_distancia), {
            'analisis_llm': analisis,
            'hiperparametros': hiperparametros,
            'tiempo_total': tiempo_total,
            'lb1': int(lb1),
            'gap': gap_txt,
        }

    def idx_a_posiciones(self, contenedores_idx: List[int]) -> List[Tuple[int, int]]:
        return [self.lista_posiciones_validas[idx] for idx in contenedores_idx]

    def generar_mapa_solucion(self, contenedores_idx: List[int]) -> str:
        mapa_solucion = self.mapa.copy()
        contenedores = self.idx_a_posiciones(contenedores_idx)

        for i, j in contenedores:
            mapa_solucion[i, j] = 'C'

        return '\n'.join([''.join(map(str, fila)) for fila in mapa_solucion.tolist()])


async def main():
    ruta_entrada = f"entradas/{NOMBRE_MAPA}"

    with open(ruta_entrada, "r") as f:
        mapa = [list(line.strip()) for line in f.readlines() if line.strip()]

    os.makedirs("salidas", exist_ok=True)

    print("="*60)
    print("AGENTE HÍBRIDO: LLM + ALGORITMOS")
    print("="*60)

    # Crear agente híbrido
    agente = AgenteHibridoLLM(mapa, CANTIDAD_C)

    # Resolver con guía de LLM
    contenedores_idx, distancia_final, metadata = await agente.resolver_con_llm_guiada()

    # Generar solución
    mapa_solucion = agente.generar_mapa_solucion(contenedores_idx)
    contenedores_coords = agente.idx_a_posiciones(contenedores_idx)

    print("\n" + "="*60)
    print("MAPA SOLUCIÓN:")
    print("="*60)
    if agente.filas <= 30:
        print(mapa_solucion)
    else:
        print(f"(Mapa {agente.filas}×{agente.cols} - demasiado grande para visualizar)")
        print(f"Contenedores en: {contenedores_coords}")
    print("="*60)

    # Guardar solución
    ruta_salida = f"salidas/{NOMBRE_MAPA}"
    with open(ruta_salida, "w") as f:
        f.write(mapa_solucion)

    print(f"\nSolución guardada en: {ruta_salida}")
    print(f"Distancia total: {distancia_final:,}")


if __name__ == "__main__":
    asyncio.run(main())
