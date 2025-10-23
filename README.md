He mandado una carpeta con un creador de mapas por si es útil y con el propio evaluador (sin modificar)
porque me estaba dando problemas al ejecutarlo en la carpeta que nos mandaste.


Ahora sí, el README:


Agente híbrido LLM + optimizador para colocación de contenedores

DESCRIPCIÓN
Resuelve una variante del problema p-median en una rejilla con métrica Manhattan. Dado un mapa con casas (O), carreteras (X) y vacíos (-), coloca n contenedores (C) para minimizar la suma de distancias desde cada casa a su contenedor más cercano. Los contenedores deben situarse en celdas vacías adyacentes (8 vecinos) a alguna carretera.

El agente combina:

    LLM (Gemini 2.5-flash vía LlamaIndex) para analizar el mapa y proponer hiperparámetros.

    Optimización algorítmica determinista y vectorizada con varias capas de mejora.


REQUISITOS

    Python 3.10 o superior

    Instalar dependencias con: pip install -r requirements.txt
    (requirements.txt incluye: numpy, llama-index, google-generativeai)

    Configurar la clave de Gemini en la variable de entorno GOOGLE_API_KEY


ENTRADA Y SALIDA

    Entrada: archivo de texto en el directorio “entradas/” con nombre NOMBRE_MAPA (por defecto 1.txt).
    Cada línea es una fila del mapa y solo contiene los caracteres O, X y -.

    Salida: mapa solución en “salidas/NOMBRE_MAPA”, con los contenedores marcados como C.


PARÁMETROS BÁSICOS

    NOMBRE_MAPA: nombre del fichero de entrada (por defecto 1.txt).

    CANTIDAD_C: número de contenedores a colocar.


EJECUCIÓN

    Editar NOMBRE_MAPA y CANTIDAD_C en el script si es necesario.

    Ejecutar: python AgenteFenomeno.py


FUNCIONAMIENTO POR FASES

    Fase 0 — Carga y preprocesado

        Lee el mapa desde entradas/NOMBRE_MAPA.

        Extrae la lista de casas (H), carreteras y vacíos.

        Genera el conjunto de candidatos válidos S: todas las celdas “-” con al menos una “X” en sus 8 vecinos.

        Precalcula y almacena en forma vectorizada la matriz de distancias Manhattan D de tamaño |H| × |S|.
        Esta matriz permite evaluar rápidamente el coste de cualquier conjunto de contenedores.

    Fase 1 — Análisis con LLM

        Se envía al LLM una muestra representativa del mapa (si es grande, una ventana 30×30) y metadatos (tamaño, número de casas, etc.).

        El LLM devuelve un análisis en JSON (patrones, densidad, estrategia sugerida).

        Este análisis es informativo y sirve para elegir hiperparámetros; no condiciona la validez final.

    Fase 2 — Selección de hiperparámetros con LLM

        Con el análisis anterior y las características del mapa, el LLM propone:
        n_starts (número de inicios), radio_busqueda (vecindario local), max_iter, etc.

        Si falla la llamada al LLM, se aplican valores por defecto adaptativos.

    Fase 3 — Optimización paralela multi-start

        Se lanzan varias corridas en paralelo (n_starts).

        Cada corrida:

            Inicialización (k-means++ sobre casas) y proyección “snap” al candidato válido más cercano.

            Optimización multicapa:

                Capa 0: búsqueda local rápida en un vecindario de radio fijo por contenedor.

                Capa 1: paso de mediana L1 + snap (recolocar cada contenedor en la mediana de su clúster y proyectar a un candidato válido).

                Capa 2: 1-swap (mejor mejora) con evaluación de deltas en O(|H|) usando best y second-best por casa.

                Capa 3: VNS (vecindarios de radio creciente) para explorar sin disparar el coste.

                Capa 4: 2-swap aleatorio con pocos intentos para escapar de mínimos locales.

                Capa 5: shake + repair (perturbar una solución y repararla con mediana + 1-swap).

        Cada corrida devuelve su mejor conjunto de contenedores y su coste.

    Fase 4 — Selección de la mejor solución y métricas

        Se selecciona la mejor corrida (menor coste total).

        Se calcula una cota inferior LB1: para cada casa, distancia al candidato válido más cercano (como si hubiera infinitos contenedores). LB1 es siempre menor o igual que el óptimo.

        Se informa el GAP = (UB − LB1) / UB, donde UB es el coste de la mejor solución. Un GAP pequeño (≤ 5 %) indica solución near-optimal.

    Fase 5 — Salida

       Se pinta el mapa final con los contenedores C y se guarda en salidas/NOMBRE_MAPA.

        En consola se imprimen: coste total, LB1, GAP, tiempo de análisis LLM y tiempo de optimización.


DETALLES DE DISEÑO Y COMPLEJIDAD

    La evaluación de una solución con p contenedores se hace con D[:, contenedores] y un mínimo por filas, coste aproximado O(|H|·p), muy eficiente para p pequeño (2–10).

    La construcción de D es vectorizada con NumPy para acelerar en mapas grandes.

    Las capas de mejora (1-swap, VNS, 2-swap, shake) limitan el espacio de búsqueda a vecindarios prometedores, manteniendo el tiempo razonable.


COMPROBACIONES Y FALLBACKS

    Si no hay casas, el coste es 0 y se corta el flujo.

    Si no hay suficientes candidatos válidos para colocar n contenedores, se informa error claro.

    Si todas las corridas fallan, se aplica un fallback determinista: elección de candidatos cerca de la mediana global de casas y una pasada de mejora.# HackAI_Fenomeno
