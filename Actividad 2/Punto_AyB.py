rutas = {
    'A': {'B': 5, 'C': 10, 'D': 15}, 
    'B': {'A': 5, 'C': 3, 'D': 8},
    'C': {'A': 10, 'B': 3, 'D': 6},
    'D': {'A': 15, 'B': 8, 'C': 6}
}

import heapq

def ruta_mas_corta(grafo, inicio, fin):
    # Inicializar las distancias
    distancias = {nodo: float('inf') for nodo in grafo}
    distancias[inicio] = 0
    
    # Cola de prioridad para explorar nodos
    pq = [(0, inicio)]  # (costo, nodo)
    rutas = {}
    
    while pq:
        (costo, nodo) = heapq.heappop(pq)
        
        # Si llegamos al nodo destino, retornamos la ruta
        if nodo == fin:
            ruta = []
            while nodo in rutas:
                ruta.append(nodo)
                nodo = rutas[nodo]
            ruta.append(inicio)
            return ruta[::-1]  # Retorna la ruta invertida
        
        for vecino, peso in grafo[nodo].items():
            nuevo_costo = costo + peso
            if nuevo_costo < distancias[vecino]:
                distancias[vecino] = nuevo_costo
                heapq.heappush(pq, (nuevo_costo, vecino))
                rutas[vecino] = nodo

    return None  # Si no hay ruta

# Uso de la función para encontrar la ruta más corta entre A y D
ruta = ruta_mas_corta(rutas, 'B', 'D')
print("Ruta más corta:", ruta)
