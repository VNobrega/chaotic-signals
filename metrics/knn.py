import numpy as np
from sklearn.neighbors import NearestNeighbors


"""
 Erro de predição NÃO-LINEAR (kNN) de 1 passo em espaço de estados.
 Procedimento:
   - Constrói embedding de dimensão m (defasagens τ).
   - Para cada estado, pega k vizinhos e usa a média do sucessor deles como previsão.
   - Calcula RMSE entre y verdadeiro e y previsto.
 Em WGC, a estrutura dinâmica pode favorecer esse tipo de predição vs. WGN.
"""
 
def nonlinear_prediction_rmse(x, m=3, tau=1, k=10):
    x = np.asarray(x, float)
    T = len(x) - (m * tau)             # nº de amostras válidas para (estado, alvo)
    if T <= 50:                        # série muito curta: retorna NaN
        return np.nan
    # Estados (colunas são atrasos) e alvo (amostra seguinte)
    X = np.column_stack([x[i : i+T] for i in range(0, m*tau, tau)])
    y = x[m*tau : m*tau + T]          # alvo: x no passo à frente relativo a X
    # Ajusta kNN (por padrão, métrica Euclidiana). n_neighbors ≤ (#amostras-1)
    nbrs = NearestNeighbors(n_neighbors=min(k, len(X) - 1), algorithm='auto').fit(X)
    _, idx = nbrs.kneighbors(X, return_distance=True)  # índices dos vizinhos (inclui o próprio em idx[:,0])
    # Previsão do próximo valor como média dos sucessores dos vizinhos
    yhat = np.zeros_like(y)
    for t in range(len(X)):
        neigh = idx[t][1:] if idx.shape[1] > 1 else idx[t]  # ignora o próprio ponto (primeiro vizinho)
        nxt = []
        for j in neigh:
            if j + 1 < len(y):          # garante que o vizinho tem sucessor definido
                nxt.append(y[j])
        yhat[t] = np.mean(nxt) if len(nxt) else y[t]  # *fallback* trivial se não houver sucessores válidos
    rmse = np.sqrt(np.mean((y - yhat)**2))            # raiz do erro quadrático médio
    return rmse


