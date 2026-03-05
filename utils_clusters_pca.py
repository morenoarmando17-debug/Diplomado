import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from scipy.cluster.hierarchy import dendrogram, linkage


def evaluate_clustering(X, labels, name="Model"):
    """Métricas internas (sin etiquetas reales)."""
    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)

    print(f"\n=== {name} ===")
    print(f"Silhouette:         {sil:.4f}  (↑ mejor)")
    print(f"Davies-Bouldin:     {dbi:.4f}  (↓ mejor)")
    print(f"Calinski-Harabasz:  {ch:.2f}   (↑ mejor)")


def plot_clusters_2d(X2, labels, title, xlabel, ylabel):
    """Gráfica 2D (X2 debe tener 2 columnas)."""
    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=35)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()


def plot_dendrogram(X, method="ward"):
    """Dendrograma usando SciPy (con truncado para que se vea limpio)."""
    Z = linkage(X, method=method)
    plt.figure(figsize=(10, 4))
    dendrogram(Z, truncate_mode="level", p=5)
    plt.title(f"Dendrograma (linkage='{method}') [truncado]")
    plt.xlabel("Índices / clusters combinados")
    plt.ylabel("Distancia")
    plt.tight_layout()
    
def reconstruction_error_mse(X, X_reconstructed) -> float:
    """Error de reconstrucción (MSE) promedio por elemento."""
    return float(np.mean((X - X_reconstructed) ** 2))