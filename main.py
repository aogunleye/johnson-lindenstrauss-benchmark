import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.spatial.distance import pdist
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection, johnson_lindenstrauss_min_dim

# Experience 1 :

data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)

X_sparse = TfidfVectorizer().fit_transform(data.data)

N = 1000
X = X_sparse[:N].toarray()
D = X.shape[1]

pdist_original = pdist(X, metric='euclidean')

k_values = [10, 20, 50, 100, 250, 500, 1000, 2000, 4000, 8000]
epsilons_mesures = []
epsilons_theoriques = []
mask = pdist_original > 0 # pour éviter les erreurs de division par zero

for k in k_values:
    transformer = GaussianRandomProjection(n_components=k, random_state=42)
    X_transformed = transformer.fit_transform(X)
    pdist_transformed = pdist(X_transformed, metric='euclidean')
    ratios = pdist_transformed[mask] / pdist_original[mask]
    errors = np.abs(ratios - 1)
    eps_95 = np.percentile(errors, 95)
    epsilons_mesures.append(eps_95)
    epsilon = np.sqrt((8 * np.log(N)) / k)
    epsilons_theoriques.append(epsilon)

plt.figure(figsize=(8, 5))
plt.xscale('log')  # pour bien visualiser l'évolution sur k
plt.plot(k_values, epsilons_mesures, label='Epsilon mesuré (95e percentile)', color='blue')
plt.plot(k_values, epsilons_theoriques, label='Epsilon théorique', linestyle='--', color='red')
plt.xlabel('Dimension cible k')
plt.ylabel('Erreur de distorsion epsilon')
plt.title('Comparaison des epsilons mesurés et théoriques')
plt.legend()
plt.savefig('epsilon_comparison.png')
plt.close()


# ----------------------------------------------------------------------------

# Experience 2 :

N = 1000
k = 100
D_values = [1000, 2500, 5000, 10000, 25000, 50000]
times = {'PCA': [], 'Gaussian': [], 'Sparse': []}

for D in D_values:
    X = X_sparse[:N, :D].toarray()
    start_time = time.time()
    transformer_PCA = PCA(n_components=k, svd_solver='randomized', random_state=42)
    X_PCA = transformer_PCA.fit_transform(X)
    time_PCA = time.time() - start_time

    start_time = time.time()
    transformer_Gauss = GaussianRandomProjection(n_components=k, random_state=42)
    X_Gauss = transformer_Gauss.fit_transform(X)
    time_Gauss= time.time() - start_time

    start_time = time.time()
    transformer_Sparse = SparseRandomProjection(n_components=k, random_state=42)
    X_Sparse = transformer_Sparse.fit_transform(X)
    time_Sparse = time.time() - start_time

    print(f"D = {D}: PCA time = {time_PCA:.4f}s, Gaussian time = {time_Gauss:.4f}s, Sparse time = {time_Sparse:.4f}s")
    times['PCA'].append(time_PCA)
    times['Gaussian'].append(time_Gauss)
    times['Sparse'].append(time_Sparse)

plt.figure(figsize=(8, 5))
plt.plot(D_values, times['PCA'], label='PCA', marker='o')
plt.plot(D_values, times['Gaussian'], label='Gaussian Random Projection', marker='o')
plt.plot(D_values, times['Sparse'], label='Sparse Random Projection', marker='o')
plt.xlabel('Dimension originale D')
plt.ylabel('Temps d\'exécution en secondes')
plt.title('Comparaison des temps d\'exécution')
plt.legend()
plt.savefig('execution_time_comparison.png')
plt.close()

min_dim = johnson_lindenstrauss_min_dim(N, eps=0.1)
print(f"Dimension minimale selon le théorème de Johnson-Lindenstrauss pour N={N} et epsilon=0.1 : {min_dim}")