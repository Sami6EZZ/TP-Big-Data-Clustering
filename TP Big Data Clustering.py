# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

# 2. Utilisation de l'algorithme k-means

# Chargement du jeu de données Iris
iris = load_iris()
X, y = iris.data, iris.target

# print(iris)

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Essai de différentes valeurs de k
k_values = range(2, 26)
inertia_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Choix du meilleur k en utilisant la méthode du coude
plt.plot(k_values, inertia_values, marker='o')
plt.title("Méthode du coude pour choisir k")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Somme des carrés des distances intra-cluster")
plt.show()

# Choix du meilleur k en utilisant la méthode de la silhouette
best_coude_k = k_values[np.argmin(inertia_values)]
print(f"Le meilleur k selon la méthode de la coude est : {best_coude_k}")

# Choix du meilleur k en utilisant la méthode de la silhouette
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Affichage des scores de silhouette pour différentes valeurs de k
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Score de silhouette pour choisir k")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Score de silhouette moyen")
plt.show()

# Choix du meilleur k en utilisant la méthode de la silhouette
best_k = k_values[np.argmax(silhouette_scores)]
print(f"Le meilleur k selon la méthode de la silhouette est : {best_k}")


# 3. Présentation graphique des résultats avec le meilleur k
best_kmeans_coude = KMeans(n_clusters=best_coude_k, random_state=42)
best_kmeans_coude.fit(X_scaled)
cluster_labels_coude = best_kmeans_coude.predict(X_scaled)

# Calcul de l'indice de Rand ajusté pour la méthode du coude
ari_coude = adjusted_rand_score(y, cluster_labels_coude)

# Réduction de la dimension pour la visualisation (utilisation de PCA)
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# Affichage graphique des résultats avec le meilleur k en 3D (Coude)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=cluster_labels_coude, cmap='viridis', edgecolors='k')
ax.set_title(f"Résultats du clustering coude (k={best_coude_k})")
ax.set_xlabel("Composante principale 1")
ax.set_ylabel("Composante principale 2")
ax.set_zlabel("Composante principale 3")
plt.show()

# ...

# 3. Présentation graphique des résultats avec le meilleur k
best_kmeans_silhouette = KMeans(n_clusters=best_k, random_state=42)
best_kmeans_silhouette.fit(X_scaled)
cluster_labels_silhouette = best_kmeans_silhouette.predict(X_scaled)

# Calcul de l'indice de Rand ajusté pour la méthode de la silhouette
ari_silhouette = adjusted_rand_score(y, cluster_labels_silhouette)

# Réduction de la dimension pour la visualisation (utilisation de PCA)
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# Affichage graphique des résultats avec le meilleur k en 3D (Silhouette)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=cluster_labels_silhouette, cmap='viridis', edgecolors='k')
ax.set_title(f"Résultats du clustering silhouette (k={best_k})")
ax.set_xlabel("Composante principale 1")
ax.set_ylabel("Composante principale 2")
ax.set_zlabel("Composante principale 3")
plt.show()

# Affichage des résultats de l'indice de Rand ajusté
print(f"Indice de Rand ajusté (Coude): {ari_coude}")
print(f"Indice de Rand ajusté (Silhouette): {ari_silhouette}")
