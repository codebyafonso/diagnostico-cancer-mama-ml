from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Diretórios robustos
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VIS_DIR = BASE_DIR / "visualizations"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

# Carregar dataset
df = pd.read_csv(DATA_DIR / 'breast_cancer_data.csv')

# Separar features e target
X = df.drop(columns=['target', 'target_names', 'target_pt'])
y = df['target']

# 1) Pré-processamento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')

# 2) Aprendizagem supervisionada: Regressão Logística
print("--- Aprendizagem Supervisionada: Regressao Logistica ---")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
log_reg = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

report_txt = classification_report(y_test, y_pred, target_names=['Maligno', 'Benigno'])
print("\nRelatorio de Classificacao:\n", report_txt)

# Salvar métricas para o app
metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'report': classification_report(y_test, y_pred, target_names=['Maligno', 'Benigno'], output_dict=True),
}
with (DATA_DIR / 'metrics.json').open('w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

# Salvar modelo standalone e um pipeline (scaler + modelo)
joblib.dump(log_reg, MODELS_DIR / 'logistic_regression_model.pkl')
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')),
])
pipeline.fit(X, y)
joblib.dump(pipeline, MODELS_DIR / 'model_pipeline.pkl')

# 3) Aprendizagem não supervisionada: PCA e K-Means
print("\n--- Aprendizagem Nao Supervisionada: PCA e K-Means ---")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['target'] = y
joblib.dump(pca, MODELS_DIR / 'pca.pkl')

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df_comparison = df[['target_names']].copy()
df_comparison['Cluster'] = clusters
df_comparison.to_csv(DATA_DIR / 'kmeans_comparison.csv', index=False)

explained_variance = pca.explained_variance_ratio_
print(f"Variancia explicada pelos 2 primeiros componentes: {explained_variance.sum():.2f}")

pca_df['Cluster'] = clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', style='Cluster')
plt.title('PCA e K-Means Clustering (k=2)')
plt.xlabel(f'Componente Principal 1 ({explained_variance[0]*100:.1f}%)')
plt.ylabel(f'Componente Principal 2 ({explained_variance[1]*100:.1f}%)')
plt.grid(True)
plt.savefig(VIS_DIR / 'pca_kmeans_visualization.png')
plt.close()

print("\nModelos treinados e avaliados.")
print("Modelos salvos na pasta 'models/':")
print("- scaler.pkl, logistic_regression_model.pkl, model_pipeline.pkl, pca.pkl")
print("Dados salvos na pasta 'data/':")
print("- kmeans_comparison.csv, metrics.json")
print("Visualizacao salva na pasta 'visualizations/':")
print("- pca_kmeans_visualization.png")

