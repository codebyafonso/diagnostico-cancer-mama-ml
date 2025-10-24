from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# 0. Project directories (robust to CWD)
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
VIS_DIR = BASE_DIR / "visualizations"

DATA_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load dataset
data = load_breast_cancer(as_frame=True)
df = data.frame
df["target"] = data.target
df["target_names"] = df["target"].apply(lambda x: data.target_names[x])

# Traduções PT-BR
TARGET_PT_MAP = {
    'malignant': 'Maligno',
    'benign': 'Benigno',
}
FEATURE_LABELS_PT = {
    'mean radius': 'raio médio',
    'mean texture': 'textura média',
    'mean perimeter': 'perímetro médio',
    'mean area': 'área média',
    'mean smoothness': 'suavidade média',
    'mean compactness': 'compacidade média',
    'mean concavity': 'concavidade média',
    'mean concave points': 'pontos côncavos médios',
    'mean symmetry': 'simetria média',
    'mean fractal dimension': 'dimensão fractal média',
    'radius error': 'erro do raio',
    'texture error': 'erro da textura',
    'perimeter error': 'erro do perímetro',
    'area error': 'erro da área',
    'smoothness error': 'erro da suavidade',
    'compactness error': 'erro da compacidade',
    'concavity error': 'erro da concavidade',
    'concave points error': 'erro dos pontos côncavos',
    'symmetry error': 'erro da simetria',
    'fractal dimension error': 'erro da dimensão fractal',
    'worst radius': 'pior raio',
    'worst texture': 'pior textura',
    'worst perimeter': 'pior perímetro',
    'worst area': 'pior área',
    'worst smoothness': 'pior suavidade',
    'worst compactness': 'pior compacidade',
    'worst concavity': 'pior concavidade',
    'worst concave points': 'piores pontos côncavos',
    'worst symmetry': 'pior simetria',
    'worst fractal dimension': 'pior dimensão fractal',
}
df['target_pt'] = df['target_names'].map(TARGET_PT_MAP).fillna(df['target_names'])

# 2. Basic info
print("--- Informacoes do Dataset ---")
print(df.info())
print("\n--- Primeiras Linhas ---")
print(df.head())
print("\n--- Estatisticas Descritivas ---")
print(df.describe().T)

# 3. Target distribution
print("\n--- Distribuicao da Variavel Alvo ---")
target_counts = df["target_names"].value_counts()
print(target_counts)

# Target distribution plot
plt.figure(figsize=(6, 4))
sns.countplot(x="target_pt", data=df)
plt.title("Distribuição da Classe (Maligno vs. Benigno)")
plt.xlabel("Diagnóstico")
plt.ylabel("Contagem")
plt.savefig(VIS_DIR / "target_distribution.png")
plt.close()

# 4. Correlation analysis
features = list(data.feature_names)
corr_matrix = df[features].corr()

plt.figure(figsize=(12, 10))
top_idx = list(range(10))
pt_labels = [FEATURE_LABELS_PT.get(features[i], features[i]) for i in top_idx]
ax = sns.heatmap(corr_matrix.iloc[:10, :10], annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
ax.set_xticklabels(pt_labels, rotation=45, ha='right')
ax.set_yticklabels(pt_labels, rotation=0)
plt.title("Matriz de Correlação (Top 10 características)")
plt.savefig(VIS_DIR / "correlation_heatmap.png")
plt.close()

# 5. Key feature distribution (mean radius by target)
plt.figure(figsize=(10, 6))
sns.boxplot(x="target_pt", y="mean radius", data=df)
plt.title('Distribuição de "raio médio" por Diagnóstico')
plt.xlabel("Diagnóstico")
plt.ylabel("raio médio")
plt.savefig(VIS_DIR / "mean_radius_boxplot.png")
plt.close()

print("\nEDA concluida. Arquivos salvos na pasta 'visualizations/'.")

# Save processed dataframe for later stages
df.to_csv(DATA_DIR / "breast_cancer_data.csv", index=False)
print("DataFrame salvo em 'data/breast_cancer_data.csv'.")
