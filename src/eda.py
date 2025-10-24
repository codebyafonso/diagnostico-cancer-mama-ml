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
sns.countplot(x="target_names", data=df)
plt.title("Distribuicao da Classe (Maligno vs. Benigno)")
plt.xlabel("Diagnostico")
plt.ylabel("Contagem")
plt.savefig(VIS_DIR / "target_distribution.png")
plt.close()

# 4. Correlation analysis
corr_matrix = df.drop(columns=["target_names"]).corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix.iloc[:10, :10], annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Matriz de Correlacao (Top 10 Features)")
plt.savefig(VIS_DIR / "correlation_heatmap.png")
plt.close()

# 5. Key feature distribution (mean radius by target)
plt.figure(figsize=(10, 6))
sns.boxplot(x="target_names", y="mean radius", data=df)
plt.title('Distribuicao de "mean radius" por Diagnostico')
plt.xlabel("Diagnostico")
plt.ylabel("mean radius")
plt.savefig(VIS_DIR / "mean_radius_boxplot.png")
plt.close()

print("\nEDA concluida. Arquivos salvos na pasta 'visualizations/'.")

# Save processed dataframe for later stages
df.to_csv(DATA_DIR / "breast_cancer_data.csv", index=False)
print("DataFrame salvo em 'data/breast_cancer_data.csv'.")

