# Machine Learning na Saúde: Diagnóstico de Câncer de Mama

Aplicação interativa em Streamlit que demonstra, com dados reais, técnicas de Aprendizagem Supervisionada e Não Supervisionada para auxiliar no diagnóstico de tumores mamários utilizando o dataset Breast Cancer Wisconsin (Diagnostic) do scikit-learn.

## Sumário

- Visão Geral
- Estrutura do Projeto
- Funcionalidades
- Requisitos
- Como Executar
- Técnicas e Modelos
- Resultados
- Aviso
- Licença

## Visão Geral

- Classifica tumores como benignos ou malignos a partir de características nucleares das células.
- Demonstra ML supervisionado (Regressão Logística) e não supervisionado (PCA + K-Means).
- Interface interativa para simular diagnósticos e visualizar padrões nos dados.

## Estrutura do Projeto

```
data/
  breast_cancer_data.csv          # Dataset processado a partir do scikit-learn
  kmeans_comparison.csv           # Resultado do clustering para comparação
  metrics.json                    # Métricas do modelo (test set)
docs/
  Atividade Final - Machine Learning Aplicado à Saúde.md
  Roteiro para Vídeo de Apresentação (5 a 8 minutos).md
models/
  logistic_regression_model.pkl   # Modelo de classificação (standalone)
  model_pipeline.pkl              # Pipeline (StandardScaler + modelo)
  pca.pkl                         # PCA treinado (2 componentes p/ visualização)
  scaler.pkl                      # Scaler treinado (se usar standalone)
src/
  app.py                          # Aplicação Streamlit principal
  eda.py                          # Análise exploratória e geração de gráficos/base
  models.py                       # Treino, avaliação e geração de artefatos
visualizations/
  correlation_heatmap.png         # Matriz de correlação (top 10 features)
  mean_radius_boxplot.png         # Boxplot do mean radius por diagnóstico
  pca_kmeans_visualization.png    # Projeção PCA com clusters K-Means
  target_distribution.png         # Distribuição das classes
requirements.txt                  # Dependências do projeto
README.md                         # Este arquivo
```

## Funcionalidades

### 1) Análise Exploratória de Dados (EDA)
- Estatísticas descritivas das 30 features do dataset.
- Distribuição das classes (benigno vs. maligno) e matriz de correlação.
- Explorações interativas: escolha uma feature e visualize histograma, boxplot ou violin por diagnóstico.

### 2) Classificação (Supervisionada)
- Modelo: Regressão Logística com StandardScaler (via pipeline salvo) e versão standalone.
- Sliders na página principal para ajustar até 10 features “mean”.
- Probabilidade prevista exibida com métricas reais do teste (accuracy, F1 por classe) carregadas de `data/metrics.json`.
- Gráfico de barras com as probabilidades da previsão atual.
- Importância de features: top 10 coeficientes do modelo (positivo tende a benigno, negativo a maligno).

### 3) Agrupamento (Não Supervisionada)
- PCA para redução de dimensionalidade (2D) e visualização.
- K-Means com k=2 e comparação com rótulos reais.
- Distribuição dos clusters e heatmap da tabela de contingência (diagnóstico vs. cluster).

## Requisitos

- Python 3.8+ e `pip`
- Opcional: ambiente virtual (`venv`)

## Como Executar

1) Criar e ativar ambiente virtual (opcional, recomendado)

- Windows (PowerShell):
```
python -m venv .venv
.venv\Scripts\Activate.ps1
# Se houver erro de execução de script:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

- Linux/Mac:
```
python3 -m venv .venv
source .venv/bin/activate
```

2) Instalar dependências
```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3) Gerar dados e artefatos (visuais e modelos)
```
python src/eda.py
python src/models.py
```

4) Executar a aplicação Streamlit
```
python -m streamlit run src/app.py
```
Abre em: http://localhost:8501

Observação: os caminhos são resolvidos com `pathlib` a partir da raiz do projeto; você pode rodar os scripts a partir da raiz sem problemas de diretório de trabalho.

## Técnicas e Modelos

- Regressão Logística (solver `lbfgs`, `max_iter=1000`) para classificação.
- StandardScaler para normalização.
- Pipeline salvo (`models/model_pipeline.pkl`) para inferência consistente.
- PCA (2 componentes) para visualização e K-Means (k=2) para agrupamento.

## Resultados (exemplos)

- Accuracy ~0.98 no conjunto de teste (pode variar por ambiente/semente).
- F1-Score alto para ambas as classes.
- K-Means produz clusters alinhados aos rótulos reais (visualizado via PCA).

As métricas exibidas no app são lidas de `data/metrics.json`, gerado por `src/models.py`.

## Aviso

Este projeto é educativo e demonstrativo. Não deve ser utilizado como ferramenta de diagnóstico médico real. Consulte sempre profissionais de saúde qualificados.

## Licença

Uso educacional. Sinta-se livre para utilizar como referência de aprendizado.

