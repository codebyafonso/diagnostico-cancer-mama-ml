import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import sys

# --- Configuração Inicial ---
st.set_page_config(
    page_title="ML na Saude: Diagnostico de Cancer de Mama",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Diretórios robustos
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
VIS_DIR = BASE_DIR / "visualizations"

# Carregar o dataset completo para a seção EDA
@st.cache_data
def load_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    df['target'] = data.target
    df['target_names'] = df['target'].apply(lambda x: data.target_names[x])
    return df, list(data.feature_names)

@st.cache_resource
def load_artifacts():
    # Tenta carregar pipeline; se não existir, usa scaler + modelo
    pipeline = None
    pipeline_path = MODELS_DIR / 'model_pipeline.pkl'
    if pipeline_path.exists():
        pipeline = joblib.load(pipeline_path)

    scaler = None
    model = None
    scaler_path = MODELS_DIR / 'scaler.pkl'
    model_path = MODELS_DIR / 'logistic_regression_model.pkl'
    if scaler_path.exists() and model_path.exists():
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)

    pca_path = MODELS_DIR / 'pca.pkl'
    kmeans_comp_path = DATA_DIR / 'kmeans_comparison.csv'

    if not pca_path.exists() or not kmeans_comp_path.exists() or (pipeline is None and (scaler is None or model is None)):
        raise FileNotFoundError("Artefatos ausentes. Execute src/eda.py e src/models.py.")

    pca = joblib.load(pca_path)
    df_kmeans_comp = pd.read_csv(kmeans_comp_path)

    metrics_path = DATA_DIR / 'metrics.json'
    metrics = None
    if metrics_path.exists():
        try:
            with metrics_path.open('r', encoding='utf-8') as f:
                metrics = json.load(f)
        except Exception:
            metrics = None

    return pipeline, scaler, model, pca, df_kmeans_comp, metrics

df_full, feature_names = load_data()

# Mapas de tradução para PT-BR
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
FEATURE_LABELS_PT_INV = {v: k for k, v in FEATURE_LABELS_PT.items()}

TARGET_PT_MAP = {
    'malignant': 'Maligno',
    'benign': 'Benigno',
}

# Coluna auxiliar com rótulo da classe em PT para gráficos/legendas
if 'target_names' in df_full.columns:
    df_full['target_pt'] = df_full['target_names'].map(TARGET_PT_MAP).fillna(df_full['target_names'])

try:
    pipeline, scaler, model, pca, df_kmeans_comp, metrics = load_artifacts()
except FileNotFoundError:
    with st.spinner("Gerando artefatos (EDA e modelos) pela primeira vez..."):
        eda_script = BASE_DIR / "src" / "eda.py"
        models_script = BASE_DIR / "src" / "models.py"
        try:
            subprocess.run([sys.executable, str(eda_script)], check=True)
            subprocess.run([sys.executable, str(models_script)], check=True)
        except Exception as gen_err:
            st.error(f"Falha ao gerar artefatos automaticamente: {gen_err}")
            st.stop()
    # Tenta carregar novamente
    try:
        pipeline, scaler, model, pca, df_kmeans_comp, metrics = load_artifacts()
    except Exception as e2:
        st.error(f"Artefatos ainda ausentes ou inválidos: {e2}")
        st.stop()

# --- Título Principal ---
st.title("Machine Learning na Saúde: Diagnóstico de Câncer de Mama")
st.markdown("Este projeto demonstra ML supervisionado (classificação) e não supervisionado (PCA + K-Means) aplicados ao diagnóstico de tumores.")

# --- Sidebar para Navegação ---
st.sidebar.title("Navegação do Projeto")
page = st.sidebar.radio("Explore as Seções", [
    "1. Analise Exploratória de Dados (EDA)",
    "2. Classificacao (Supervisionada)",
    "3. Agrupamento (Nao Supervisionada)"
])

# --- SEÇÃO 1: EDA ---
if page == "1. Analise Exploratória de Dados (EDA)":
    st.header("Entendendo os Dados: Analise Exploratória")
    st.info("Equipe: Afonso Estevão Luna")

    # 1.1 Distribuição da Variável Alvo
    st.subheader("Distribuição do Diagnóstico")

    # --- Calcular valores ---
    benigno = int(df_full['target_names'].value_counts().get('benign', 0))
    maligno = int(df_full['target_names'].value_counts().get('malignant', 0))
    total = len(df_full)

    p_benigno = benigno / total * 100
    p_maligno = maligno / total * 100

    # --- Layout ---
    col1, col2, col3 = st.columns([1, 1, 1], gap="large")

    style_box = """
        text-align: center;
        padding: 18px;
        border-radius: 12px;
        background-color: #F8F9FA;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    """

    with col1:
        st.markdown(
            f"""
            <div style="{style_box}">
                <h4 style='color:#2ECC71; margin-bottom:5px;'>Benigno</h4>
                <h2 style='margin:0; font-size:2.5em;'>{benigno}</h2>
                <p style='margin:3px auto 0 auto; color:#2ECC71; font-weight:bold; font-size:1.1em; text-align:center;'>{p_benigno:.1f}%</p>
            </div>
            """, unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="{style_box}">
                <h4 style='color:#E74C3C; margin-bottom:5px;'>Maligno</h4>
                <h2 style='margin:0; font-size:2.5em;'>{maligno}</h2>
                <p style='margin:3px auto 0 auto; color:#E74C3C; font-weight:bold; font-size:1.1em; text-align:center;'>{p_maligno:.1f}%</p>
            </div>
            """, unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div style="{style_box}">
                <h4 style='color:#3498DB; margin-bottom:5px;'>Total de Amostras</h4>
                <h2 style='margin:0; font-size:2.5em;'>{total}</h2>
                <p style='margin:3px auto 0 auto; color:#3498DB; font-weight:bold; font-size:1.1em; text-align:center;'>100%</p>
            </div>
            """, unsafe_allow_html=True
        )

    # --- Imagem ---
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Container para centralizar a imagem
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(
            str(VIS_DIR / 'target_distribution.png'),
            caption='Contagem de casos Benignos e Malignos.',
            use_container_width=False,
            width=600
        )



    # 1.2 Boxplot de Feature Chave
    st.subheader("Característica Chave: Raio Médio")
    st.markdown("O raio médio é uma característica crucial. O boxplot mostra maior raio médio para tumores malignos.")
    st.image(str(VIS_DIR / 'mean_radius_boxplot.png'), caption='Boxplot de "raio médio" por Diagnóstico.')

    # 1.3 Matriz de Correlação
    st.subheader("Relação entre as Características")
    st.warning("Alta correlação entre características de dimensão (Raio, Perímetro, Área) indica redundância e oportunidade de usar PCA.")
    st.image(str(VIS_DIR / 'correlation_heatmap.png'), caption='Matriz de Correlação (Top 10 características)')

    # 1.4 Exploracoes Interativas
    st.subheader("Explorações Interativas")
    feature_options_pt = [FEATURE_LABELS_PT.get(f, f) for f in feature_names]
    feature_sel_pt = st.selectbox(
        "Escolha uma característica para explorar",
        options=feature_options_pt,
        index=0
    )
    feature_sel_en = FEATURE_LABELS_PT_INV.get(feature_sel_pt, feature_sel_pt)
    chart_type = st.radio("Tipo de gráfico", ["Histograma", "Boxplot", "Violin"], horizontal=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    if chart_type == "Histograma":
        sns.histplot(data=df_full, x=feature_sel_en, hue='target_pt', kde=True, ax=ax)
        ax.set_xlabel(feature_sel_pt)
        ax.set_ylabel('Contagem')
    elif chart_type == "Boxplot":
        sns.boxplot(data=df_full, x='target_pt', y=feature_sel_en, ax=ax)
        ax.set_xlabel('Diagnóstico')
        ax.set_ylabel(feature_sel_pt)
    else:
        sns.violinplot(data=df_full, x='target_pt', y=feature_sel_en, ax=ax, inner='quartile', cut=0)
        ax.set_xlabel('Diagnóstico')
        ax.set_ylabel(feature_sel_pt)
    st.pyplot(fig)

# --- SEÇÃO 2: APRENDIZAGEM SUPERVISIONADA ---
elif page == "2. Classificacao (Supervisionada)":
    st.header("Previsão: Modelo de Classificação")

    # Interface de Input na página principal (não na sidebar)
    st.subheader("Simulador Interativo")
    st.caption("Ajuste os parâmetros do tumor nas barras abaixo")

    cols = st.columns(2)
    input_data = {}
    for i in range(10):
        feature = feature_names[i]
        min_val = float(df_full[feature].min())
        max_val = float(df_full[feature].max())
        mean_val = float(df_full[feature].mean())
        with cols[i % 2]:
            input_data[feature] = st.slider(
                FEATURE_LABELS_PT.get(feature, feature),
                min_val,
                max_val,
                mean_val
            )

    def predict_with_artifacts(values_array: np.ndarray):
        if pipeline is not None:
            proba = pipeline.predict_proba(values_array)
            # classes do estimador final
            try:
                classes = pipeline.named_steps['clf'].classes_
            except Exception:
                # fallback para ultimo step
                classes = pipeline[-1].classes_
            return pipeline.predict(values_array), proba, classes
        # fallback scaler + modelo
        if scaler is None or model is None:
            raise RuntimeError("Artefatos de modelo ausentes.")
        values_scaled = scaler.transform(values_array)
        proba = model.predict_proba(values_scaled)
        classes = model.classes_
        return model.predict(values_scaled), proba, classes

    # Botão de Previsão
    if st.button("Obter Previsao", help="Executa o modelo de Regressao Logistica"):
        # Preparar os dados de entrada (30 features)
        full_input_array = []
        for feature in feature_names:
            if feature in input_data:
                full_input_array.append(input_data[feature])
            else:
                full_input_array.append(float(df_full[feature].mean()))
        input_array = np.array(full_input_array).reshape(1, -1)

        prediction, prediction_proba, classes = predict_with_artifacts(input_array)

        # Mapear probabilidades de forma robusta
        # No dataset, 0 = malignant, 1 = benign. Usamos classes_ para garantir.
        idx_maligno = int(np.where(classes == 0)[0][0]) if 0 in set(classes) else 0
        idx_benigno = int(np.where(classes == 1)[0][0]) if 1 in set(classes) else 1
        p_maligno = float(prediction_proba[0][idx_maligno])
        p_benigno = float(prediction_proba[0][idx_benigno])

        # Apresentar o resultado
        st.subheader("Resultado do Diagnóstico:")
        col_res, col_prob1, col_prob2 = st.columns(3)

        if int(prediction[0]) == 0:
            with col_res:
                st.error("Diagnóstico: MALIGNO (Alto Risco)")
        else:
            with col_res:
                st.success("Diagnóstico: BENIGNO (Baixo Risco)")

        with col_prob1:
            st.metric("Probabilidade Maligno", f"{p_maligno*100:.2f}%")
        with col_prob2:
            st.metric("Probabilidade Benigno", f"{p_benigno*100:.2f}%")

        # Grafico de probabilidades
        prob_df = pd.DataFrame({
            'Classe': ['Maligno', 'Benigno'],
            'Probabilidade': [p_maligno, p_benigno]
        })
        st.bar_chart(prob_df.set_index('Classe'))

    st.subheader("Performance do Modelo")
    if metrics is not None:
        acc = metrics.get('accuracy')
        rep = metrics.get('report', {})
        f1_maligno = rep.get('Maligno', {}).get('f1-score')
        f1_benigno = rep.get('Benigno', {}).get('f1-score')
        st.code(f"""
Acuracia (Accuracy): {acc:.4f}
F1-Score (Maligno): {f1_maligno:.4f}
F1-Score (Benigno): {f1_benigno:.4f}
""")
    else:
        st.info("Métricas não encontradas. Execute src/models.py para gerar metrics.json.")

    # Importancia das features (coeficientes da Regressao Logistica)
    st.subheader("Importância das Características (Coeficientes)")
    clf = None
    if pipeline is not None:
        try:
            clf = pipeline.named_steps.get('clf', None)
        except Exception:
            clf = None
        if clf is None:
            try:
                clf = pipeline[-1]
            except Exception:
                clf = None
    else:
        clf = model
    if clf is not None and hasattr(clf, 'coef_'):
        coefs = clf.coef_[0]
        coef_df = pd.DataFrame({
            'feature_en': feature_names,
            'coef': coefs,
            'abs_coef': np.abs(coefs)
        })
        coef_df['feature_pt'] = coef_df['feature_en'].map(FEATURE_LABELS_PT).fillna(coef_df['feature_en'])
        coef_df = coef_df.sort_values('abs_coef', ascending=False).head(10)
        fig_coef, ax_coef = plt.subplots(figsize=(8, 5))
        sns.barplot(y='feature_pt', x='coef', data=coef_df, ax=ax_coef, palette='vlag')
        ax_coef.set_title('Top 10 coeficientes (positivo=Benigno, negativo=Maligno)')
        ax_coef.set_xlabel('Coeficiente')
        ax_coef.set_ylabel('Característica')
        st.pyplot(fig_coef)

# --- SEÇÃO 3: APRENDIZAGEM NÃO SUPERVISIONADA ---
elif page == "3. Agrupamento (Nao Supervisionada)":
    st.header("Desvendando Padrões: PCA e K-Means")
    st.info("PCA para visualizar os dados em 2D e K-Means para descobrir agrupamentos sem usar o rotulo de diagnostico.")

    st.subheader("Visualização dos Clusters")
    st.image(str(VIS_DIR / 'pca_kmeans_visualization.png'), caption='Projeção 2D (PCA) colorida pelos clusters K-Means (k=2).')

    st.subheader("Análise do PCA: Simplificando 30 Dimensões")
    explained_variance = pca.explained_variance_ratio_
    total_variance = explained_variance.sum()
    col_pca1, col_pca2, col_pca_total = st.columns(3)
    with col_pca_total:
        st.metric("Variância Total Explicada", f"{total_variance*100:.2f}%")
    with col_pca1:
        st.metric("PC1 (Eixo Principal)", f"{explained_variance[0]*100:.2f}%")
    with col_pca2:
        st.metric("PC2 (Segundo Eixo)", f"{explained_variance[1]*100:.2f}%")

    st.markdown("- Insight: Dois componentes capturam grande parte da informacao, validando a reducao de dimensionalidade.")

    st.subheader("Coerência do K-Means com o Diagnóstico Real")
    st.markdown("O K-Means foi capaz de distinguir casos malignos e benignos sem conhecer as respostas reais.")


    target_pt_series = df_kmeans_comp['target_names'].map(TARGET_PT_MAP).fillna(df_kmeans_comp['target_names'])
    contingency_table = pd.crosstab(target_pt_series, df_kmeans_comp['Cluster'])
    if 'Maligno' in contingency_table.index and 0 in contingency_table.columns and 1 in contingency_table.columns:
        if contingency_table.loc['Maligno', 0] > contingency_table.loc['Maligno', 1]:
            contingency_table.columns = ['Cluster 0 (Majoritariamente Maligno)', 'Cluster 1 (Majoritariamente Benigno)']
        else:
            contingency_table.columns = ['Cluster 0 (Majoritariamente Benigno)', 'Cluster 1 (Majoritariamente Maligno)']
    st.dataframe(contingency_table)

    st.success("Conclusao: O K-Means agrupou os dados de forma bem alinhada ao diagnostico real, indicando separacao natural entre os tumores.")

    # Graficos adicionais de agrupamento
    st.subheader("Distribuição dos Clusters")
    cluster_counts = df_kmeans_comp['Cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)

    st.subheader("Contingência (Heatmap)")
    fig_ct, ax_ct = plt.subplots(figsize=(6, 4))
    contingency_table = pd.crosstab(target_pt_series, df_kmeans_comp['Cluster'])
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', ax=ax_ct)
    ax_ct.set_xlabel('Cluster')
    ax_ct.set_ylabel('Diagnóstico real')
    st.pyplot(fig_ct)

# --- Rodapé ---
st.sidebar.markdown("---")
