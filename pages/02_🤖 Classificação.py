import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# ---- Estilização ----
st.set_page_config(page_title="Classificação de Mudanças de Notas", page_icon="📊", layout="wide")
st.markdown("""
    <style>
    .big-font { font-size:24px !important; }
    .small-font { font-size:18px !important; }
    .stDataFrame { margin-top: 10px !important; }
    .main { background-color: #F0EFF4; }
    h1, h2, h3 { color: #191970; }
        div.stButton > button {
            background-color: #FCE762;
            color: #353531;
            font-size: 16px;
            border-radius: 8px;
            border: none;
        }
        footer { visibility: hidden; }
        
    </style>
""", unsafe_allow_html=True)
# ---- Título ----
st.title("📊 Classificação de Mudança de Notas dos Estudantes")
st.write("---")

# ---- Carregar o Dataset ----
st.subheader("1. Carregando e Explorando o Dataset")
try:
    df_students = pd.read_parquet("./EDA/unistudents.csv")
    st.success("Dataset carregado com sucesso!")
    st.write(df_students.head())

    # Visualizar distribuição da variável alvo
    if 'Change_Grades' in df_students.columns:
        st.write("**Distribuição da variável alvo (Change_Grades):**")
        st.bar_chart(df_students['Change_Grades'].value_counts())
    else:
        st.warning("A variável 'Change_Grades' não foi encontrada no dataset!")
except Exception as e:
    st.error(f"Erro ao carregar o dataset: {e}")

# ---- Pré-processamento ----
st.subheader("2. Pré-processamento dos Dados")
try:
    X = df_students.drop('Change_Grades', axis=1)
    y = df_students['Change_Grades']

    # Codificar colunas categóricas
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Escalar colunas numéricas
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Codificar a coluna alvo
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Dividir os dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    st.success("Dados pré-processados com sucesso!")
    st.write(f"**Tamanho do conjunto de treino:** {X_train.shape[0]} linhas")
    st.write(f"**Tamanho do conjunto de teste:** {X_test.shape[0]} linhas")
except Exception as e:
    st.error(f"Erro no pré-processamento: {e}")

# ---- Seleção do Modelo ----
st.subheader("3. Treinando e Avaliando os Modelos")
modelos = {
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=101),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=101),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=101)
}

modelo_selecionado = st.selectbox("Escolha um modelo de classificação:", list(modelos.keys()))
botao_avaliar = st.button("Treinar e Avaliar Modelo")

# ---- Função para Treinar e Avaliar ----
def treinar_e_avaliar(modelo, X_train, y_train, X_test, y_test):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Acurácia
    acuracia = accuracy_score(y_test, y_pred)
    st.success(f"**Acurácia:** {acuracia:.2f}")

    # Matriz de Confusão
    st.write("**Matriz de Confusão:**")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    st.pyplot(fig)

    # Relatório de Classificação
    st.write("**Relatório de Classificação:**")
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Importância das variáveis (para modelos baseados em árvores)
    if hasattr(modelo, 'feature_importances_'):
        st.write("**Importância das Variáveis:**")
        importances = modelo.feature_importances_
        importances_df = pd.DataFrame({'Variável': X.columns, 'Importância': importances})
        importances_df = importances_df.sort_values(by='Importância', ascending=False)

        fig_imp, ax_imp = plt.subplots()
        sns.barplot(x='Importância', y='Variável', data=importances_df.head(10), palette='viridis', ax=ax_imp)
        ax_imp.set_title("Top 10 Variáveis mais Importantes")
        st.pyplot(fig_imp)

# ---- Avaliar Modelo Selecionado ----
if botao_avaliar:
    st.write(f"### Avaliação do Modelo: **{modelo_selecionado}**")
    treinar_e_avaliar(modelos[modelo_selecionado], X_train, y_train, X_test, y_test)
