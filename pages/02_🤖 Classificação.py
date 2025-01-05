import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ---- Configuração da Página ----
st.set_page_config(
    page_title="Classificação de Desempenho Escolar", page_icon="📊", layout="wide"
)
st.title("📊 Classificação de Desempenho Escolar")
st.write("---")

# ---- Carregar o Dataset ----
st.subheader("1. Carregando e Explorando o Dataset")
try:
    df_students = pd.read_parquet("./data/new_unistudents.parquet")
    st.success("Dataset carregado com sucesso!")
    if "Change_Grades" in df_students.columns:
        st.write("**Distribuição da variável alvo (Change_Grades):**")
        distribution = df_students["Change_Grades"].value_counts().sort_index()
        st.bar_chart(distribution)
    else:
        st.warning("A variável 'Change_Grades' não foi encontrada no dataset!")

except Exception as e:
    st.error(f"Erro ao carregar o dataset: {e}")
    st.stop()

# ---- Pré-processamento ----
st.subheader("2. Pré-processamento dos Dados")
try:
    X = df_students.drop("Change_Grades", axis=1)
    y = df_students["Change_Grades"]

    # Escalar colunas numéricas
    num_cols = X.select_dtypes(include=["float64", "int64"]).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Codificar a coluna alvo
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Dividir os dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=101
    )

    st.success("Dados pré-processados com sucesso!")
    st.write(f"**Tamanho do conjunto de treino:** {X_train.shape[0]} linhas")
    st.write(f"**Tamanho do conjunto de teste:** {X_test.shape[0]} linhas")
except Exception as e:
    st.error(f"Erro no pré-processamento: {e}")
    st.stop()

# ---- Seleção de Modelos ----
st.subheader("3. Treinando e Avaliando os Modelos")
modelos = {
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=101),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=101),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=101),
}
modelo_selecionado = st.selectbox(
    "Escolha um modelo de classificação:", list(modelos.keys())
)
botao_avaliar = st.button("Treinar e Avaliar Modelo")


# ---- Função para Treinar e Avaliar ----
def treinar_e_avaliar(modelo, X_train, y_train, X_test, y_test):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Acurácia
    acuracia = accuracy_score(y_test, y_pred)
    st.write(f"**Acurácia:** {acuracia:.2f}")

    # Matriz de Confusão
    st.write("**Matriz de Confusão:**")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax,
    )
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    st.pyplot(fig)

    # Relatório de Classificação
    st.write("**Relatório de Classificação:**")
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose())

    # Importância das Variáveis (para modelos baseados em árvores)
    if hasattr(modelo, "feature_importances_"):
        st.write("**Importância das Variáveis:**")
        importances = modelo.feature_importances_
        importances_df = pd.DataFrame(
            {"Variável": X.columns, "Importância": importances}
        )
        importances_df = importances_df.sort_values(by="Importância", ascending=False)

        fig_imp, ax_imp = plt.subplots(figsize=(6, 4), dpi=120)
        sns.barplot(
            x="Importância",
            y="Variável",
            data=importances_df.head(10),
            palette="viridis",
            ax=ax_imp,
        )
        ax_imp.set_title("Top 10 Variáveis mais Importantes")
        st.pyplot(fig_imp)

        # Gráfico adicional: Distribuição da variável mais importante
        top_var = importances_df.iloc[0]["Variável"]
        st.write(f"**Distribuição da variável mais importante:** {top_var}")
        fig_dist, ax_dist = plt.subplots(
            figsize=(6, 4), dpi=120
        )  # Tamanho reduzido com DPI ajustado
        sns.histplot(df_students[top_var], kde=True, ax=ax_dist, color="purple")
        ax_dist.set_title(f"Distribuição de {top_var}")
        st.pyplot(fig_dist)


# ---- Avaliar Modelo Selecionado ----
if botao_avaliar:
    st.write(f"### Avaliação do Modelo: **{modelo_selecionado}**")
    treinar_e_avaliar(modelos[modelo_selecionado], X_train, y_train, X_test, y_test)