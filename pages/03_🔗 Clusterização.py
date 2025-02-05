import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import streamlit as st

# ===== Configurações de Página ===== #
st.set_page_config(
    page_title="Clusterização de Estudantes", page_icon="📊", layout="wide"
)
st.title("📊 Análise de Clusterização de Estudantes")
st.sidebar.header("Configurações")

# ===== Carregar Dataset ===== #
try:
    df_students = pd.read_parquet(
        "./data/new_unistudents_classification.parquet")
    st.success("Dataset carregado com sucesso!")

    # Imputação de valores faltantes
    imputer = SimpleImputer(strategy="mean")
    numeric_cols = df_students.select_dtypes(
        include=["float64", "int64"]).columns
    df_students[numeric_cols] = imputer.fit_transform(
        df_students[numeric_cols])

    # Normalização
    scaler = MinMaxScaler()
    df_students[numeric_cols] = scaler.fit_transform(df_students[numeric_cols])

    # Seleção de colunas para clusterização (mantido do segundo código)
    cluster_features = [
        "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
        "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores", "Motivation_Level_Low",
        "Motivation_Level_Medio", "Motivation_Level_High", "Internet_Access", "Tutoring_Sessions",
        "Family_Income", "Teacher_Quality", "School_Type", "Peer_Influence", "Physical_Activity",
        "Learning_Disabilities", "Parental_Education_Level", "Distance_from_Home", "Exam_Score"
    ]
    available_features = [
        col for col in cluster_features if col in df_students.columns]
    df_cluster = df_students[available_features]

    # ===== Método Elbow ===== #
    st.subheader("Método Elbow para Determinar o Número de Clusters")
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(df_cluster)
        inertia.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 11), inertia, marker="o")
    ax.set_title("Método Elbow para Definir k Ótimo")
    st.pyplot(fig)

    # ===== Clusterização ===== #
    optimal_k = st.sidebar.slider(
        "Número de Clusters (k)", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init=10)
    df_students["Cluster"] = kmeans.fit_predict(df_cluster)

    # ===== Resumo dos Clusters ===== #
    st.subheader("Resumo dos Clusters")
    cluster_summary = df_students.groupby(
        "Cluster")[available_features].agg(["mean", "count"])
    st.dataframe(cluster_summary)

    # ===== Seção Modificada (Caracterização por Médias do Primeiro Código) ===== #
    st.subheader("Análise Detalhada por Cluster")
    # Usando colunas numéricas do primeiro código
    analysis_features = numeric_cols.tolist()
    cluster_means = df_students.groupby(
        "Cluster")[analysis_features].mean().reset_index()

    st.write("**Médias Padronizadas por Cluster (0-1):**")
    st.dataframe(
        cluster_means.style.background_gradient(cmap="Blues").format("{:.2f}"),
        use_container_width=True
    )

    # Boxplots interativos (mantido do primeiro código)
    st.write("**Distribuição Detalhada por Característica:**")
    selected_feature = st.selectbox(
        "Selecione uma característica para análise detalhada:", analysis_features)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_students, x="Cluster",
                y=selected_feature, palette="viridis", ax=ax)
    ax.set_title(
        f"Distribuição de {selected_feature.replace('_', ' ')} por Cluster")
    st.pyplot(fig)

    # ===== Mantido do Segundo Código ===== #
    # Visualização dos Clusters
    st.subheader("Visualização dos Clusters")
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df_students["Hours_Studied"], df_students["Exam_Score"],
        c=df_students["Cluster"], cmap="viridis", alpha=0.7,
    )
    ax.scatter(
        kmeans.cluster_centers_[:, available_features.index("Hours_Studied")],
        kmeans.cluster_centers_[:, available_features.index("Exam_Score")],
        s=300, c="red", label="Centroids",
    )
    ax.set_xlabel("Horas de Estudo")
    ax.set_ylabel("Nota do Exame")
    st.pyplot(fig)

    # Cluster de Melhor Desempenho
    st.subheader("Identificação do Melhor Cluster")
    cluster_means = df_students.groupby(
        "Cluster")[["Hours_Studied", "Exam_Score"]].mean()
    sorted_clusters = cluster_means.sort_values(
        by=["Exam_Score", "Hours_Studied"], ascending=[False, True]
    )
    best_cluster = sorted_clusters.index[0]
    st.dataframe(sorted_clusters.loc[best_cluster].to_frame().T)

    # Fatores Adicionais
    st.subheader("Fatores Adicionais")
    st.write("""
    Aqui estão os fatores adicionais, como motivação, níveis de estresse e qualidade do ambiente escolar, 
    que podem ser analisados para entender melhor as características dos estudantes de cada cluster.
    """)
    additional_factors = [
        "Motivation_Level_Low", "Parental_Involvement_Low", "Parental_Involvement_Medium",
        "Extracurricular_Activities_Yes", "Sleep_Hours"
    ]

    for factor in additional_factors:
        if factor == "Sleep_Hours":
            st.write(f"Distribuição de {factor} por Cluster")
            fig, ax = plt.subplots(figsize=(8, 5))
            min_sleep = df_students[factor].min()
            max_sleep = df_students[factor].max()

            for cluster in sorted(df_students["Cluster"].unique()):
                sns.kdeplot(
                    df_students.loc[df_students["Cluster"] == cluster, factor],
                    ax=ax, label=f"Cluster {cluster}", shade=True
                )

            ax.set_xticks(np.linspace(min_sleep, max_sleep, 5))
            ax.set_xticklabels(
                [f"{x:.2f}" for x in np.linspace(min_sleep, max_sleep, 5)])
            ax.legend()
            st.pyplot(fig)
        else:
            st.write(f"Distribuição de {factor} por Cluster (Contagem)")
            fig, ax = plt.subplots(figsize=(8, 5))

            for cluster in sorted(df_students["Cluster"].unique()):
                cluster_data = df_students[df_students["Cluster"] == cluster]
                value_counts = cluster_data[factor].value_counts().reindex([
                    0, 1], fill_value=0)

                ax.bar(
                    value_counts.index, value_counts.values,
                    width=0.3, label=f"Cluster {cluster}", align="center"
                )

            ax.legend()
            st.pyplot(fig)

except FileNotFoundError:
    st.error("Arquivo não encontrado.")
except Exception as e:
    st.error(f"Erro ao processar os dados: {e}")
