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
    df_students = pd.read_parquet("./data/new_unistudents_classification.parquet")
    st.success("Dataset carregado com sucesso!")

    # Imputação de valores faltantes
    imputer = SimpleImputer(strategy="mean")
    numeric_cols = df_students.select_dtypes(include=["float64", "int64"]).columns
    df_students[numeric_cols] = imputer.fit_transform(df_students[numeric_cols])

    # Normalização das colunas numéricas
    scaler = MinMaxScaler()
    df_students[numeric_cols] = scaler.fit_transform(df_students[numeric_cols])

    # Seleção de colunas para clusterização (com colunas One-Hot Encoded)
    cluster_features = [
        "Hours_Studied",
        "Attendance",
        "Parental_Involvement",
        "Access_to_Resources",
        "Extracurricular_Activities",
        "Sleep_Hours",
        "Previous_Scores",
        "Motivation_Level_Low",
        "Motivation_Level_Medio",
        "Motivation_Level_High",
        "Internet_Access",
        "Tutoring_Sessions",
        "Family_Income",
        "Teacher_Quality",
        "School_Type",
        "Peer_Influence",
        "Physical_Activity",
        "Learning_Disabilities",
        "Parental_Education_Level",
        "Distance_from_Home",
        "Exam_Score",
    ]

    # Garantir que apenas colunas existentes sejam usadas
    available_features = [col for col in cluster_features if col in df_students.columns]

    if not available_features:
        st.error(
            "Nenhuma das colunas necessárias para a clusterização está disponível no dataset."
        )
    else:
        df_cluster = df_students[available_features]

    # ===== Método Elbow ===== #
    st.subheader("Método Elbow para Determinar o Número de Clusters")
    st.write(
        """O Elbow, também chamado de método do cotovelo, é amplamente utilizado para determinar o número ideal de clusters (𝑘) em algoritmos de clusterização como o K-Means. 
        Ele ajuda a encontrar o equilíbrio entre a simplicidade do modelo e a capacidade de representar os dados."""
    )
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(df_cluster)
        inertia.append(kmeans.inertia_)

    # Plotagem do gráfico Elbow
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 11), inertia, marker="o")
    ax.set_title("Método Elbow para Definir k Ótimo")
    ax.set_xlabel("Número de Clusters (k)")
    ax.set_ylabel("Inércia")
    st.pyplot(fig)

    # ===== Clusterização ===== #
    optimal_k = st.sidebar.slider(
        "Número de Clusters (k)", min_value=2, max_value=10, value=3
    )
    kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init=10)
    df_students["Cluster"] = kmeans.fit_predict(df_cluster)

    # ===== Resumo dos Clusters ===== #
    st.subheader("Resumo dos Clusters")

    # Filtra apenas as colunas numéricas para agregação
    numeric_cols_for_agg = df_students.select_dtypes(
        include=["float64", "int64"]
    ).columns

    # Realiza a agregação de médias para as colunas numéricas
    try:
        cluster_summary = df_students.groupby("Cluster")[numeric_cols_for_agg].agg(
            ["mean", "count"]
        )
        st.dataframe(cluster_summary)
    except Exception as e:
        st.error(f"Erro ao calcular a agregação: {e}")

    # ===== Visualização dos Clusters ===== #
    st.subheader("Visualização dos Clusters")
    st.write(
        """
        Este gráfico apresenta a distribuição dos estudantes em diferentes clusters, com base em suas horas de estudo e notas. 
        Os centróides (em vermelho) representam o ponto médio de cada cluster, indicando as características médias dos estudantes em cada grupo.
        """
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df_students["Hours_Studied"],
        df_students["Exam_Score"],
        c=df_students["Cluster"],
        cmap="viridis",
        alpha=0.7,
    )
    ax.scatter(
        kmeans.cluster_centers_[:, available_features.index("Hours_Studied")],
        kmeans.cluster_centers_[:, available_features.index("Exam_Score")],
        s=300,
        c="red",
        label="Centroids",
    )
    ax.set_xlabel("Horas de Estudo")
    ax.set_ylabel("Nota do Exame")
    ax.set_title("Clusterização de Estudantes com Base em Horas de Estudo e Notas")
    ax.legend()
    st.pyplot(fig)

    # ===== Cluster de Melhor Desempenho ===== #
    st.subheader("Identificação do Cluster com Maiores Notas e Menores Horas de Estudo")
    cluster_means = df_students.groupby("Cluster")[
        ["Hours_Studied", "Exam_Score"]
    ].mean()

    # Ordenar os clusters: maiores notas e menores horas
    sorted_clusters = cluster_means.sort_values(
        by=["Exam_Score", "Hours_Studied"], ascending=[False, True]
    )
    best_cluster = sorted_clusters.index[0]

    # Exibir o resumo desse cluster
    st.write(f"### Cluster {best_cluster} - Maiores Notas com Menores Horas de Estudo")
    st.write("Médias desse cluster:")
    st.dataframe(sorted_clusters.loc[best_cluster].to_frame().T)

    # Exibir a amostra de estudantes pertencentes ao cluster escolhido
    st.subheader("Amostra dos Estudantes no Cluster de Alto Desempenho")
    best_cluster_students = df_students[df_students["Cluster"] == best_cluster]
    st.dataframe(best_cluster_students.head())

    # ===== Fatores Adicionais ===== #
    st.subheader("Fatores Adicionais que Influenciam o Desempenho")
    st.write(
        """
        Aqui estão os fatores adicionais, como motivação, níveis de estresse e qualidade do ambiente escolar, 
        que podem ser analisados para entender melhor as características dos estudantes de cada cluster.
        """
    )
    additional_factors = [
        "Motivation_Level_Low",
        "Motivation_Level_Medium",
        "Parental_Involvement_Low",
        "Parental_Involvement_Medium",
        "Sleep_Hours",
    ]
    for factor in additional_factors:
        if factor == "Sleep_Hours":
            st.write(f"Distribuição de {factor} por Cluster")
            fig, ax = plt.subplots(figsize=(8, 5))
            for cluster in sorted(df_students["Cluster"].unique()):
                sns.kdeplot(
                    df_students.loc[df_students["Cluster"] == cluster, factor],
                    ax=ax,
                    label=f"Cluster {cluster}",
                )
            ax.set_title(f"Distribuição de {factor} por Cluster")
            ax.set_xlabel(factor)
            ax.set_ylabel("Densidade")
            ax.legend()
            st.pyplot(fig)
        else:
            st.write(f"Distribuição de {factor} por Cluster (Contagem)")
            fig, ax = plt.subplots(figsize=(8, 5))

            # Gerando a contagem para cada valor de 0 e 1 nas colunas One-Hot Encoded
            for cluster in sorted(df_students["Cluster"].unique()):
                cluster_data = df_students[df_students["Cluster"] == cluster]

                # Calculando a contagem de 0 e 1
                value_counts = (
                    cluster_data[factor].value_counts().reindex([0, 1], fill_value=0)
                )

                ax.bar(
                    value_counts.index,
                    value_counts.values,
                    width=0.3,
                    label=f"Cluster {cluster}",
                    align="center",
                )

            ax.set_title(f"Distribuição de {factor} por Cluster")
            ax.set_xlabel(factor)
            ax.set_ylabel("Contagem")
            ax.legend()
            st.pyplot(fig)

except FileNotFoundError:
    st.error(f"Arquivo não encontrado.")
except Exception as e:
    st.error(f"Erro ao processar os dados: {e}")
