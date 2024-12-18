import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans

# Configurações iniciais
st.title("Análise de Clusterização de Estudantes")
st.sidebar.header("Configurações")

# Carregar dados
csv_path = "./data/unistudents.csv"

try:
    df_unistudents = pd.read_csv(csv_path, low_memory=False)
    pd.set_option("display.max_columns", None)

    # Conversão de tipo
    df_unistudents["Family_Income"] = pd.to_numeric(
        df_unistudents["Family_Income"], errors="coerce"
    )

    # Imputação de valores faltantes
    # Numéricas
    num_imputer = SimpleImputer(strategy="mean")
    num_col = df_unistudents.select_dtypes(include=["float64", "int32"]).columns
    df_unistudents[num_col] = num_imputer.fit_transform(df_unistudents[num_col])
    df_unistudents[num_col] = df_unistudents[num_col].apply(np.round)

    # Categóricas
    cat_imputer = SimpleImputer(strategy="most_frequent")
    cat_col = df_unistudents.select_dtypes(include=["object"]).columns
    df_unistudents[cat_col] = cat_imputer.fit_transform(df_unistudents[cat_col])

    # Codificação de colunas categóricas
    colunas_categoricas = [
        "Gender",
        "Parental_Education",
        "Previous_Grades",
        "Class_Participation",
        "Major",
        "School_Type",
        "Financial_Status",
        "Parental_Involvement",
        "Educational_Resources",
        "Motivation",
        "Self_Esteem",
        "Stress_Levels",
        "School_Environment",
        "Professor_Quality",
        "Extracurricular_Activities",
        "Nutrition",
        "Physical_Activity",
        "Educational_Tech_Use",
        "Peer_Group",
        "Bullying",
        "Study_Space",
        "Learning_Style",
        "Tutoring",
        "Mentoring",
        "Lack_of_Interest",
        "Sports_Participation",
        "Grades",
    ]
    label_encoder = LabelEncoder()
    for column in colunas_categoricas:
        df_unistudents[column] = label_encoder.fit_transform(
            df_unistudents[column].astype(str)
        )

    # Seleção de colunas para clusterização
    df_cluster = df_unistudents[
        ["Study_Hours", "Grades", "Class_Participation"]
    ].dropna()

    # Normalização dos dados
    scaler = MinMaxScaler()
    df_cluster_scaled = scaler.fit_transform(df_cluster[["Study_Hours", "Grades"]])

    # Determinar o número ótimo de clusters
    st.subheader("Método Elbow para Determinar o Número de Clusters")
    st.write(
        """ O Elbow, também chamado de método do cotovelo,  é amplamente utilizado para determinar o número ideal de clusters (𝑘) em algoritmos de clusterização como o K-Means. Ele ajuda a encontrar o equilíbrio entre a simplicidade do modelo e a capacidade de representar os dados.
    """
    )
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(df_cluster_scaled)
        inertia.append(kmeans.inertia_)

    # Exibir gráfico Elbow
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 11), inertia, marker="o")
    ax.set_title("Método Elbow para Definir k Ótimo")
    ax.set_xlabel("Número de Clusters (k)")
    ax.set_ylabel("Inércia")
    st.pyplot(fig)

    # Seleção de k ótimo e aplicação do KMeans
    optimal_k = st.sidebar.slider(
        "Número de Clusters (k)", min_value=2, max_value=10, value=3
    )
    kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init=10)
    df_cluster["Cluster"] = kmeans.fit_predict(df_cluster_scaled)

    # Resumo dos clusters
    st.subheader("Resumo dos Clusters")
    cluster_summary = df_cluster.groupby("Cluster").agg(["mean", "count"])
    st.dataframe(cluster_summary)

    # Visualização dos clusters
    st.subheader("Visualização dos Clusters")
    st.write(
        """
    Este gráfico apresenta a distribuição dos estudantes em diferentes clusters, com base em suas horas de estudo e notas. 
    Os centróides (em vermelho) representam o ponto médio de cada cluster, indicando as características médias dos estudantes em cada grupo.
    """
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df_cluster["Study_Hours"],
        df_cluster["Grades"],
        c=df_cluster["Cluster"],
        cmap="viridis",
        alpha=0.7,
    )
    ax.scatter(
        kmeans.cluster_centers_[:, 0]
        * (df_cluster["Study_Hours"].max() - df_cluster["Study_Hours"].min())
        + df_cluster["Study_Hours"].min(),
        kmeans.cluster_centers_[:, 1]
        * (df_cluster["Grades"].max() - df_cluster["Grades"].min())
        + df_cluster["Grades"].min(),
        s=300,
        c="red",
        label="Centroids",
    )
    ax.set_xlabel("Horas de Estudo")
    ax.set_ylabel("Notas")
    ax.set_title("Clusterização de Estudantes com Base em Horas de Estudo e Notas")
    ax.legend()
    st.pyplot(fig)

    # Análise do cluster com maior desempenho e menor esforço
    cluster_means = df_cluster.groupby("Cluster")[["Study_Hours", "Grades"]].mean()
    sorted_cluster = cluster_means.sort_values(
        by=["Grades", "Study_Hours"], ascending=[False, True]
    )
    high_grade_low_hours_cluster = sorted_cluster.index[0]

    st.subheader(
        f"Cluster com maiores notas e menos horas de estudo: {high_grade_low_hours_cluster}"
    )
    st.write("Médias desse cluster:")
    st.dataframe(sorted_cluster.loc[high_grade_low_hours_cluster])

    # Exibição dos estudantes do cluster escolhido
    st.subheader("Amostra dos Estudantes no Cluster de Alto Desempenho")
    cluster_students = df_unistudents.loc[
        df_cluster["Cluster"] == high_grade_low_hours_cluster
    ]
    st.dataframe(cluster_students.head())
except FileNotFoundError:
    st.error(
        "Arquivo './data/unistudents.csv' não encontrado. Certifique-se de que o caminho está correto."
    )
