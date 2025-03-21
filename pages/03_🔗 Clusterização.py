import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_samples, silhouette_score


# Configurações de Página
st.set_page_config(
    page_title="Clusterização de Estudantes", page_icon="📊", layout="wide"
)
st.title("📊 Análise de Clusterização de Estudantes")
st.sidebar.header("Configurações")

# Carregar Dataset
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

    # Seleção de colunas para clusterização
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

    # Método Elbow
    st.subheader("Método Elbow para Determinar o Número de Clusters")
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        kmeans.fit(df_cluster)
        inertia.append(kmeans.inertia_)
    fig = go.Figure()

    # Adicionando a linha do método Elbow
    fig.add_trace(go.Scatter(
        x=list(range(1, 11)),
        y=inertia,
        mode="lines+markers",
        name="Inércia",
        marker=dict(color='blue'),
        line=dict(color='blue', width=2)
    ))
    fig.update_layout(
        title="Método Elbow para Definir o k Ótimo",
        xaxis_title="Número de Clusters (k)",
        yaxis_title="Inércia",
        template="plotly_dark",  # Traz um tema escuro para o gráfico
        showlegend=False,
    )
    st.plotly_chart(fig)

    # Clusterização
    optimal_k = st.sidebar.slider(
        "Número de Clusters (k)", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init=10)
    df_students["Cluster"] = kmeans.fit_predict(df_cluster)

    # Resumo dos Clusters
    st.subheader("Resumo dos Clusters")
    cluster_summary = df_students.groupby(
        "Cluster")[available_features].agg(["mean", "count"])
    st.dataframe(cluster_summary)

    # Caracterização por Médias
    st.subheader("Análise Detalhada por Cluster")
    analysis_features = numeric_cols.tolist()
    cluster_means = df_students.groupby(
        "Cluster")[analysis_features].mean().reset_index()

    st.write("**Médias Padronizadas por Cluster (0-1):**")
    st.dataframe(
        cluster_means.style.background_gradient(cmap="Blues").format("{:.2f}"),
        use_container_width=True
    )

    # Distribuição Detalhada por Característica
    st.write("**Distribuição Detalhada por Característica:**")
    selected_feature = st.selectbox(
        "Selecione uma característica para análise detalhada:", analysis_features)
    fig = px.box(
        df_students,
        x="Cluster",
        y=selected_feature,
        color="Cluster",  # Diferencia as cores dos clusters
        title=f"Distribuição de {selected_feature.replace('_', ' ')} por Cluster",
        labels={"Cluster": "Cluster",
                selected_feature: selected_feature.replace('_', ' ')}
    )
    st.plotly_chart(fig)

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

    # Fatores que serão analisados
    additional_factors = [
        "Motivation_Level_Low", "Parental_Involvement_Low", "Parental_Involvement_Medium",
        "Extracurricular_Activities_Yes", "Sleep_Hours"
    ]

    # Análise de Sleep_Hours
    if "Sleep_Hours" in additional_factors:
        st.write("Distribuição de Sleep Hours por Cluster")
        fig = go.Figure()
        min_sleep = df_students["Sleep_Hours"].min()
        max_sleep = df_students["Sleep_Hours"].max()
        for cluster in sorted(df_students["Cluster"].unique()):
            cluster_data = df_students[df_students["Cluster"] == cluster]
            fig.add_trace(go.Histogram(
                x=cluster_data["Sleep_Hours"],
                name=f"Cluster {cluster}",
                nbinsx=20,
                hovertemplate="Cluster: %{name}<br>Sleep Hours: %{x}<br>Count: %{y}",
            ))
        fig.update_layout(
            title="Distribuição de Sleep Hours por Cluster",
            xaxis_title="Horas de Sono",
            yaxis_title="Contagem",
            barmode='overlay',
            template="plotly_dark",
            showlegend=True
        )
        st.plotly_chart(fig)

    # Análise de Motivação por Cluster
    for factor in additional_factors:
        if factor != "Sleep_Hours":
            st.write(
                f"Distribuição de {factor.replace('_', ' ')} por Cluster (Contagem)")
            fig = go.Figure()
            for cluster in sorted(df_students["Cluster"].unique()):
                cluster_data = df_students[df_students["Cluster"] == cluster]
                value_counts = cluster_data[factor].value_counts().reindex([
                    0, 1], fill_value=0)
                fig.add_trace(go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    name=f"Cluster {cluster}",
                    hovertemplate="Cluster: %{data.name}<br>{factor.replace('_', ' ')}: %{x}<br>Contagem: %{y}",
                    width=0.3
                ))
            fig.update_layout(
                title=f"Distribuição de {factor.replace('_', ' ')} por Cluster (Contagem)",
                xaxis_title=f"{factor.replace('_', ' ')}",
                yaxis_title="Contagem",
                barmode='stack',
                template="plotly_dark",
                showlegend=True
            )
            st.plotly_chart(fig)

    # Gráfico de Silhueta
    st.subheader("Avaliação de Qualidade dos Clusters - Gráfico de Silhueta")
    silhouette_avg = silhouette_score(df_cluster, df_students["Cluster"])
    sample_silhouette_values = silhouette_samples(
        df_cluster, df_students["Cluster"])
    color_palette = px.colors.qualitative.Pastel
    fig = go.Figure()
    current_y = 0
    gap = 10

    for i in range(optimal_k):
        cluster_mask = df_students["Cluster"] == i
        silhouette_values = sample_silhouette_values[cluster_mask]
        silhouette_values_sorted = np.sort(silhouette_values)
        size = len(silhouette_values_sorted)
        y_positions = np.arange(current_y, current_y + size)
        color_hex = color_palette[i % len(color_palette)]
        fig.add_trace(go.Bar(
            x=silhouette_values_sorted,
            y=y_positions,
            orientation="h",
            marker=dict(
                color=color_hex,
                line=dict(color='rgba(0, 0, 255, 1)',
                          width=1)
            ),
            showlegend=False,
            hovertemplate=f"Cluster {i}<br>Coeficiente: %{{x:.2f}}"
        ))

        fig.add_annotation(
            x=-0.05,
            y=current_y + size / 2,
            text=f"Cluster {i} (n={size})",
            showarrow=False,
            font=dict(color="white")
        )
        current_y += size + gap

    fig.add_vline(x=silhouette_avg, line=dict(color="red", dash="dash"),
                  annotation_text=f"Média: {silhouette_avg:.2f}",
                  annotation_position="top right")

    fig.update_layout(
        title="Gráfico de Silhueta para os Clusters",
        xaxis_title="Coeficiente de Silhueta",
        yaxis_title="",
        template="plotly_dark",
        xaxis=dict(range=[-0.1, 1]),
        yaxis=dict(showticklabels=False)
    )
    st.plotly_chart(fig)
    st.write(f"Coeficiente médio de silhueta: {silhouette_avg:.2f}")


except FileNotFoundError:
    st.error("Arquivo não encontrado.")
except Exception as e:
    st.error(f"Erro ao processar os dados: {e}")
