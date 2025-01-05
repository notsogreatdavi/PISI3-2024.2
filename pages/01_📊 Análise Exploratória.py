import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify  # Para treemap
import pywaffle  # Para waffle chart

# Título da página
st.title("Análise Univariada")

# Introdução
st.write("""
Nesta seção, exploramos a distribuição e as características individuais das variáveis do dataset. 
Escolha uma variável para visualizar sua distribuição e estatísticas descritivas.
""")

# Carregar o dataset
df_unistudents = pd.read_parquet("./data/new_unistudents.parquet")

# Seleção de variáveis
st.sidebar.header("Selecione uma Variável")
variable = st.sidebar.selectbox(
    "Escolha uma variável para análise:",
    options=df_unistudents.columns
)

# Análise de variáveis numéricas
if df_unistudents[variable].dtype in ['int64', 'float64']:
    st.subheader(f"Análise da Variável Numérica: {variable}")

    # Histograma com KDE
    st.write("#### Distribuição (Histograma com KDE)")
    fig, ax = plt.subplots()
    sns.histplot(df_unistudents[variable], kde=True, ax=ax, color='skyblue')
    st.pyplot(fig)

    # Boxplot
    st.write("#### Identificação de Outliers (Boxplot)")
    fig, ax = plt.subplots()
    sns.boxplot(x=df_unistudents[variable], ax=ax, color='lightgreen')
    st.pyplot(fig)

    # Violin Plot
    st.write("#### Distribuição e Densidade (Violin Plot)")
    fig, ax = plt.subplots()
    sns.violinplot(x=df_unistudents[variable], ax=ax, color='orange')
    st.pyplot(fig)

    # Estatísticas descritivas
    st.write("#### Estatísticas Descritivas")
    st.write(df_unistudents[variable].describe())

# Análise de variáveis categóricas
else:
    st.subheader(f"Análise da Variável Categórica: {variable}")

    # Gráfico de Barras
    st.write("#### Contagem de Categorias (Gráfico de Barras)")
    fig, ax = plt.subplots()
    sns.countplot(x=df_unistudents[variable], ax=ax, palette='viridis')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Treemap
    st.write("#### Proporção de Categorias (Treemap)")
    category_counts = df_unistudents[variable].value_counts()
    fig, ax = plt.subplots()
    squarify.plot(sizes=category_counts, label=category_counts.index, alpha=0.8, color=sns.color_palette('viridis', len(category_counts)))
    plt.axis('off')
    st.pyplot(fig)

    # Waffle Chart
    st.write("#### Proporção de Categorias (Waffle Chart)")
    category_percent = df_unistudents[variable].value_counts(normalize=True) * 100
    fig = plt.figure(
        FigureClass=pywaffle.Waffle,
        rows=5,
        columns=10,
        values=category_percent,
        labels=[f"{k} ({v:.1f}%)" for k, v in category_percent.items()],
        colors=sns.color_palette('viridis', len(category_percent)),
        legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)}
    )
    st.pyplot(fig)

# Insights e Observações
st.subheader("Insights e Observações")
st.write("""
- **Distribuição:** [Comente sobre a forma da distribuição].
- **Outliers:** [Comente sobre a presença ou ausência de outliers].
- **Categorias Dominantes:** [Comente sobre as categorias mais frequentes, se aplicável].
- **Relação com as Perguntas de Pesquisa:** [Destaque como essa variável pode estar relacionada às perguntas de pesquisa].
""")