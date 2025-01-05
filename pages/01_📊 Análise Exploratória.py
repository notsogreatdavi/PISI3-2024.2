import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import squarify  # Para treemap
import matplotlib.pyplot as plt

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

    # Histograma com KDE (Plotly)
    st.write("#### Distribuição (Histograma com KDE)")
    fig = px.histogram(df_unistudents, x=variable, nbins=30, title=f"Distribuição de {variable}")
    st.plotly_chart(fig, use_container_width=True)

    # Boxplot Horizontal (Plotly)
    st.write("#### Identificação de Outliers (Boxplot)")
    fig = px.box(df_unistudents, y=variable, title=f"Boxplot de {variable}")
    st.plotly_chart(fig, use_container_width=True)

    # Violin Plot Horizontal (Plotly)
    st.write("#### Distribuição e Densidade (Violin Plot)")
    fig = px.violin(df_unistudents, y=variable, box=True, title=f"Violin Plot de {variable}")
    st.plotly_chart(fig, use_container_width=True)

    # Estatísticas descritivas
    st.write("#### Estatísticas Descritivas")
    st.write(df_unistudents[variable].describe())

# Análise de variáveis categóricas
else:
    st.subheader(f"Análise da Variável Categórica: {variable}")

    # Gráfico de Barras (Plotly)
    st.write("#### Contagem de Categorias (Gráfico de Barras)")
    fig = px.bar(df_unistudents[variable].value_counts(), x=df_unistudents[variable].value_counts().index, y=df_unistudents[variable].value_counts().values, labels={'x': variable, 'y': 'Contagem'}, title=f"Contagem de {variable}")
    st.plotly_chart(fig, use_container_width=True)

    # Treemap (Matplotlib + Squarify)
    st.write("#### Proporção de Categorias (Treemap)")
    category_counts = df_unistudents[variable].value_counts()
    fig, ax = plt.subplots()
    squarify.plot(sizes=category_counts, label=category_counts.index, alpha=0.8, color=sns.color_palette('viridis', len(category_counts)))
    plt.axis('off')
    st.pyplot(fig)

    # Waffle Chart (Matplotlib + Pywaffle)
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
# Distribuição
st.write("""
- **Distribuição:**
  - A média de horas estudadas é 20 e a distribuição é simétrica.
  - A presença nas aulas tem uma média de 80%.
  - A média de horas de sono é 7, e a distribuição é normal com a maioria dos estudantes dormindo entre 6 e 8 horas
  - A média de sessões de tutoria é 1, e a distribuição é concentrada em 0, com poucos estudantes frequentando mais de 2 sessões.
  - A média de atividade física é 3 horas por semana.
""")

# Categorias Dominantes
st.write("""
- **Categorias Dominantes:**
  - O envolvimento dos pais é predominantemente médio (3500 estudantes), mas há uma parcela significativa com envolvimento baixo (1500 estudantes).
  - A maioria dos estudantes participa de atividades extracurriculares (3700 estudantes).
  - O acesso à internet é predominante (6000 estudantes), mas ainda há uma parcela sem acesso.
  - A motivação média é a categoria dominante (3400 estudantes), mas há uma parcela significativa com motivação baixa (2000 estudantes).
  - A maioria dos estudantes vem de escolas públicas (5000 estudantes).
""")

# Relação com as Perguntas da Pesquisa
st.write("""
- **Relação com as Perguntas da Pesquisa:**
  - Os padrões emergentes sugerem que a maioria dos estudantes tem envolvimento médio dos pais, motivação média e acesso à internet, mas há grupos significativos com características diferentes.
  - A relação entre horas estudadas e desempenho acadêmico pode variar entre esses grupos, especialmente para estudantes com menos horas de estudo e notas altas.
  - Características como alta motivação, envolvimento dos pais e acesso a recursos podem explicar o desempenho de estudantes que estudam menos horas.
""")