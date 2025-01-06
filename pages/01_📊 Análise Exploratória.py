import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import squarify  # Para treemap
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, f_oneway

# Título da página
st.title("Análise Univariada")

# Introdução
st.write("""
Nesta seção, exploramos a distribuição e as características individuais das variáveis do dataset. 
Escolha uma variável para visualizar sua distribuição e estatísticas descritivas.
""")

# Carregar o dataset
df_unistudents = pd.read_csv("./data/StudentPerformanceFactors.csv")

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
    fig = px.box(df_unistudents, x=variable, title=f"Boxplot de {variable}")
    st.plotly_chart(fig, use_container_width=True)

    # Violin Plot Horizontal (Plotly)
    st.write("#### Distribuição e Densidade (Violin Plot)")
    fig = px.violin(df_unistudents, x=variable, box=True, title=f"Violin Plot de {variable}")
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

# Título da página
st.title("Análise Bivariada")

# Introdução
st.write("""
Nesta seção, exploramos as relações entre pares de variáveis. 
Escolha duas variáveis para visualizar sua relação e obter insights detalhados.
""")


# Seleção de variáveis para análise bivariada
st.sidebar.header("Selecione Variáveis para Análise Bivariada")
var1 = st.sidebar.selectbox(
    "Escolha a primeira variável:",
    options=df_unistudents.columns
)
var2 = st.sidebar.selectbox(
    "Escolha a segunda variável:",
    options=df_unistudents.columns
)

# Análise Bivariada
st.subheader(f"Análise Bivariada: {var1} vs {var2}")

# Relação entre duas variáveis numéricas
if df_unistudents[var1].dtype in ['int64', 'float64'] and df_unistudents[var2].dtype in ['int64', 'float64']:
    # Scatterplot com Linha de Tendência
    st.write("#### Scatterplot com Linha de Tendência")
    fig_scatter = px.scatter(df_unistudents, x=var1, y=var2, trendline="ols", title=f"Scatterplot: {var1} vs {var2}")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Gráfico de Hexbin
    st.write("#### Gráfico de Hexbin")
    fig_hexbin = px.density_heatmap(df_unistudents, x=var1, y=var2, nbinsx=20, nbinsy=20, title=f"Hexbin: {var1} vs {var2}")
    st.plotly_chart(fig_hexbin, use_container_width=True)

    # Coeficiente de Correlação de Pearson
    corr, p_value = pearsonr(df_unistudents[var1], df_unistudents[var2])
    st.write(f"**Coeficiente de Correlação de Pearson:** {corr:.2f}")
    st.write(f"**Valor-p:** {p_value:.4f}")
    st.write("""
    - **Interpretação do Coeficiente de Correlação:**
      - Valor próximo de 1: Correlação positiva forte.
      - Valor próximo de -1: Correlação negativa forte.
      - Valor próximo de 0: Sem correlação significativa.
    """)

# Relação entre uma variável numérica e uma categórica
elif df_unistudents[var1].dtype in ['int64', 'float64'] or df_unistudents[var2].dtype in ['int64', 'float64']:
    num_var = var1 if df_unistudents[var1].dtype in ['int64', 'float64'] else var2
    cat_var = var2 if df_unistudents[var2].dtype == 'object' else var1

    st.write(f"#### Boxplot: Distribuição de {num_var} por {cat_var}")
    fig_boxplot = px.box(df_unistudents, x=cat_var, y=num_var, title=f"Boxplot: {num_var} por {cat_var}")
    st.plotly_chart(fig_boxplot, use_container_width=True)

    st.write(f"#### Violin Plot: Distribuição e Densidade de {num_var} por {cat_var}")
    fig_violin = px.violin(df_unistudents, x=cat_var, y=num_var, box=True, title=f"Violin Plot: {num_var} por {cat_var}")
    st.plotly_chart(fig_violin, use_container_width=True)

    # Teste de Hipóteses (ANOVA)
    st.write("#### Teste de Hipóteses (ANOVA)")
    groups = [df_unistudents[df_unistudents[cat_var] == category][num_var] for category in df_unistudents[cat_var].unique()]
    f_stat, p_value = f_oneway(*groups)
    st.write(f"**Estatística F:** {f_stat:.2f}")
    st.write(f"**Valor-p:** {p_value:.4f}")
    st.write("""
    - **Interpretação do Teste ANOVA:**
      - Valor-p < 0.05: Há diferenças significativas entre os grupos.
      - Valor-p >= 0.05: Não há diferenças significativas entre os grupos.
    """)

# Relação entre duas variáveis categóricas
else:
    st.write(f"#### Heatmap: Relação entre {var1} e {var2}")
    cross_tab = pd.crosstab(df_unistudents[var1], df_unistudents[var2])
    fig_heatmap = px.imshow(cross_tab, labels=dict(x=var2, y=var1, color="Contagem"), title=f"Heatmap: {var1} vs {var2}")
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.write(f"#### Gráfico de Barras Empilhadas: Proporção de {var1} por {var2}")
    fig_stacked_bar = px.bar(cross_tab, barmode="stack", title=f"Gráfico de Barras Empilhadas: {var1} por {var2}")
    st.plotly_chart(fig_stacked_bar, use_container_width=True)

# Insights e Observações
# Insights e Observações
st.subheader("Insights e Observações")

st.write("""
#### Relação entre Variáveis Numéricas:
- **Horas Estudadas (Hours Studied) vs Nota no Exame (Exam Score):**
  - Observamos uma tendência de crescimento nas notas à medida que as horas estudadas aumentam, mas o coeficiente de Pearson não indica uma correlação significativa. Isso sugere que outros fatores, como a qualidade do estudo, podem ser mais importantes.
  
- **Nota no Exame (Exam Score) vs Presença (Attendance):**
  - Há uma correlação positiva moderada (coeficiente de Pearson = 0.58), confirmada por um valor-p de 0. Isso indica que a frequência nas aulas é um fator importante para o desempenho acadêmico.
""")

st.write("""
#### Relação entre Variável Numérica e Categórica:
- **Horas Estudadas (Hours Studied) vs Acesso a Recursos (Access to Resources):**
  - Não há diferenças significativas nas horas estudadas entre os níveis de acesso a recursos (estatística F = 0.63, valor-p = 0.53).

- **Horas Estudadas (Hours Studied) vs Nível de Motivação (Motivation Level):**
  - Embora o boxplot sugira uma tendência, o teste ANOVA não confirma diferenças significativas (estatística F = 1.66, valor-p = 0.199).

- **Nota no Exame (Exam Score) vs Envolvimento dos Pais (Parental Involvement):**
  - Estudantes com alto envolvimento dos pais têm notas significativamente mais altas (estatística F = 84.49, valor-p = 0).

- **Nota no Exame (Exam Score) vs Acesso a Recursos (Access to Resources):**
  - Estudantes com alto acesso a recursos têm notas médias mais altas (estatística F = 98, valor-p = 0).

- **Nota no Exame (Exam Score) vs Nível de Motivação (Motivation Level):**
  - Estudantes com maior motivação tendem a ter notas mais altas (estatística F = 25, valor-p = 0).
""")

st.write("""
#### Relação entre Variáveis Categóricas:
- **Envolvimento dos Pais (Parental Involvement) vs Nível de Motivação (Motivation Level):**
  - Estudantes com alto envolvimento dos pais tendem a ter níveis de motivação mais altos, sugerindo que o suporte familiar influencia positivamente a motivação.

- **Tipo de Escola (School Type) vs Acesso a Recursos (Access to Resources):**
  - Estudantes de escolas particulares têm maior acesso a recursos, o que pode contribuir para diferenças no desempenho acadêmico.
""")

st.write("""
#### Conclusão Geral:
- A presença nas aulas, o envolvimento dos pais, o acesso a recursos e a motivação são os fatores mais fortemente associados a um bom desempenho acadêmico.
- A falta de correlação significativa entre horas estudadas e notas sugere que a qualidade do estudo e outros fatores contextuais são mais importantes do que o tempo dedicado.
""")