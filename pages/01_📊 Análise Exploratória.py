import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ===== Configuração Inicial ===== #
st.set_page_config(
    page_title="📊 Análise Exploratória - Desempenho Escolar",
    page_icon="📊",
    layout="wide"
)

# ===== Estilos Personalizados com CSS ===== #
st.markdown("""
    <style>
        .main {
            background-color: #F0EFF4;
        }

        h1, h2 {
            color: #191970;
        }

        div.stButton > button {
            background-color: #FCE762;
            color: #353531;
            font-size: 16px;
            border-radius: 8px;
            border: none;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True)

# ===== Função para carregar os dados ===== #
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath, index_col=0)

# ===== Carregar os Dados Brutos e Tratados ===== #
raw_data_path = "./data/unistudents.csv"
treated_data_path = "./data/unistudents_treated.csv"

raw_data = load_data(raw_data_path)
treated_data = load_data(treated_data_path)

# ===== Título e Introdução ===== #
st.title("📊 Análise Exploratória de Dados")

# ===== Visão Geral ===== #
st.header("🔍 Visão Geral dos Dados")
st.write("**Dataset Bruto (Antes do Tratamento):**")
st.dataframe(raw_data.head())

st.write("**Dataset Tratado (Após o Tratamento):**")
st.dataframe(treated_data.head())

import matplotlib.pyplot as plt

# ===== Gráfico de Valores Nulos ===== #
st.header("📉 Análise de Valores Nulos")
missing_percent = (raw_data.isnull().sum() / len(raw_data)) * 100
missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)

if not missing_percent.empty:
    # Criando o gráfico de barras horizontal com matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    missing_percent.plot(kind='barh', ax=ax, color='skyblue')

    # Ajustando a legenda e os títulos
    ax.set_xlabel('Porcentagem de Valores Nulos', fontsize=12)
    ax.set_ylabel('Variáveis', fontsize=12)
    ax.set_title('Percentual de Valores Nulos por Variável', fontsize=16)

    # Exibindo o gráfico no Streamlit
    st.pyplot(fig)

    st.write("""
    **Observação**: O dataset bruto possui cerca de 0.10 de seu total de valores como nulos. Esses valores foram tratados no dataset final.
    """)
else:
    st.success("Não há valores nulos no dataset bruto.")



    
# ===== Matriz de Correlação ===== #
st.header("📈 Matriz de Correlação")
st.write("""
Explore a correlação entre variáveis numéricas. Relações mais próximas de **1** ou **-1** indicam forte correlação positiva ou negativa, enquanto valores próximos de **0** indicam correlação fraca ou inexistente.
""")

# Seleção de variáveis numéricas
num_cols = treated_data.select_dtypes(include=['float64', 'int64']).columns

if len(num_cols) > 1:
    # Cálculo da matriz de correlação
    correlation_matrix = treated_data[num_cols].corr()

    # Heatmap interativo com Plotly
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,  # Exibe os valores diretamente no heatmap
        color_continuous_scale='RdBu',  # Mudando a escala de cores para 'RdBu' que cobre o intervalo de -1 a 1
        zmin=-1,  # Definindo o mínimo como -1
        zmax=1,   # Definindo o máximo como 1
        title="Matriz de Correlação - Variáveis Numéricas",
        labels=dict(color="Correlação"),
        aspect="auto"
    )
    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=20),
        coloraxis_colorbar=dict(title="Correlação"),
    )

    # Exibir o gráfico
    st.plotly_chart(fig, use_container_width=True)

    st.write("""
    💡 Aqui a gente percebe que não há correlações muito fortes no nosso dataset a primeira vista e que é necessária fazer uma análise mais detalhada para entender melhor como essas variáveis se relacionam
    """)
else:
    st.warning("Não há variáveis numéricas suficientes para calcular a matriz de correlação.")


# ===== Gráfico de Caixa ===== #
st.header("📦 Análise de Distribuição com Gráficos de Caixa")

# Selecionando variáveis numéricas
num_cols = treated_data.select_dtypes(include=['float64', 'int64']).columns

if len(num_cols) > 0:
    # Seleção da variável para análise
    selected_box = st.selectbox("Selecione a variável para o gráfico de caixa:", num_cols, key="boxplot_horizontal_select")

    # Gráfico de caixa horizontal com Plotly
    fig = px.box(
        treated_data, 
        x=selected_box,  # Altere para "x" em vez de "y" para exibição horizontal
        points="all", 
        orientation="h",
        color_discrete_sequence=["#6883BA"], 
        title=f"Gráfico de Caixa (Horizontal) para {selected_box}"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Insight sobre o gráfico
    st.write(f"**Insight**: O gráfico mostra a distribuição de **{selected_box}** e possíveis outliers.")
else:
    st.warning("Não há variáveis numéricas disponíveis para exibir no gráfico de caixa.")

# ===== Histograma ===== #
st.header("📊 Análise de Frequência com Histogramas")
st.write("Visualize a distribuição de variáveis numéricas.")

if len(num_cols) > 0:
    selected_hist = st.selectbox("Selecione a variável para o histograma:", num_cols, key="histogram_select")
    fig = px.histogram(
        treated_data, 
        x=selected_hist, 
        nbins=19, 
        color_discrete_sequence=["#FF7F50"], 
        title=f"Histograma de {selected_hist}"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"**Insight**: O histograma mostra a distribuição da variável **{selected_hist}**.")


# ===== Gráfico de Caixa Multivariado ===== #
st.subheader("📦 Gráfico de Caixa com Duas Variáveis")

# Seleção de variáveis
col1, col2 = st.columns(2)

with col1:
    numeric_var = st.selectbox("Selecione a variável numérica:", treated_data.select_dtypes(include='number').columns)

with col2:
    categorical_var = st.selectbox("Selecione a variável categórica:", treated_data.select_dtypes(include='number').columns)

# Gráfico interativo
fig = px.box(
    treated_data,
    x=categorical_var,
    y=numeric_var,
    color=categorical_var,  # Colore por categoria
    title=f"Distribuição de {numeric_var} para cada valor de {categorical_var}",
    template="simple_white"
)

# Ajustes visuais
fig.update_layout(
    xaxis_title=categorical_var,
    yaxis_title=numeric_var,
    boxmode="group",
    width=800,
    height=500
)

st.plotly_chart(fig, use_container_width=True)

st.write(f"""
**Insight**: Este gráfico permite analisar a distribuição de **{numeric_var}** para cada categoria da variável **{categorical_var}**, facilitando a comparação das distribuições entre grupos.
""")

# ===== Histograma multivariado===== #
st.subheader("📊 Histograma Multivariado")

# Seleção de colunas
col1, col2 = st.columns(2)

with col1:
    attendance_col = st.selectbox("Selecione a variável para o histograma:", num_cols, key="attendance_col")

with col2:
    grades_col = st.selectbox("Selecione a variável para o histograma:", num_cols, key="grades_col")

# Criação do Histograma com Facet Col
fig = px.histogram(
    treated_data,
    x=attendance_col,
    color=grades_col,  # Coloração por categoria de Notas (opcional)
    facet_col=grades_col,  # Divide o histograma em subplots para cada valor de Notas
    title=f"Distribuição de {attendance_col} ao longo de cada valor de {grades_col}",
    template="simple_white",
    histnorm="density"  # Conta a frequência dos valores
)

# Ajustes visuais
fig.update_layout(
    xaxis_title=attendance_col,
    yaxis_title="Contagem",
    showlegend=False,  # Remove legenda para evitar redundância
    height=500,
    width=1000
)

# Renderização no Streamlit
st.plotly_chart(fig, use_container_width=True)

st.write(f"""
**Insight**: Este histograma apresenta a distribuição da variável **{attendance_col}** (eixo X) para cada valor distinto da variável **{grades_col}**.  
Os subplots facilitam a análise comparativa entre os diferentes grupos de **Notas**.
""")


# ===== Conclusão ===== #
st.header("📝 Conclusões e Insights")
st.write("""
1. **Análise Exploratória**: Exploramos o dataset tratado e bruto com várias visualizações.
2. **Distribuição de Variáveis**: Identificamos padrões de frequência e possíveis outliers em variáveis importantes.
3. **Próximos Passos**: Implementar modelos de Machine Learning para responder às questões norteadoras.
""")
