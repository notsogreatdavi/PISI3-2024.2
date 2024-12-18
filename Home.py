import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


# ===== Configuração Inicial ===== #
st.set_page_config(
    page_title="🏫 Desempenho Escolar - Home",
    page_icon="🏠",
    layout="centered"
)

# ===== Estilos Personalizados com CSS ===== #
st.markdown("""
    <style>
        .main { background-color: #F0EFF4; }
        h1, h2 { color: #191970; }
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

# ===== Título e Introdução ===== #
st.title("🏫 Análise de Desempenho Escolar com Machine Learning")
st.write("Aqui utilizamos **análise de dados** e **modelos de aprendizado de máquina** para explorar o desempenho acadêmico dos estudantes. 🚀")

# ===== Carregar o Dataset ===== #
@st.cache_data
def load_data():
    return pd.read_csv("./data/unistudents_treated.csv")  # Ajuste o caminho correto do arquivo

df = load_data()

# ===== Perguntas Norteadoras ===== #
st.subheader("🔍 Perguntas Norteadoras da Pesquisa")
st.write("""
1. **Com base no histórico acadêmico de um estudante e suas condições socioeconômicas, é possível prever a probabilidade de ele enfrentar dificuldades no desempenho escolar?**  
2. **Ao agrupar perfis de estudantes, quais padrões emergem entre os grupos de estudantes quando se analisa a relação entre tempo dedicado ao estudo e desempenho acadêmico, e quais características definem os estudantes que alcançam altas notas com menos horas de estudo?**
""")

# ===== Navegação entre Páginas ===== #
st.subheader("📂 Explore as Páginas do Projeto")
st.write("Navegue pelas seções abaixo para acompanhar a análise de dados e os resultados obtidos.")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Análise Exploratória"):
        st.switch_page("./pages/01_Análise Exploratória.py")
with col2:
    if st.button("🤖 Classificação"):
        st.switch_page("pages/02_Classificacao.py")
with col3:
    if st.button("🔗 Clusterização"):
        st.switch_page("pages/03_Clusterizacao.py")

        # ===== Gráfico Hexbin ===== #
st.header("🔷 Análise de Densidade com Hexbin")

st.write("""
O gráfico hexbin é utilizado para identificar a densidade de pontos entre variáveis numéricas, sendo útil para detectar padrões de concentração.
""")

num_cols = df.select_dtypes(include=['float64', 'int64']).columns
if len(num_cols) > 1:
    col1, col2 = st.columns(2)

    with col1:
        x_hexbin = st.selectbox("Selecione a variável X:", num_cols, index=0, key="hexbin_x")

    with col2:
        y_hexbin = st.selectbox("Selecione a variável Y:", num_cols, index=1, key="hexbin_y")

    # Removendo valores nulos
    hexbin_data = df[[x_hexbin, y_hexbin]].dropna()

    if hexbin_data.empty:
        st.error("Não há dados suficientes para criar o gráfico Hexbin. Selecione outras variáveis.")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        hb = ax.hexbin(
            hexbin_data[x_hexbin],
            hexbin_data[y_hexbin],
            gridsize=30,  # Tamanho dos hexágonos
            cmap='Blues',  # Paleta de cores
            mincnt=1  # Mostrar hexágonos com pelo menos 1 ponto
        )
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label('Contagem')
        ax.set_xlabel(x_hexbin)
        ax.set_ylabel(y_hexbin)
        ax.set_title(f"Densidade entre {x_hexbin} e {y_hexbin}", fontsize=14)
        st.pyplot(fig)

        st.write(f"""
        **Insight**: O gráfico mostra a densidade de pontos entre **{x_hexbin}** e **{y_hexbin}**.  
        Verifique as regiões de maior concentração para identificar possíveis padrões.
        """)
else:
    st.warning("Não há variáveis numéricas suficientes para exibir o gráfico hexbin.")

# ===== Rodapé ===== #
st.write("---")
st.write("""
📚 **Projeto Interdisciplinar - 2024**  
🎓 Desenvolvido por **Davi Vieira, Guilherme Leonardo e Ronaldo Araújo**
""")
