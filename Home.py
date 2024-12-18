import streamlit as st
import pandas as pd
import plotly.express as px

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
    return pd.read_parquet("data/unistudents.parquet")  # Ajuste o caminho correto do arquivo

df = load_data()

# ===== Gráfico Interativo: Gráfico de Barras ===== #
st.subheader("📊 Visualize a Distribuição das Variáveis")

# Dropdown para escolher a variável categórica
column = st.selectbox("Escolha uma variável categórica para visualizar:", 
                      ["Parental_Education", "Family_Income", "Stress_Levels", "Grades", 
                       "Motivation", "School_Environment", "Sports_Participation"])

# Contagem das categorias e ajuste do índice
data = df[column].value_counts().reset_index()
data.columns = [column, "count"]  # Renomeia as colunas para o Plotly

# Gráfico de barras
fig = px.bar(data, 
             x=column, 
             y="count", 
             title=f"Distribuição de {column}",
             labels={column: column, "count": "Contagem"},
             color=column,  # Adiciona cor para tornar mais visual
             color_discrete_sequence=px.colors.qualitative.Safe)

# Plot do gráfico
st.plotly_chart(fig)

# ===== Rodapé ===== #
st.write("---")
st.write("""
📚 **Projeto Interdisciplinar - 2024**  
🎓 Desenvolvido por **Davi Vieira, Guilherme Leonardo e Ronaldo Araújo**
""")
