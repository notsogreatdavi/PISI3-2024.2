import streamlit as st
import pandas as pd
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

# ===== Título e Introdução ===== #
st.title("🏫 Análise de Desempenho Escolar com Machine Learning")
st.write("""
Bem-vindo ao MindRats! 🐀  
Aqui utilizamos **análise de dados** e **modelos de aprendizado de máquina** para explorar e entender os principais fatores que afetam o desempenho dos estudantes.  
""")

# ===== Carregar Dataset Parquet ===== #
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("./data/unistudents.csv")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo Parquet: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("O dataset não foi carregado. Verifique o arquivo e tente novamente.")
else:
    st.success("Dataset carregado com sucesso! 🎉")

    # ===== Visualizações ===== #
    st.subheader("📊 Visualizações Iniciais")
    st.write("Aqui estão algumas visualizações rápidas do nosso conjunto de dados para que você tenha uma visão geral dos padrões presentes.")

    # ===== Gráfico 1: Distribuição de Notas ===== #
    st.write("### 🎓 Distribuição de Notas dos Estudantes")
    fig, ax = plt.subplots()
    df['Grades'].value_counts().plot(kind='bar', color="#FCE762", ax=ax)
    plt.title("Distribuição de Notas")
    plt.xlabel("Notas")
    plt.ylabel("Quantidade")
    st.pyplot(fig)

    # ===== Gráfico 2: Distribuição de Horas Estudadas ===== #
    st.write("### ⏳ Distribuição de Horas Estudadas")
    fig, ax = plt.subplots()
    plt.hist(df['Study_Hours'], bins=10, color="#414E95", edgecolor="white")
    plt.title("Distribuição de Horas de Estudo")
    plt.xlabel("Horas Estudadas")
    plt.ylabel("Quantidade de Estudantes")
    st.pyplot(fig)

    # ===== Gráfico 3: Níveis de Atividade Física ===== #
    if 'Physical_Activity' in df.columns:
        st.write("### 🏃 Níveis de Atividade Física entre Estudantes")
        fig, ax = plt.subplots()
        df['Physical_Activity'].value_counts().plot(kind='pie', autopct='%1.1f%%', 
                                                    colors=["#6883BA", "#FCE762", "#353531"],
                                                    startangle=140)
        plt.title("Distribuição da Atividade Física")
        plt.ylabel("")  # Remove o label padrão
        st.pyplot(fig)

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
        st.switch_page("pages/2_Analise_Exploratoria.py")
with col2:
    if st.button("🤖 Classificação"):
        st.switch_page("pages/3_Classificacao.py")
with col3:
    if st.button("🔗 Clusterização"):
        st.switch_page("pages/4_Clusterizacao.py")

# ===== Rodapé ===== #
st.write("---")
st.write("""
📚 **Projeto Interdisciplinar - 2024**  
🎓 Desenvolvido por **Davi Vieira, Guilherme Leonardo e Ronaldo Araújo**  
""")
