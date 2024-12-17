import streamlit as st

# ===== Configuração Inicial ===== #
st.set_page_config(
    page_title="🏫 Desempenho Escolar - Home",
    page_icon="🏠",
    layout="centered"
)

# ===== Estilos Personalizados com CSS ===== #
st.markdown("""
    <style>
        /* Fundo geral */
        .main {
            background-color: #F0EFF4;
        }

        /* Título principal */
        h1, h2 {
            color: #191970;
        }

        /* Botões personalizados */
        div.stButton > button {
            background-color: #FCE762;
            color: #353531;
            font-size: 16px;
            border-radius: 8px;
            border: none;
        }

        /* Rodapé */
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

# ===== Seção de Overview ===== #
st.subheader("📋 Visão Geral do Projeto")
st.write("""
Este estudo utiliza um **dataset de 10.064 estudantes** com 35 características, incluindo fatores acadêmicos, sociais e de estilo de vida.  
O objetivo é identificar padrões que influenciam o sucesso acadêmico e sugerir estratégias para otimizar o aprendizado.
""")

# ===== Perguntas Norteadoras ===== #
st.subheader("🔍 Perguntas Norteadoras da Pesquisa")
st.write("""
1. **Com base no histórico acadêmico de um estudante e suas condições socioeconômicas, é possível prever a probabilidade de ele enfrentar dificuldades no desempenho escolar?**  
2. **Ao agrupar perfis de estudantes, quais padrões emergem entre os grupos de estudantes quando se analisa a relação entre tempo dedicado ao estudo e desempenho acadêmico, e quais características definem os estudantes que alcançam altas notas com menos horas de estudo?**
""")

# ===== Navegação entre Páginas ===== #
st.subheader("📂 Explore as Páginas do Projeto")
st.write("Navegue pelas seções abaixo para acompanhar a análise de dados e os resultados obtidos.")

col1, col2, col3 = st.columns(3)  # Layout com 3 colunas para organização

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
