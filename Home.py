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

# ===== Título e Introdução ===== #
st.title("🏫 Análise de Desempenho Escolar com Machine Learning")
st.write("Aqui utilizamos **análise de dados** e **modelos de aprendizado de máquina** para explorar o desempenho acadêmico dos estudantes. 🚀")


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
        st.switch_page("pages/01_📊 Análise Exploratória.py")
with col2:
    if st.button("🤖 Classificação"):
        st.switch_page("pages/02_🤖 Classificação.py")
with col3:
    if st.button("🔗 Clusterização"):
        st.switch_page("pages/03_🔗 Clusterização.py")

# Introdução ao Contexto
st.header("Introdução ao Contexto")

# Apresentação do dataset
st.write("""
Este painel interativo tem como objetivo analisar o desempenho acadêmico de estudantes com base em diversos fatores, como horas de estudo, motivação, suporte familiar e acesso a recursos educacionais. O dataset utilizado contém informações detalhadas sobre os estudantes e suas características.
""")

# Carregar o dataset
df_unistudents = pd.read_parquet("./data/new_unistudents.parquet")

# Exibir o dataset
st.subheader("Dataset Utilizado")
st.write("""
Abaixo estão as primeiras linhas do dataset após o pré-processamento:
""")
st.dataframe(df_unistudents.head())

# Explicação das variáveis
st.subheader("Variáveis Disponíveis")
st.write("""
O dataset contém as seguintes variáveis:

- **Hours_Studied**: Número de horas dedicadas ao estudo por semana.
- **Attendance**: Porcentagem de aulas frequentadas.
- **Parental_Involvement**: Nível de envolvimento dos pais na educação do estudante (Baixo, Médio, Alto).
- **Access_to_Resources**: Disponibilidade de recursos educacionais (Baixo, Médio, Alto).
- **Extracurricular_Activities**: Participação em atividades extracurriculares (Sim, Não).
- **Sleep_Hours**: Média de horas de sono por noite.
- **Previous_Scores**: Notas obtidas em exames anteriores.
- **Motivation_Level**: Nível de motivação do estudante (Baixo, Médio, Alto).
- **Internet_Access**: Acesso à internet (Sim, Não).
- **Tutoring_Sessions**: Número de sessões de tutoria por mês.
- **Family_Income**: Nível de renda familiar (Baixo, Médio, Alto).
- **Teacher_Quality**: Qualidade dos professores (Baixo, Médio, Alto).
- **School_Type**: Tipo de escola (Pública, Privada).
- **Peer_Influence**: Influência dos colegas no desempenho acadêmico (Positiva, Neutra, Negativa).
- **Physical_Activity**: Média de horas de atividade física por semana.
- **Learning_Disabilities**: Presença de dificuldades de aprendizagem (Sim, Não).
- **Parental_Education_Level**: Nível de educação dos pais (Ensino Médio, Graduação, Pós-Graduação).
- **Distance_from_Home**: Distância de casa até a escola (Perto, Moderada, Longe).
- **Gender**: Gênero do estudante (Masculino, Feminino).
- **Exam_Score**: Nota final no exame.
- **Change_Grades**: Mudança nas notas em relação ao exame anterior (Aumento, Diminuição, Sem Mudança).
""")

# Finalização da introdução
st.write("""
A seguir, exploraremos os dados para responder a essas perguntas e identificar insights relevantes sobre o desempenho dos estudantes.
""")

# ===== Rodapé ===== #
st.write("---")
st.write("""
📚 **Projeto Interdisciplinar - 2024**  
🎓 Desenvolvido por **Davi Vieira, Guilherme Leonardo e Ronaldo Araújo**
""")
