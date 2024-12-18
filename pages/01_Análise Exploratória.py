# Importação das bibliotecas necessárias
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração de estilo para gráficos
sns.set(style="whitegrid", palette="husl")

# Função para carregar dados
@st.cache_data
def load_data(filepath):
    df = pd.read_parquet(filepath)
    return df

# Função para exibir valores ausentes
def missing_values_analysis(df):
    st.subheader("📊 Análise de Valores Ausentes")
    missing_percent = (df.isnull().sum() / len(df)) * 100
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
    
    if not missing_percent.empty:
        st.write("**Percentual de Valores Ausentes por Coluna:**")
        st.bar_chart(missing_percent)
        st.write("""
        **Insight**: Colunas com valores ausentes podem impactar a análise. 
        Considere remover ou preencher com a média/mediana para variáveis numéricas 
        ou o valor mais frequente para variáveis categóricas.
        """)
    else:
        st.success("Não há valores ausentes no dataset.")

# Função para exibir variáveis categóricas
def categorical_analysis(df):
    st.subheader("🔢 Distribuição de Variáveis Categóricas")
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        selected_cat = st.selectbox("Selecione uma variável categórica para visualização:", cat_cols)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x=selected_cat, palette="husl", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.write(f"""
        **Insight**: A variável **{selected_cat}** apresenta {df[selected_cat].nunique()} categorias únicas. 
        Analise se há desequilíbrio ou presença de categorias pouco representativas.
        """)
    else:
        st.warning("Não foram encontradas variáveis categóricas no dataset.")

# Função para exibir gráfico hexbin
def hexbin_analysis(df):
    st.subheader("🔷 Gráfico Hexbin")
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(num_cols) > 1:
        x_var = st.selectbox("Selecione a variável X:", num_cols, index=0)
        y_var = st.selectbox("Selecione a variável Y:", num_cols, index=1)
        gridsize = st.slider("Tamanho do Grid (granularidade):", min_value=10, max_value=50, value=30)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        hb = ax.hexbin(df[x_var], df[y_var], gridsize=gridsize, cmap='Blues', mincnt=1)
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label('Contagem')
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        st.pyplot(fig)
        st.write(f"""
        **Insight**: O gráfico hexbin é útil para identificar padrões de densidade entre as variáveis 
        **{x_var}** e **{y_var}**. Verifique regiões de maior concentração de pontos.
        """)
    else:
        st.warning("Não há variáveis numéricas suficientes para o gráfico hexbin.")

# Função principal
def main():
    # Carregar datasets
    st.sidebar.title("Configurações")
    raw_file_path = st.sidebar.text_input("Caminho do Dataset Bruto:", "./data/unistudents.parquet")
    treated_file_path = st.sidebar.text_input("Caminho do Dataset Tratado:", "./data/unistudents.parquet")
    
    st.sidebar.write("---")
    st.sidebar.write("**⚙️ Carregando Dados...**")
    
    if st.sidebar.button("Carregar Dados"):
        raw_data = load_data(raw_file_path)
        treated_data = load_data(treated_file_path)
        
        # Tabs
        st.title("📊 Análise Exploratória de Dados")
        tabs = st.tabs(["🔍 Visão Geral", "📊 Análise de Valores Nulos", "🔢 Variáveis Categóricas", "🔷 Hexbin"])
        
        # Tab 1: Visão Geral
        with tabs[0]:
            st.header("🔍 Visão Geral dos Dados")
            st.write("**Dados Brutos:**")
            st.dataframe(raw_data.head())
            st.write("**Dados Tratados:**")
            st.dataframe(treated_data.head())
            
            st.write("""
            **Questões para Reflexão:**
            - Como os dados foram tratados? Quais transformações foram aplicadas?
            - Existem diferenças significativas entre os datasets bruto e tratado?
            """)
        
        # Tab 2: Valores Nulos
        with tabs[1]:
            st.header("📊 Valores Nulos nos Dados Brutos")
            missing_values_analysis(raw_data)
            
            st.header("📊 Valores Nulos nos Dados Tratados")
            missing_values_analysis(treated_data)
        
        # Tab 3: Variáveis Categóricas
        with tabs[2]:
            st.header("🔢 Análise das Variáveis Categóricas - Dados Brutos")
            categorical_analysis(raw_data)
            
            st.header("🔢 Análise das Variáveis Categóricas - Dados Tratados")
            categorical_analysis(treated_data)
        
        # Tab 4: Hexbin
        with tabs[3]:
            st.header("🔷 Análise de Densidade com Hexbin - Dados Brutos")
            hexbin_analysis(raw_data)
            
            st.header("🔷 Análise de Densidade com Hexbin - Dados Tratados")
            hexbin_analysis(treated_data)

if __name__ == "__main__":
    main()
