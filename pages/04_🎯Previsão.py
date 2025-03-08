import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# Título da página
st.title("Predição de Desempenho Escolar")

# Descrição do objetivo do modelo
st.write("""
## Objetivo da Predição
Este modelo de inteligência artificial foi desenvolvido para prever se haverá um aumento, uma diminuição ou uma 
manutenção nas notas escolares com base em diversos fatores socioeconômicos e comportamentais do estudante.

A partir das informações que você inserir abaixo, o sistema utilizará um algoritmo de aprendizado de máquina 
para analisar padrões similares em milhares de casos anteriores e determinar a tendência mais provável para o 
desempenho acadêmico.

**Importante:** Os intervalos de valores sugeridos são baseados nos dados utilizados para treinar o modelo. 
Embora você possa inserir valores fora desses intervalos, isso pode resultar em previsões menos precisas, 
pois o modelo não foi treinado com esses valores extremos.
""")

# Carregar o dataset para obter informações sobre as colunas
try:
    df_students = pd.read_parquet("./data/new_unistudents.parquet")
    st.write("Por favor, insira os dados abaixo para prever o desempenho escolar:")
except Exception as e:
    st.error(f"Erro ao carregar o dataset para obter informações das colunas: {e}")
    st.stop()

# Definições de categorias agrupadas
categorical_groups = {
    "Parental_Involvement": ["Baixo", "Médio", "Alto"],
    "Access_to_Resources": ["Baixo", "Médio", "Alto"],
    "Motivation_Level": ["Baixo", "Médio", "Alto"],
    "Family_Income": ["Baixo", "Médio", "Alto"],
    "Teacher_Quality": ["Baixo", "Médio", "Alto"],
    "School_Type": ["Público", "Privado"],
    "Peer_Influence": ["Positivo", "Neutro", "Negativo"],
    "Parental_Education_Level": ["Ensino Médio", "Graduação", "Pós-Graduação"],
    "Distance_from_Home": ["Perto", "Moderado", "Longe"]
}

# Mapeamento para colunas OneHot
column_mapping = {
    "Parental_Involvement": {
        "Baixo": {"Parental_Involvement_Low": 1, "Parental_Involvement_Medium": 0},
        "Médio": {"Parental_Involvement_Low": 0, "Parental_Involvement_Medium": 1},
        "Alto": {"Parental_Involvement_Low": 0, "Parental_Involvement_Medium": 0}
    },
    "Access_to_Resources": {
        "Baixo": {"Access_to_Resources_Low": 1, "Access_to_Resources_Medium": 0},
        "Médio": {"Access_to_Resources_Low": 0, "Access_to_Resources_Medium": 1},
        "Alto": {"Access_to_Resources_Low": 0, "Access_to_Resources_Medium": 0}
    },
    "Motivation_Level": {
        "Baixo": {"Motivation_Level_Low": 1, "Motivation_Level_Medium": 0},
        "Médio": {"Motivation_Level_Low": 0, "Motivation_Level_Medium": 1},
        "Alto": {"Motivation_Level_Low": 0, "Motivation_Level_Medium": 0}
    },
    "Family_Income": {
        "Baixo": {"Family_Income_Low": 1, "Family_Income_Medium": 0},
        "Médio": {"Family_Income_Low": 0, "Family_Income_Medium": 1},
        "Alto": {"Family_Income_Low": 0, "Family_Income_Medium": 0}
    },
    "Teacher_Quality": {
        "Baixo": {"Teacher_Quality_Low": 1, "Teacher_Quality_Medium": 0},
        "Médio": {"Teacher_Quality_Low": 0, "Teacher_Quality_Medium": 1},
        "Alto": {"Teacher_Quality_Low": 0, "Teacher_Quality_Medium": 0}
    },
    "School_Type": {
        "Público": {"School_Type_Public": 1},
        "Privado": {"School_Type_Public": 0}
    },
    "Peer_Influence": {
        "Positivo": {"Peer_Influence_Neutral": 0, "Peer_Influence_Positive": 1},
        "Neutro": {"Peer_Influence_Neutral": 1, "Peer_Influence_Positive": 0},
        "Negativo": {"Peer_Influence_Neutral": 0, "Peer_Influence_Positive": 0}
    },
    "Parental_Education_Level": {
        "Ensino Médio": {"Parental_Education_Level_High School": 1, "Parental_Education_Level_Postgraduate": 0},
        "Graduação": {"Parental_Education_Level_High School": 0, "Parental_Education_Level_Postgraduate": 0},
        "Pós-Graduação": {"Parental_Education_Level_High School": 0, "Parental_Education_Level_Postgraduate": 1}
    },
    "Distance_from_Home": {
        "Perto": {"Distance_from_Home_Moderate": 0, "Distance_from_Home_Near": 1},
        "Moderado": {"Distance_from_Home_Moderate": 1, "Distance_from_Home_Near": 0},
        "Longe": {"Distance_from_Home_Moderate": 0, "Distance_from_Home_Near": 0}
    }
}

# Dicionário de descrições das colunas
column_descriptions = {
    "Hours_Studied": "Horas Estudadas (por semana)",
    "Attendance": "Frequência na Escola (porcentagem de aulas assistidas)",
    "Sleep_Hours": "Horas de Sono (média por noite)",
    "Tutoring_Sessions": "Sessões de Tutoria (quantidade por mês)",
    "Physical_Activity": "Atividade Física (horas por semana)",
    
    # Campos categóricos agrupados
    "Parental_Involvement": "Envolvimento dos Pais (nível de participação na educação do estudante)",
    "Access_to_Resources": "Acesso a Recursos Educacionais (disponibilidade de materiais e recursos)",
    "Motivation_Level": "Nível de Motivação (motivação do estudante para os estudos)",
    "Family_Income": "Renda Familiar (nível de renda da família)",
    "Teacher_Quality": "Qualidade dos Professores (nível de qualidade do ensino)",
    "School_Type": "Tipo de Escola (categoria da instituição)",
    "Peer_Influence": "Influência dos Colegas (como os colegas afetam o desempenho acadêmico)",
    "Parental_Education_Level": "Nível de Educação dos Pais (maior nível educacional dos pais)",
    "Distance_from_Home": "Distância de Casa (proximidade entre casa e escola)",
    
    # Campos binários
    "Extracurricular_Activities_Yes": "Atividades Extracurriculares (participação em atividades fora do currículo)",
    "Internet_Access_Yes": "Acesso à Internet (disponibilidade de acesso à internet)",
    "Learning_Disabilities_Yes": "Dificuldades de Aprendizagem (presença de dificuldades específicas)",
    "Class_Participation_Yes": "Participação em Aula (participação ativa nas aulas)",
    "Social_Support_Yes": "Apoio Social (presença de rede de apoio social)",
    "School_Environment_Positive": "Ambiente Escolar Positivo (qualidade do ambiente escolar)",
    "Performance_Anxiety_Yes": "Ansiedade de Desempenho (presença de ansiedade relacionada às avaliações)",
    "Peer_Pressure_Yes": "Pressão dos Colegas (pressão exercida pelo grupo social)",
    "Bullying_Yes": "Bullying (experiência como vítima de bullying)",
    "Family_Relationship_Poor": "Relacionamento Familiar Ruim (qualidade das relações familiares)"
}

# Criar inputs para cada categoria (agrupando colunas relacionadas)
inputs = {}

# 1. Primeiro tratamos os campos numéricos
numeric_cols = ["Hours_Studied", "Attendance", "Sleep_Hours", "Tutoring_Sessions", "Physical_Activity"]
for col in numeric_cols:
    if col in df_students.columns:
        min_value = float(df_students[col].min())
        max_value = float(df_students[col].max())
        default_value = (min_value + max_value) / 2
        
        st.write(f"### {column_descriptions.get(col, col)}")
        st.write(f"*Valores típicos entre {min_value:.1f} e {max_value:.1f}*")
        
        # Permitir qualquer valor (sem mínimo ou máximo rígidos)
        inputs[col] = st.number_input(
            f"{column_descriptions.get(col, col)}:",
            value=default_value,
            step=0.1
        )

# 2. Campos categóricos agrupados
st.write("## Categorias")
categorical_inputs = {}
for category, options in categorical_groups.items():
    if category in column_descriptions:
        st.write(f"### {column_descriptions.get(category, category)}")
        categorical_inputs[category] = st.selectbox(
            f"Selecione o nível para {category}:",
            options=options
        )

# 3. Campos binários (sim/não)
st.write("## Campos Sim/Não")
binary_cols = [col for col in df_students.columns if df_students[col].max() == 1 and df_students[col].min() == 0 
               and col not in ['Parental_Involvement_Low', 'Parental_Involvement_Medium', 
                              'Access_to_Resources_Low', 'Access_to_Resources_Medium',
                              'Motivation_Level_Low', 'Motivation_Level_Medium',
                              'Family_Income_Low', 'Family_Income_Medium',
                              'Teacher_Quality_Low', 'Teacher_Quality_Medium',
                              'School_Type_Public', 
                              'Peer_Influence_Neutral', 'Peer_Influence_Positive',
                              'Parental_Education_Level_High School', 'Parental_Education_Level_Postgraduate',
                              'Distance_from_Home_Moderate', 'Distance_from_Home_Near',
                              'Gender_Male', 'Gender_Female']]

for col in binary_cols:
    if col in column_descriptions:
        st.write(f"### {column_descriptions.get(col, col)}")
        inputs[col] = 1 if st.selectbox(f"{column_descriptions.get(col, col)}:", ["Não", "Sim"]) == "Sim" else 0

# Campo para Gênero
st.write("### Gênero")
genero = st.selectbox("Selecione o gênero:", options=["Masculino", "Feminino"])

# Botão de Previsão
if st.button("Prever Desempenho"):
    # Processar entradas categóricas agrupadas
    for category, value in categorical_inputs.items():
        if category in column_mapping and value in column_mapping[category]:
            for col, val in column_mapping[category][value].items():
                inputs[col] = val

    # Criar DataFrame com os dados de entrada
    input_data = pd.DataFrame([inputs])
    
    # Garantir que todas as colunas necessárias estejam presentes
    X = df_students.drop("Change_Grades", axis=1)
    for col in X.columns:
        if col not in input_data.columns and col not in ['Gender_Male', 'Gender_Female']:
            input_data[col] = 0

    # Adicionar o gênero
    input_data["Gender_Male"] = 1 if genero == "Masculino" else 0
    
    # Pré-processamento dos dados
    num_cols = X.select_dtypes(include=["float64", "int64"]).columns
    
    # Inicializar variáveis importantes
    model = None
    scaler = None
    le = None
    
    try:
        # Verificar se os arquivos do modelo existem
        model_path = "./models/gradient_boosting_model.pkl"
        scaler_path = "./models/standard_scaler.pkl"
        label_path = "./models/label_encoder.pkl"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(label_path):
            # Carregar o modelo e o scaler pré-treinados
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            with open(label_path, "rb") as f:
                le = pickle.load(f)
                
            st.success("✅ Modelo pré-treinado carregado com sucesso!")
        else:
            # Fallback: Treinar um modelo temporário
            st.warning("⚠️ Modelo pré-treinado não encontrado. Treinando um modelo temporário...")
            
            # Criar e treinar o scaler
            scaler = StandardScaler()
            X[num_cols] = scaler.fit_transform(X[num_cols])
            
            # Treinar o modelo usando os dados disponíveis
            model = GradientBoostingClassifier(n_estimators=50, random_state=101)
            model.fit(X, df_students["Change_Grades"])
            
            st.info("ℹ️ Para obter resultados mais precisos, exporte o modelo na página de Classificação.")
    except Exception as e:
        st.error(f"Erro ao carregar ou treinar o modelo: {e}")
        st.stop()
    
    try:
        # Escalar as colunas numéricas usando o scaler (carregado ou recém-treinado)
        input_data[num_cols] = scaler.transform(input_data[num_cols])
        
        # Reordenar as colunas para corresponder à ordem de X
        input_data = input_data[X.columns]
        
        # Fazer a previsão
        prediction = model.predict(input_data)
        
        # Mapeamento para garantir a tradução correta dos valores numéricos para classes
        class_mapping = {"0": "No Change", "1": "Decrease", "2": "Increase", 
                        0: "No Change", 1: "Decrease", 2: "Increase"}
        
        # Obter a predição como string para garantir compatibilidade
        pred_value = str(prediction[0])
        prediction_class = class_mapping.get(pred_value, pred_value)
        
        if prediction_class not in class_mapping.values():
            prediction_class = prediction[0]  # Usar o valor bruto se não conseguir mapear

        # Traduzir o resultado da previsão
        resultado = {
            "Increase": "Haverá um possível aumento nas notas (Increase)",
            "Decrease": "Haverá uma possível diminuição nas notas (Decrease)",
            "No Change": "Haverá um possível mantimento nas notas (No Change)"
        }

        # Exibir a previsão traduzida
        st.subheader("Resultado da Predição:")
        st.write(f"{resultado.get(prediction_class, prediction_class)}")
        
    except Exception as e:
        st.error(f"Erro ao fazer a previsão: {e}")
        st.exception(e)  # Mostrar detalhes do erro para depuração