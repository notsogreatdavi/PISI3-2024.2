import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from imblearn.over_sampling import SMOTE

# Configuração da Página
st.set_page_config(
    page_title="Classificação de Desempenho Escolar", page_icon="📊", layout="wide"
)
st.title("📊 Classificação de Desempenho Escolar")
st.write("---")

# Carregar o Dataset
st.subheader("1. Carregando e Explorando o Dataset")
try:
    df_students = pd.read_parquet("./data/new_unistudents.parquet")
    st.success("Dataset carregado com sucesso!")
    if "Change_Grades" in df_students.columns:
        st.write("**Distribuição da variável alvo (Change_Grades):**")
        distribution = df_students["Change_Grades"].value_counts().sort_index()
        st.bar_chart(distribution)
    else:
        st.warning("A variável 'Change_Grades' não foi encontrada no dataset!")
except Exception as e:
    st.error(f"Erro ao carregar o dataset: {e}")
    st.stop()

# Pré-processamento
st.subheader("2. Pré-processamento dos Dados")
try:
    X = df_students.drop("Change_Grades", axis=1)
    y = df_students["Change_Grades"]

    num_cols = X.select_dtypes(include=["float64", "int64"]).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Aplicar SMOTE para balancear as classes
    smote = SMOTE(random_state=101)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    # Dividir os dados balanceados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=101
    )

    st.success("Dados pré-processados com sucesso!")
    st.success("Classes balanceadas com SMOTE!")
    st.write(f"**Tamanho do conjunto de treino:** {X_train.shape[0]} linhas")
    st.write(f"**Tamanho do conjunto de teste:** {X_test.shape[0]} linhas")
    
    # Decodificar os valores numéricos de volta para os rótulos originais
    y_balanced_labels = le.inverse_transform(y_balanced)

    # Criar um DataFrame para exibir a nova distribuição
    distribution_balanced = pd.Series(y_balanced_labels).value_counts().sort_index()

    # Exibir o gráfico com os rótulos originais
    st.write("**Distribuição das classes após o balanceamento:**")
    st.bar_chart(distribution_balanced)
except Exception as e:
    st.error(f"Erro no pré-processamento: {e}")
    st.stop()

# Função para Gerar o Gráfico SHAP
def plot_shap(modelo, X_train, X_test):
    st.write("**Gráfico SHAP:**")
    try:
        try:
            explainer = shap.TreeExplainer(modelo)
            shap_values = explainer.shap_values(X_test)
        except Exception as e_tree:
            st.warning(
                "TreeExplainer não suportou esse modelo. Utilizando KernelExplainer..."
            )
            X_sample = shap.sample(X_train, 100, random_state=101)
            explainer = shap.KernelExplainer(modelo.predict, X_sample)
            shap_values = explainer.shap_values(X_test, nsamples=100)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1] if len(
                shap_values) > 1 else shap_values[0]
        else:
            shap_vals = shap_values
        shap.summary_plot(shap_vals, X_test, plot_type="dot", show=False)
        fig = plt.gcf()
        st.pyplot(fig)
        plt.clf()
    except Exception as e:
        st.error(f"Erro ao gerar gráfico SHAP: {e}")

# Seleção de Modelos
st.subheader("3. Treinando e Avaliando os Modelos")
modelos = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,  # Número de árvores
        max_depth=10,  # Profundidade máxima das árvores
        min_samples_split=10,  # Mínimo de amostras para dividir um nó
        min_samples_leaf=5,  # Mínimo de amostras em uma folha
        random_state=101,
    ),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=101),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=101),
}
modelo_selecionado = st.selectbox(
    "Escolha um modelo de classificação:", list(modelos.keys())
)
botao_avaliar = st.button("Treinar e Avaliar Modelo")

# Função para Treinar e Avaliar
def treinar_e_avaliar(modelo, X_train, y_train, X_test, y_test):
    # Treinar o modelo
    modelo.fit(X_train, y_train)
    y_train_pred = modelo.predict(X_train)
    y_test_pred = modelo.predict(X_test)

    # Acurácia
    acuracia_treino = accuracy_score(y_train, y_train_pred)
    acuracia_teste = accuracy_score(y_test, y_test_pred)
    st.write(f"**Acurácia (Treino):** {acuracia_treino:.2f}")
    st.write(f"**Acurácia (Teste):** {acuracia_teste:.2f}")

    # Matriz de Confusão
    st.write("**Matriz de Confusão (Teste):**")
    cm_teste = confusion_matrix(y_test, y_test_pred)
    fig_teste, ax_teste = plt.subplots(figsize=(5, 4), dpi=120)
    sns.heatmap(
        cm_teste,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax_teste,
    )
    ax_teste.set_xlabel("Predito")
    ax_teste.set_ylabel("Real")
    st.pyplot(fig_teste)

    # Relatório de Classificação
    st.write("**Relatório de Classificação (Treino):**")
    report_treino = classification_report(
        y_train, y_train_pred, target_names=le.classes_, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report_treino).transpose())

    st.write("**Relatório de Classificação (Teste):**")
    report_teste = classification_report(
        y_test, y_test_pred, target_names=le.classes_, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report_teste).transpose())

    # Importância das Variáveis
    if hasattr(modelo, "feature_importances_"):
        st.write("**Importância das Variáveis:**")
        importances = modelo.feature_importances_
        importances_df = pd.DataFrame(
            {"Variável": X.columns, "Importância": importances}
        )
        importances_df = importances_df.sort_values(
            by="Importância", ascending=False
        )

        fig_imp, ax_imp = plt.subplots(figsize=(6, 4), dpi=120)
        sns.barplot(
            x="Importância",
            y="Variável",
            data=importances_df.head(10),
            palette="viridis",
            ax=ax_imp,
        )
        ax_imp.set_title("Top 10 Variáveis mais Importantes")
        st.pyplot(fig_imp)
        
    # Chamada do gráfico SHAP para o modelo treinado
    plot_shap(modelo, X_train, X_test)

# Avaliar Modelo Selecionado
if botao_avaliar:
    st.write(f"### Avaliação do Modelo: **{modelo_selecionado}**")
    treinar_e_avaliar(modelos[modelo_selecionado], X_train, y_train, X_test, y_test)

# Exportação do Modelo
st.subheader("4. Exportação do Modelo para Predição")
st.write("""
Após a análise comparativa dos modelos de classificação, você pode escolher qual modelo 
deseja exportar para ser utilizado na interface de predição.

Compare as métricas de desempenho e exporte o modelo que apresentou melhor resultado 
para o seu caso de uso.
""")

if st.button(f"Exportar Modelo {modelo_selecionado}"):
    try:
        # Treinar o modelo com todos os dados balanceados para máxima precisão
        modelo_final = modelos[modelo_selecionado]
        modelo_final.fit(X_balanced, y_balanced)
        
        # Criar diretório para modelos se não existir
        os.makedirs("./models", exist_ok=True)
        
        # Salvar o modelo para uso na página de predição
        with open(f"./models/{modelo_selecionado.lower().replace(' ', '_')}_model.pkl", "wb") as f:
            pickle.dump(modelo_final, f)
        
        # Salvar também o LabelEncoder para interpretar as classes de saída
        with open("./models/label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)
        
        # Salvar o StandardScaler para normalizar os dados de entrada
        with open("./models/standard_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
            
        st.success(f"✅ Modelo {modelo_selecionado} exportado com sucesso! O algoritmo está pronto para realizar predições em tempo real na página de previsão.")
        
        # Detalhes técnicos para desenvolvedores
        with st.expander("Detalhes técnicos da exportação"):
            st.info(f"""
            Os seguintes artefatos foram salvos no diretório './models/':
            - {modelo_selecionado.lower().replace(' ', '_')}_model.pkl: Modelo de classificação treinado
            - label_encoder.pkl: Codificador para interpretação das classes de saída
            - standard_scaler.pkl: Normalizador para padronização dos dados de entrada
            
            Estes arquivos serão automaticamente utilizados na interface de predição.
            """)
            
    except Exception as e:
        st.error(f"Erro ao exportar o modelo: {e}")