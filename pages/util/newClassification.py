import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o dataset
df_students = pd.read_parquet("../../data/new_unistudents.parquet")

# Separar as features e a coluna alvo
X = df_students.drop('Change_Grades', axis=1)
y = df_students['Change_Grades']

# Codificar colunas categóricas
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    
# Escalar colunas numéricas
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Codificar a coluna alvo
le = LabelEncoder()
y = le.fit_transform(y)

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Definir os modelos
modelos = {
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=101),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=101),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=101)
}

# Função para treinar e avaliar o modelo
def treinar_e_avaliar(modelo, X_train, y_train, X_test, y_test):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Acurácia
    acuracia = accuracy_score(y_test, y_pred)
    print(f"**Acurácia:** {acuracia:.2f}")

    # Matriz de Confusão
    print("**Matriz de Confusão:**")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    plt.show()

    # Relatório de Classificação
    print("**Relatório de Classificação:**")
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print(report)

    # Importância das variáveis (para modelos baseados em árvores)
    if hasattr(modelo, 'feature_importances_'):
        print("**Importância das Variáveis:**")
        importances = modelo.feature_importances_
        importances_df = pd.DataFrame({'Variável': X.columns, 'Importância': importances})
        importances_df = importances_df.sort_values(by='Importância', ascending=False)

        fig_imp, ax_imp = plt.subplots()
        sns.barplot(x='Importância', y='Variável', data=importances_df.head(10), palette='viridis', ax=ax_imp)
        ax_imp.set_title("Top 10 Variáveis mais Importantes")
        plt.show()

# Avaliar cada modelo
for nome_modelo, modelo in modelos.items():
    print(f"### Avaliação do Modelo: {nome_modelo} ###")
    treinar_e_avaliar(modelo, X_train, y_train, X_test, y_test)
    print("\n" + "#" * 50 + "\n")
