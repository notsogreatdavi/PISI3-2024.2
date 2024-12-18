import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df_students = pd.read_parquet("../data/unistudents.parquet")

# Divide o dataframe em x e y
X = df_students.drop('Change_Grades', axis=1)
y = df_students['Change_Grades']

# Carregar colunas categóricas em X
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Escalar colunas numéricas em X
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Transformar a Change_Grades em numérica
le = LabelEncoder()
y = le.fit_transform(y)

# Divisão do teste de treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Definindo os aprendizes base
base_learners = [
                 ('rf', RandomForestClassifier(n_estimators=50, random_state=101)),
                 ('ada', AdaBoostClassifier(n_estimators=50, random_state=101)),
                 ('gb', GradientBoostingClassifier(n_estimators=50, random_state=101)),
                 ('xgb', XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='mlogloss', random_state=101))
                ]

# Inicio do vote classifier
voting = VotingClassifier(estimators=base_learners, voting='soft')
voting.fit(X_train, y_train)

# Inicío do stacking clasifier
stacking = StackingClassifier(estimators=base_learners, final_estimator=XGBClassifier())
stacking.fit(X_train, y_train)

# Fazendo previsões com os dois modelos
voting_pred = voting.predict(X_test)
stacking_pred = stacking.predict(X_test)

print("Voting Classifier:")
print(classification_report(y_test, voting_pred, target_names=le.classes_))

print("Stacking Classifier:")
print(classification_report(y_test, stacking_pred, target_names=le.classes_))
