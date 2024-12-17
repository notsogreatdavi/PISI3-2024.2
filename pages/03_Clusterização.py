import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


df_unistudents = pd.read_csv(
    "/content/Factors_ affecting_ university_student_grades_dataset.csv",
    low_memory=False,
)
pd.set_option("display.max_columns", None)

df_unistudents["Family_Income"] = pd.to_numeric(
    df_unistudents["Family_Income"], errors="coerce"
)
df_unistudents["Family_Income"].dtype
# Numericas
num_imputer = SimpleImputer(strategy="mean")
num_col = df_unistudents.select_dtypes(include=["float64", "int32"]).columns
df_unistudents[num_col] = num_imputer.fit_transform(df_unistudents[num_col])
df_unistudents[num_col] = df_unistudents[num_col].apply(np.round)

# Categoricas
cat_imputer = SimpleImputer(strategy="most_frequent")
cat_col = df_unistudents.select_dtypes(include=["object"]).columns
df_unistudents[cat_col] = cat_imputer.fit_transform(df_unistudents[cat_col])

colunas_categoricas = [
    "Gender",
    "Parental_Education",
    "Previous_Grades",
    "Class_Participation",
    "Major",
    "School_Type",
    "Financial_Status",
    "Parental_Involvement",
    "Educational_Resources",
    "Motivation",
    "Self_Esteem",
    "Stress_Levels",
    "School_Environment",
    "Professor_Quality",
    "Extracurricular_Activities",
    "Nutrition",
    "Physical_Activity",
    "Educational_Tech_Use",
    "Peer_Group",
    "Bullying",
    "Study_Space",
    "Learning_Style",
    "Tutoring",
    "Mentoring",
    "Lack_of_Interest",
    "Sports_Participation",
    "Grades",
]


label_encoder = LabelEncoder()

# Aplica o LabelEncoder a cada coluna categórica
for column in colunas_categoricas:
    df_unistudents[column] = label_encoder.fit_transform(
        df_unistudents[column].astype(str)
    )

    unique_values = df_unistudents[column].unique()
    inverse_transformed = label_encoder.inverse_transform(unique_values)
