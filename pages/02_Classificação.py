import numpy as np 
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

file_path = '..\data\unistudents.parquet'
df = pd.read_parquet(file_path)
print(df.head(35))