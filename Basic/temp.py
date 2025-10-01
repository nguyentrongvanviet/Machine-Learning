import pandas as pd 

data = pd.read_csv('test.csv') 

data.describe().to_csv('temp.csv', index=True) 
data.columns