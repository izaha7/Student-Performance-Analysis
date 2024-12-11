import pandas as pd
data = pd.read_csv('Student_performance_data.csv')
print(data.shape)
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
print(data.shape)
data.to_csv('Student_performance_data.csv', index=False)
