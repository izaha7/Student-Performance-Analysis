import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
file_path = 'Student_performance_data.csv'
data = pd.read_csv(file_path)
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Student Performance')
plt.show()