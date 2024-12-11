import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('student_performance_data_updated.csv')
print(data.head())
print(data.isnull().sum())
data[['StudyTimeWeekly', 'Absences', 'GPA']] = data[['StudyTimeWeekly', 'Absences', 'GPA']].apply(pd.to_numeric, errors='coerce')
X = data[['StudyTimeWeekly', 'Absences']]
y = data['GPA']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

plt.scatter(y_test, y_pred, alpha=0.7, label='Predictions')
plt.xlabel('Actual GPA')
plt.ylabel('Predicted GPA')
plt.title('Actual vs Predicted GPA')
plt.legend()
plt.show()
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted GPA')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

print(model.coef_)
print(model.intercept_)
