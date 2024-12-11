import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


test_data = pd.DataFrame({
    'Age': [18],
    'StudyTimeWeekly': [12],
    'Absences': [3]
})

predicted_gpa = model.predict(test_data)
print(f'Predicted GPA: {predicted_gpa[0]:.2f}')
