import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from process import *
import matplotlib.pyplot as plt

data = pd.read_csv('./data/sh600000.csv')
data = process_data(data)
data.drop(['date','ticker','qfq_factor'], axis=1, inplace=True)

# ARIMA时间序列分析
data = arima_analysis(data)

X_train = np.array(data.iloc[:, :-1].values)
y_train = np.array(data.iloc[:, -1].values)

x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize the scaler
scaler = StandardScaler()
# Fit and transform the training data
X_train = scaler.fit_transform(X_train)
# Transform the test data
X_test = scaler.transform(x_test)

# Define XGBoost model
xgb = XGBRegressor(random_state=42)

# Set the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of boosting rounds
    'max_depth': [3, 6, 9],  # Maximum depth of a tree
    'learning_rate': [0.1, 0.01, 0.001],  # Learning rate
    'min_child_weight': [1, 3, 5],  # Minimum sum of instance weight needed in a child
    'subsample': [0.6, 0.8, 1.0],  # Subsample ratio of the training instances
    'colsample_bytree': [0.6, 0.8, 1.0],  # Subsample ratio of columns when constructing each tree
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Train the model
grid_search.fit(x_train, y_train)

# Output the best parameters and corresponding MSE
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best score (MSE): {grid_search.best_score_}")

# Use the best parameters for prediction
best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(x_test)
print(y_pred)
pd.DataFrame(y_pred).to_csv('xgb_test.csv',index=False)
# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Test Mean Squared Error: {mse}")

# R^2 coefficient
r2 = r2_score(y_test, y_pred)
print(f"Test R2 is: {r2}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Values', color='blue')
plt.plot(y_pred, label='Predicted Values', color='red', linestyle='--')
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.legend()
plt.grid(True)
plt.show()
