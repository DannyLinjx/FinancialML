import pandas as pd
from process import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

y_pred_rf = pd.read_csv('./ft_test.csv').values
y_pred_xgb = pd.read_csv('./xgb_test.csv').values
y_pred_lstm = pd.read_csv('./lstm_test.csv').values
print(y_pred_rf)

data = pd.read_csv("./data/sh600000.csv")
data = process_data(data)
data.drop(['date','ticker','qfq_factor'], axis=1, inplace=True)
X_train = np.array(data.iloc[:, :-1].values)
y_train = np.array(data.iloc[:, -1].values)

x_train, x_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2,random_state=42)

# Initialize lists to store MSE and R2 values for each model
mse_list_rf, mse_list_xgb, mse_list_lstm = [], [], []
r2_list_rf, r2_list_xgb, r2_list_lstm = [], [], []

# 打印R²列表以检查其内容


# Calculate individual MSE and R2 for each prediction
for true, pred_rf, pred_xgb, pred_lstm in zip(y_test, y_pred_rf, y_pred_xgb, y_pred_lstm):
    mse_list_rf.append(mean_squared_error([true], [pred_rf]))
    r2_list_rf.append(r2_score([true], [pred_rf]))
    mse_list_xgb.append(mean_squared_error([true], [pred_xgb]))
    r2_list_xgb.append(r2_score([true], [pred_xgb]))
    mse_list_lstm.append(mean_squared_error([true], [pred_lstm]))
    r2_list_lstm.append(r2_score([true], [pred_lstm]))

# Plotting MSE for each model with different line styles and colors
plt.figure(figsize=(12, 6))
plt.plot(mse_list_rf[:100], label='RF MSE', color='yellow', linestyle='-.')  # Dashed-dot line
plt.plot(mse_list_xgb[:100], label='XGB MSE', color='green', linestyle='--')  # Dashed line
plt.plot(mse_list_lstm[:100], label='LSTM MSE', color='red', linestyle=':')  # Dotted line
plt.title('Individual MSE for RF, XGB, and LSTM Models')
plt.xlabel('Sample Index')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

# Plotting individual predictions and actual values with different line styles
plt.figure(figsize=(12, 6))
plt.plot(y_test[:100], label='Y TEST', color='orange', linestyle='-')  # Solid line
plt.plot(y_pred_rf[:100], label='RF PRED', color='yellow', linestyle='-.')  # Dashed-dot line
plt.plot(y_pred_xgb[:100], label='XGB PRED', color='green', linestyle='--')  # Dashed line
plt.plot(y_pred_lstm[:100], label='LSTM PRED', color='red', linestyle=':')  # Dotted line
plt.title('Individual values for Y Test, RF, XGB, and LSTM Models')
plt.xlabel('Sample Index')
plt.ylabel('Y value')
plt.legend()
plt.grid(True)
plt.show()