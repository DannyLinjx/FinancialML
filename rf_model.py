import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

x_train, x_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2,random_state=42)

# 初始化标准化器
scaler = StandardScaler()
# 对训练数据进行拟合和转换
X_train = scaler.fit_transform(X_train)
# 对测试数据进行转换
X_test = scaler.transform(x_test)

# 定义随机森林模型
rf = RandomForestRegressor(random_state=42)

# 设置GridSearchCV的参数网格
param_grid = {
    'n_estimators': [100, 200, 300],  # 树的数量
    'max_features': ['auto', 'sqrt'],  # 每次分裂考虑的最大特征数
    'max_depth': [10, 20, None],  # 树的最大深度
    'min_samples_split': [2, 10, 20],  # 内部节点再划分所需最小样本数
    'min_samples_leaf': [1, 4, 10]  # 叶子节点最少样本数
}

# 使用GridSearchCV进行自动参数调优
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# 训练模型
grid_search.fit(x_train, y_train)

# 输出最优参数和对应的MSE
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best score (MSE): {grid_search.best_score_}")

# 使用最佳参数的模型进行预测
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(x_test)
# print(y_pred)
# pd.DataFrame(y_pred).to_csv('ft_test.csv',index=False)
# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
print(f"Test Mean Squared Error: {mse}")

# R^2系数
r2 = r2_score(y_test, y_pred)
print(f"Test R2 is: {r2}")

# 绘制真实值和预测值
plt.figure(figsize=(10, 6))  # 设置图形的尺寸
plt.plot(y_test, label='Actual Values', color='blue')  # 绘制实际值
plt.plot(y_pred, label='Predicted Values', color='red', linestyle='--')  # 绘制预测值
plt.title('Comparison of Actual and Predicted Values')  # 添加标题
plt.xlabel('Sample Index')  # X轴标签
plt.ylabel('Target Value')  # Y轴标签
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.show()  # 显示图形

