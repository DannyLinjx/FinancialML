import numpy as np
import pandas as pd
import json
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
# from statsmodels.tsa.stattools import adfuller
# from ruptures import detect
import ruptures as rpt
import matplotlib.pyplot as plt

stock_data = pd.read_csv('archive/stock_data.csv')
ticker_info = pd.read_csv('archive/ticker_info.csv')

"""
The "open", "close", "high" & "low" are adjusted stock prices
(keeping the latest stock prices unchanged and adjusting the stock prices back into the past).
The adjustment factor is "qfq_factor" in this dataset.

Prices are adjusted for consistency reasons,
and mostly used for designing and backtesting quantitative trading strategies of any kind.

If users want to get the true historical stock prices at the time,
just multiplying "open", "close", "high" , "low" by "qfq_factor", separately,
to reconstruct the true historical prices.
"""

def split_stock_data(tickers, data):
    tickers = tickers

    stocks_split_data = {}

    for idx,code in enumerate(tickers['ticker'].values):
        if code not in stocks_split_data:
            m = data[data['ticker'] == code].values.tolist()
            print(f"This is the {idx+1}-th stock\n")
            stocks_split_data[code] = m

    # save data into json
    with open('data/stock_data.json', 'w', encoding='utf-8') as f:
        json.dump(stocks_split_data, f, ensure_ascii=False, indent=4)

def get_stock_data(path,code=None):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("Getting our data ...")
    if code is not None:
        pd.DataFrame(data[code],columns=['date','ticker','open',
                                         'high','low','close','volume',
                                         'outstanding_share','turnover',
                                         'pe','pe_ttm','pb','ps','ps_ttm',
                                         'dv_ratio','dv_ttm','total_mv','qfq_factor']).to_csv('./data/'+code+'.csv',index=False)
        return data[code]
    else:
        return

def process_data(data):
    # 计算每列的空缺值数量
    missing_values_count = data.isna().sum()
    print("Missing values in each column:")
    print(missing_values_count)

    # 使用每列的均值填补空缺值
    for column in data.columns:
        if data[column].isna().any():  # 检查列是否含有空缺值
            mean_value = data[column].mean()  # 计算该列的均值
            data[column].fillna(mean_value, inplace=True)  # 填补空缺值

    # 对open, high, low, close列分别乘以qfq_factor列的值
    data['open'] = data['open'] * data['qfq_factor']
    data['high'] = data['high'] * data['qfq_factor']
    data['low'] = data['low'] * data['qfq_factor']
    data['close'] = data['close'] * data['qfq_factor']

    """
        针对交易量volume，流通股outstanding_share以及总市值total_mv用(x-x')/x来替换，其中x'是均值
    """
    data['volume'] = (data['volume'] - data['volume'].mean()) / data['volume']
    data['outstanding_share'] = (data['outstanding_share'] - data['outstanding_share'].mean()) / data['outstanding_share']
    data['total_mv'] = (data['total_mv'] - data['total_mv'].mean()) / data['total_mv']

    # 删除qfq_factor列
    # data.drop(['date','ticker','qfq_factor'], axis=1, inplace=True)

    return data

# ARIMA时间序列分析，用于RF和Xgboost
def arima_analysis(data):
    new_data = {}
    for name in data.columns:
        # 时间序列数据预处理
        time_series = data[name].values  # 每一列都是时间序列数据，调整新的数据列

        # 拟合ARIMA模型
        arima_model = ARIMA(time_series, order=(1, 1, 1))  # 这里的order参数(p,d,q)需要根据实际情况调整
        arima_result = arima_model.fit()

        # ARIMA预测和残差计算
        arima_pred = arima_result.predict()
        residuals = (time_series - arima_pred) * 0.8

        # 将残差添加到特征集中
        new_data[name] = residuals

    return pd.DataFrame(new_data)

# 季节性分析
def seasonal_analysis(data, column, freq):
    """
    对指定列的时间序列数据进行季节性分解。
    :param data: 包含时间序列的DataFrame。
    :param column: 要分析的列名。
    :param freq: 数据的季节周期，例如，如果数据是每天的，则一年的季节周期为365。
    :return: 分解对象，包含趋势、季节性和残差成分。
    """
    # 确保日期列是日期时间格式，这对于时间序列分析是必需的
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    # 提取需要分析的时间序列
    time_series = data[column]

    # 进行季节性分解
    result = seasonal_decompose(time_series, model='additive', period=freq)

    # 绘制分解结果
    plt.figure(figsize=(14, 7))
    result.plot()
    plt.show()

    return result

# 周期性分析
def cyclical_analysis(data, column, lags):
    """
    对时间序列数据进行周期性分析，绘制自相关函数（ACF）和偏自相关函数（PACF）。
    :param data: 包含时间序列的DataFrame。
    :param column: 要分析的列名，该列应为数值类型。
    :param lags: 考虑的滞后数量，这影响了ACF和PACF图的细节。
    :return: None，函数将直接显示图表。
    """
    # 确保日期列是日期时间格式，这对于时间序列分析是必需的
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    # 提取指定列的数据
    time_series = data[column]

    # 绘制 ACF 图
    plt.figure(figsize=(12, 6))
    plot_acf(time_series, lags=lags, alpha=0.05)
    plt.title('Autocorrelation Function (ACF)')
    plt.show()

    # 绘制 PACF 图
    plt.figure(figsize=(12, 6))
    plot_pacf(time_series, lags=lags, alpha=0.05)
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.show()

# 波动性分析
def volatility_analysis(data, column, window):
    """
    对时间序列数据进行波动性分析，通过计算滚动标准差来衡量。
    :param data: 包含时间序列的DataFrame。
    :param column: 要分析的列名，该列应为数值类型。
    :param window: 滚动标准差计算的窗口大小。
    :return: None，函数将直接显示滚动标准差的图表。
    """
    # 确保日期列是日期时间格式，这对于时间序列分析是必需的
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    # 提取指定列的数据
    time_series = data[column]

    # 计算滚动标准差
    rolling_std = time_series.rolling(window=window).std()

    # 绘制结果
    plt.figure(figsize=(12, 6))
    rolling_std.plot(title=f'Rolling Standard Deviation ({window}-day window)')
    plt.xlabel('Date')
    plt.ylabel('Rolling Standard Deviation')
    plt.show()

# 异常值和突变点检测
def detect_anomalies_and_changes(data, column, window, sigma=3):
    """
    对时间序列数据进行异常值和突变点检测。
    :param data: 包含时间序列的DataFrame。
    :param column: 要分析的列名，该列应为数值类型。
    :param window: 滚动标准差计算的窗口大小用于异常值检测。
    :param sigma: 用于定义异常值的标准差倍数。
    :return: 包含异常值和突变点的图表。
    """
    # 确保日期列是日期时间格式，这对于时间序列分析是必需的
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    # 提取指定列的数据
    time_series = data[column]

    # 异常值检测：计算滚动均值和滚动标准差
    rolling_mean = time_series.rolling(window=window).mean()
    rolling_std = time_series.rolling(window=window).std()

    # 定义异常值条件
    lower_bound = rolling_mean - (sigma * rolling_std)
    upper_bound = rolling_mean + (sigma * rolling_std)

    # 突变点检测
    points = np.array(time_series)
    model = "l1"  # 使用L1范数作为成本函数，适用于大多数情况
    algo = rpt.Pelt(model=model, min_size=1, jump=5).fit(points)
    result = algo.predict(pen=10)

    # 绘制结果
    plt.figure(figsize=(14, 7))
    plt.plot(time_series, label='Data')
    plt.plot(rolling_mean, color='yellow', label='Rolling Mean')
    plt.fill_between(time_series.index, lower_bound, upper_bound, color='gray', alpha=0.5, label='Normal Bounds')

    # 标记异常值
    anomalies = time_series[(time_series < lower_bound) | (time_series > upper_bound)]
    plt.scatter(anomalies.index, anomalies, color='red', label='Anomalies')

    plt.legend()
    plt.title(f'Anomaly and Change Point Detection for {column}')
    plt.show()

# 相关性分析(多变量分析)
def multivariate_analysis(data, variables, maxlag):
    """
    对时间序列数据集进行多变量分析和因果关系检验。
    :param data: 包含多个时间序列的DataFrame。
    :param variables: 要分析的变量列表。
    """
    # 确保日期列是日期时间格式，这对于时间序列分析是必需的
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    # 变量之间的相关性分析
    plt.figure(figsize=(10, 8))
    sns.heatmap(data[variables].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.xticks(rotation=45, ha='right')
    plt.title("Correlation Matrix")
    plt.show()

# 因果关系检验
def granger_causality_matrix(data, variables, maxlag):
    """
    执行Granger因果检验并以热图形式展示结果。
    :param data: 包含多个时间序列的DataFrame。
    :param variables: 要分析的变量列表。
    :param maxlag: Granger因果检验的最大滞后数。
    :return: 绘制热图展示因果关系的P值。
    """
    # 确保日期列是日期时间格式，这对于时间序列分析是必需的
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    results = pd.DataFrame(index=variables, columns=variables, data=np.zeros((len(variables), len(variables))))

    for var in variables:
        for caused in variables:
            if var != caused:
                test_result = grangercausalitytests(data[[var, caused]], maxlag=maxlag, verbose=False)
                p_values = [test_result[i + 1][0]['ssr_chi2test'][1] for i in range(maxlag)]
                min_p_value = np.min(p_values)
                results.loc[var, caused] = min_p_value

    plt.figure(figsize=(12, 8))
    sns.heatmap(results, annot=True, cmap='coolwarm', fmt=".4f", vmin=0, vmax=0.05)
    plt.xticks(rotation=45, ha='right')
    plt.title('Granger Causality Test Results (Min P-Value)')
    plt.show()

if __name__ == '__main__':
    # split_stock_data(ticker_info, stock_data)
    # 获取特定股票数据
    # data = get_stock_data('data/stock_data.json',code='sh600000')
    # 采用浦发银行的股票行情
    data = pd.read_csv("./data/sh600000.csv")
    data = process_data(data)

    # 如果数据是每日的，并假设想要分析年度季节性，周期设为365，以总市值total_mv为例
    # result = seasonal_analysis(data, 'total_mv', 365)

    # lags = 40，表示我们考虑40个时间滞后
    # cyclical_analysis(data, 'close', 40)

    # window = 30，表示我们考虑30天的滚动窗口
    # volatility_analysis(data, 'total_mv', 30)

    # window = 30，表示我们考虑30天的滚动窗口
    # detect_anomalies_and_changes(data, 'dv_ttm', 30, sigma=3)

    # 假设 'data' 是已经加载的DataFrame，我们关注所有的列
    # variables = ['open', 'high', 'low', 'close', 'volume', 'out_shares', 'turnover',
    #              'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm']
    # maxlag = 5  # Set max lag for causality tests
    # data.rename(columns={'outstanding_share': 'out_shares'}, inplace=True)
    # multivariate_analysis(data, variables, maxlag)

    # 假设 'data' 是已经加载的DataFrame，我们关注所有的列
    variables = ['open','high','low','close',
                 'volume','out_shares','turnover',
                 'pe','pe_ttm','pb','ps','ps_ttm','dv_ratio',
                 'dv_ttm']  # 示例变量列表
    maxlag = 5  # 最大滞后数设为5
    data.rename(columns={'outstanding_share': 'out_shares'}, inplace=True)
    granger_causality_matrix(data, variables, maxlag)
