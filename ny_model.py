# -*- coding: utf-8 -*-
# @Time : 2023/6/8 20:46
# @Author : Wangbb
# @FileName: ny_ny_model.py
from sklearnex import patch_sklearn

patch_sklearn()  # 启动加速补丁
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression, Lasso
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error  # MSE
import matplotlib

matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['simsun']})
import warnings

warnings.filterwarnings("ignore")
import math


def scale(df, x_len):
    # print(df)
    min_1, max_1 = min(df[:x_len]), max(df[:x_len])

    return ((df - min_1) / (max_1 - min_1)) * 0.8 + 0.1

def outlier(df):
    ave = df.mean()
    std_3 = df.std()*3
    outlier_up =(df >(ave + std_3)).sum()
    outlier_down = (df <(ave - std_3)).sum()
    sort_df = df.sort_values(ascending=False).values
    df[df > (ave + std_3)] = sort_df[outlier_up]
    df[df < (ave - std_3)] = sort_df[-outlier_down-1]
    return df

def nengyuan_model(diqu, df, r, jiange, params_input):
    y_len = df.iloc[:, 1].notnull().sum()
    x_len = df.iloc[:, 2].notnull().sum()
    # 删除多于列和多余行
    df = df.iloc[:x_len, :6]
    df.columns = ['地区', 'GDP能耗强度下降率', '二产用电', '三产用电', '高耗能企业用电', '居民用电']

    if params_input:
        if '异常值替换' in params_input:
            df['GDP能耗强度下降率'] = outlier(df['GDP能耗强度下降率'])


        if '交互项' in params_input:
            # 交互项
            df['二产&三产'] = np.sqrt(df['二产用电'] * df['三产用电'])
            df['二产&高耗能'] = np.sqrt(df['二产用电'] * df['高耗能企业用电'])
            df['二产&居民'] = np.sqrt(df['二产用电'] * df['居民用电'])
            df['三产&高耗能'] = np.sqrt(df['三产用电'] * df['高耗能企业用电'])
            df['三产&居民用电'] = np.sqrt(df['三产用电'] * df['居民用电'])
            df['高能耗&居民'] = np.sqrt(df['高耗能企业用电'] * df['居民用电'])

        if '归一化' in params_input:
            for column in df.columns[2:]:
                df[column] = scale(df[column], x_len)

    # 选取自变量
    x = df.iloc[:, 2:]
    y = df.iloc[:y_len, 1]

    for i in jiange:
        x[f'二产用电{i}'] = x['二产用电'] ** i
        x[f'三产用电{i}'] = x['三产用电'] ** i
        x[f'高耗能企业用电{i}'] = x['高耗能企业用电'] ** i
        x[f'居民用电{i}'] = x['居民用电'] ** i

    if params_input:
        if '交互项' in params_input:
            x_list = ['二产&三产', '二产&高耗能', '二产&居民', '三产&高耗能', '三产&居民用电', '高能耗&居民']
    else:
        x_list = []
    column_list = ['二产用电', '三产用电', '高耗能企业用电', '居民用电', 'r2']
    # 获取要预测的季节
    column_list.extend(list(df.iloc[y_len:, 0].values))
    score_df = pd.DataFrame(columns=column_list)

    for dot1 in jiange:
        for dot2 in jiange:
            for dot3 in jiange:
                for dot4 in jiange:
                    x_columns = [f'二产用电{dot1}', f'三产用电{dot2}', f'高耗能企业用电{dot3}', f'居民用电{dot4}']
                    x_columns.extend(x_columns)
                    x_ = x[x_columns]
                    trainx = x_[:y_len]
                    testx = x_[y_len:]
                    model = LinearRegression().fit(trainx, y)
                    ypre = model.predict(trainx)
                    r2 = r2_score(y, ypre)
                    if r2 > r:
                        predict = model.predict(testx).reshape(1, -1)
                        value_list = [dot1, dot2, dot3, dot4, round(r2, 4)]
                        value_list.extend(list(predict[0]))
                        score_df.loc[len(score_df)] = value_list

    score_df = score_df.sort_values(by='r2', ascending=False)
    return score_df, df


def provice_model(df, r, jiange, params_input):
    y_len = df.iloc[:, 1].notnull().sum()
    x_len = df.iloc[:, 2].notnull().sum()
    df = df.iloc[:x_len, :13]
    df.columns = ['季节', '能耗下降率', '杭州', '宁波', '温州', '绍兴', '湖州', '嘉兴', '金华', '衢州', '台州', '丽水', '舟山']

    if params_input:
        if '异常值替换' in params_input:
            for column in df.columns[2:]:
                df[column] = outlier(df[column])

        if '归一化' in params_input:
            for column in df.columns[1:]:
                df[column] = scale(df[column], x_len)

    x = df[['杭州', '宁波', '温州', '绍兴', '湖州', '嘉兴', '金华', '衢州', '台州', '丽水', '舟山']]
    y = df['能耗下降率'][:y_len]

    for i in jiange:
        x[f'杭州{i}'] = x['杭州'] ** i
        x[f'宁波{i}'] = x['宁波'] ** i
        x[f'温州{i}'] = x['温州'] ** i
        x[f'绍兴{i}'] = x['绍兴'] ** i
        x[f'湖州{i}'] = x['湖州'] ** i
        x[f'嘉兴{i}'] = x['嘉兴'] ** i
        x[f'金华{i}'] = x['金华'] ** i
        x[f'衢州{i}'] = x['衢州'] ** i
        x[f'台州{i}'] = x['台州'] ** i
        x[f'丽水{i}'] = x['丽水'] ** i
        x[f'舟山{i}'] = x['舟山'] ** i

    column_list = ['杭州', '宁波', '温州', '绍兴', '湖州', '嘉兴', '金华', '衢州', '台州', '丽水', '舟山', 'r2']
    column_list.extend(df['季节'][y_len:].values)
    score_df = pd.DataFrame(columns=column_list)
    for dot1 in jiange:
        for dot2 in jiange:
            for dot3 in jiange:
                for dot4 in jiange:
                    for dot5 in jiange:
                        for dot6 in jiange:
                            for dot7 in jiange:
                                for dot8 in jiange:
                                    for dot9 in jiange:
                                        for dot10 in jiange:
                                            for dot11 in jiange:
                                                x_ = x[[f'杭州{dot1}', f'宁波{dot2}', f'温州{dot3}', f'绍兴{dot4}', f'湖州{dot5}',
                                                        f'嘉兴{dot6}', f'金华{dot7}', f'衢州{dot8}', f'台州{dot9}',
                                                        f'丽水{dot10}', f'舟山{dot11}']]
                                                x_t = x_[:y_len]
                                                x_p = x_[y_len:]

                                                model = LinearRegression().fit(x_t.values, y.values)
                                                r2 = r2_score(y, model.predict(x_t))
                                                if r2 > r:
                                                    ypre = model.predict(np.array(x_p)).reshape(1, -1)[0]
                                                    value_list = [dot1, dot2, dot3, dot4, dot5, dot6, dot7, dot8, dot9,
                                                                  dot10, dot11,r2]
                                                    value_list.extend(list(ypre))
                                                    score_df.loc[len(score_df)] = value_list
    score_df = score_df.sort_values(by='r2', ascending=False)
    return score_df, df
