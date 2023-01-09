#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.manifold import LocallyLinearEmbedding

import pickle
import plotly.graph_objs as go
import os
import streamlit as st
import datetime
pd.set_option('plotting.backend', 'plotly')
import warnings

warnings.filterwarnings('ignore')

test_length = 7
model_path_C3 = 'pages/liquidC3_lgb.bin'
model_path_C2 = 'pages/liquidC2_lgb.bin'
model_path_C1 = 'pages/liquidC1_lgb.bin'

if __name__ == '__main__':
    df_gas_liquid = pd.read_csv('dataset/LiquidProduction.csv')
    df_gas_liquid['Date'] = pd.to_datetime(df_gas_liquid['Date'])
    df_gas_liquid['elapsed'] = (df_gas_liquid['Date'] - df_gas_liquid['Date'][0]).apply(lambda x: x.days) + 1

    lag_days = [i for i in range(1, 5)]
    df_gas_liquid = df_gas_liquid.assign(**{
        'lag_{}_C1'.format(l): df_gas_liquid['LiquidProductionC1'].transform(lambda x: x.shift(l)) for l in
        lag_days})
    df_gas_liquid = df_gas_liquid.assign(**{
        'lag_{}_C2'.format(l): df_gas_liquid['LiquidProductionC2'].transform(lambda x: x.shift(l)) for l in
        lag_days})
    df_gas_liquid = df_gas_liquid.assign(**{
        'lag_{}_C3'.format(l): df_gas_liquid['LiquidProductionC3'].transform(lambda x: x.shift(l)) for l in
        lag_days})

    featuresC1 = ['elapsed', 'GasProductionC1']  # + ['lag_{}_C1'.format(i) for i in lag_days]
    featuresC2 = ['elapsed', 'GasProductionC2']  # + ['lag_{}_C2'.format(i) for i in lag_days]
    featuresC3 = ['elapsed', 'GasProductionC3']  # + ['lag_{}_C3'.format(i) for i in lag_days]

    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 30,
        'min_data_in_leaf': 4,  # 叶子结点的最小数据数，避免过拟合
        'max_depth': 6,  # 单颗树的最大深度
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'learning_rate': 0.05,
        'max_bin': 1000,
        'n_estimators': 1000,
        'verbose': -1,
    }

    X_train_C1 = df_gas_liquid[featuresC1].iloc[:-test_length, :]
    y_train_C1 = df_gas_liquid['LiquidProductionC1'][:-test_length]
    X_test_C1 = df_gas_liquid[featuresC1].iloc[-test_length:, :]
    y_test_C1 = df_gas_liquid['LiquidProductionC1'][-test_length:]
    train_data_C1 = lgb.Dataset(X_train_C1, y_train_C1)
    if not os.path.exists(model_path_C1):
        regressor_C1 = lgb.train(lgb_params, train_data_C1, verbose_eval=100)
        pickle.dump(regressor_C1, open(model_path_C1, 'wb'))
    else:
        regressor_C1 = pickle.loads(open(model_path_C1, 'rb').read())
    y_pred_C1 = regressor_C1.predict(X_test_C1)
    mae = mean_absolute_error(y_test_C1, y_pred_C1)
    print('C1 MAE: {}'.format(mae))

    X_train_C2 = df_gas_liquid[featuresC2].iloc[:-test_length, :]
    y_train_C2 = df_gas_liquid['LiquidProductionC2'][:-test_length]
    X_test_C2 = df_gas_liquid[featuresC2].iloc[-test_length:, :]
    y_test_C2 = df_gas_liquid['LiquidProductionC2'][-test_length:]
    train_data_C2 = lgb.Dataset(X_train_C2, y_train_C2)
    if not os.path.exists(model_path_C2):
        regressor_C2 = lgb.train(lgb_params, train_data_C2, verbose_eval=100)
        pickle.dump(regressor_C2, open(model_path_C2, 'wb'))
    else:
        regressor_C2 = pickle.loads(open(model_path_C2, 'rb').read())
    y_pred_C2 = regressor_C2.predict(X_test_C2)
    mae = mean_absolute_error(y_test_C2, y_pred_C2)
    print('C2 MAE: {}'.format(mae))

    X_train_C3 = df_gas_liquid[featuresC3].iloc[:-test_length, :]
    y_train_C3 = df_gas_liquid['LiquidProductionC3'][:-test_length]
    X_test_C3 = df_gas_liquid[featuresC3].iloc[-test_length:, :]
    y_test_C3 = df_gas_liquid['LiquidProductionC3'][-test_length:]
    train_data_C3 = lgb.Dataset(X_train_C3, y_train_C3)
    if not os.path.exists(model_path_C3):
        regressor_C3 = lgb.train(lgb_params, train_data_C3, verbose_eval=100)
        pickle.dump(regressor_C3, open(model_path_C3, 'wb'))
    else:
        regressor_C3 = pickle.loads(open(model_path_C3, 'rb').read())
    y_pred_C3 = regressor_C3.predict(X_test_C3)
    mae = mean_absolute_error(y_test_C3, y_pred_C3)
    print('C3 MAE: {}'.format(mae))

    regressor = {'C1': regressor_C1, 'C2': regressor_C2, 'C3': regressor_C3}
    st.header("产液量预测")
    station = st.sidebar.selectbox(label='请选择集气站', options=['C1', 'C2', 'C3'], index=2)
    date_input = st.sidebar.date_input(label='请输入当前日期', value=pd.to_datetime('2021-12-31'))
    elasped = (pd.to_datetime(date_input) - df_gas_liquid['Date'][0]).days + 1
    gas_production = st.sidebar.number_input('请输入当日预计产气量', value=395.75)
    # lag_1 = st.sidebar.number_input(str(pd.to_datetime(date_input) - datetime.timedelta(days=1))[:10] + ' ' + station + ' 产液量:', value=142.78)
    # lag_2 = st.sidebar.number_input(str(pd.to_datetime(date_input) - datetime.timedelta(days=2))[:10] + ' ' + station + ' 产液量:', value=105.94)
    # lag_3 = st.sidebar.number_input(str(pd.to_datetime(date_input) - datetime.timedelta(days=3))[:10] + ' ' + station + ' 产液量:', value=128.68)
    # lag_4 = st.sidebar.number_input(str(pd.to_datetime(date_input) - datetime.timedelta(days=4))[:10] + ' ' + station + ' 产液量:', value=99.55)
    with st.form("my_form"):
        submitted = st.form_submit_button("submit")
        if submitted:
            # forecast = regressor[station].predict([[elasped, gas_production, lag_1, lag_2, lag_3, lag_4]])
            forecast = regressor[station].predict([[elasped, gas_production]])

            st.write('*'+str(pd.to_datetime(date_input))[:10] + ' ' + station + ' 预测产液量:   {}*'.format(forecast[0]))

    st.subheader('测试结果展示')
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_gas_liquid['Date'][-test_length:],
        y=y_test_C1,
        name='Actual'
    ))

    fig.add_trace(go.Scatter(
        x=df_gas_liquid['Date'][-test_length:],
        y=y_pred_C1,
        name='Forecast'
    ))
    fig.update_layout(title_text='C1 预测产液量和真实产液量')
    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_gas_liquid['Date'][-test_length:],
        y=y_test_C2,
        name='Actual'
    ))

    fig.add_trace(go.Scatter(
        x=df_gas_liquid['Date'][-test_length:],
        y=y_pred_C2,
        name='Forecast'
    ))
    fig.update_layout(title_text='C2 预测产液量和真实产液量')
    st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_gas_liquid['Date'][-test_length:],
        y=y_test_C3,
        name='Actual'
    ))

    fig.add_trace(go.Scatter(
        x=df_gas_liquid['Date'][-test_length:],
        y=y_pred_C3,
        name='Forecast'
    ))
    fig.update_layout(title_text='C3 预测产液量和真实产液量')
    st.plotly_chart(fig)




