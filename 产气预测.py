#!/usr/bin/env python
# coding: utf-8
import os

import numpy as np
import pandas as pd
import torch
import plotly.graph_objs as go
from torch.utils.data import DataLoader
import streamlit as st
import warnings
import datetime
warnings.filterwarnings("ignore")

import gas_tft.expt_settings.configs
from gas_tft.data_formatters.base import InputTypes
import gas_tft.tft_model
from gas_tft.data_formatters import window_generator


ExperimentConfig = gas_tft.expt_settings.configs.ExperimentConfig
config = ExperimentConfig('gas_production', 'outputs')
data_formatter = config.make_data_formatter()
data_csv_path = 'dataset/GasProduction.csv'


if __name__ == '__main__':
    raw_data = pd.read_csv(data_csv_path)
    raw_data = raw_data[pd.to_datetime(raw_data['Date']) >= '2017-01-01']
    train, valid, test, train_and_val = data_formatter.split_data(raw_data)
    # Sets up default params
    data_formatter.set_scalers(train)
    # Use all data for label encoding  to handle labels not present in training.
    test_transformed = data_formatter.transform_inputs(test)
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params

    fixed_params.update(params)
    fixed_params['batch_first'] = True
    fixed_params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fixed_params['quantiles'] = [0.5]
    #
    model = gas_tft.tft_model.TFT(fixed_params).to(fixed_params['device'])
    # model.load_state_dict(torch.load(config.model_folder + '/gas_production_best_model_loss.pth', map_location=torch.device('cpu')))
    model.load_state_dict(torch.load('gas_tft/outputs/saved_models/gas_production/gas_production_best_model_loss_45.pth', map_location=torch.device('cpu')))
    model.eval()

    well_no_paths = ['dataset/WellsClass{}.npy'.format(i) for i in range(1, 4)]
    wells_classes = [np.load(well_no_path, allow_pickle=True) for well_no_path in well_no_paths]
    wells_all = np.concatenate(wells_classes)

    l_horizon = fixed_params['total_time_steps'] - fixed_params['num_encoder_steps']


    st.title("产气量预测")
    # well_test = st.sidebar.selectbox(label='请输入要进行测试的气井号', options=(df['WellNo'].unique()))
    well = st.sidebar.selectbox(label='请输入要进行预测的气井号', options=wells_all)
    date_input = st.date_input(label='请输入日期', value=pd.to_datetime('2022-08-16'))
    with st.form("my_form"):
        submitted = st.form_submit_button("submit")
        if submitted:
            df_well = raw_data[raw_data['WellNo'] == well]
            df_tmp = pd.DataFrame()
            df_tmp['WellNo'] = pd.Series([well] * l_horizon)
            df_tmp['WellNo'] = df_tmp['WellNo'].astype("category")
            dates_future = [pd.to_datetime(date_input) + datetime.timedelta(days=d) for d in range(l_horizon)]
            df_tmp['Date'] = pd.Series(dates_future)
            df_tmp['Allocation'] = df_well['Allocation'].iloc[-1]
            df_tmp['ElapsedProduction'] = pd.Series([df_well['ElapsedProduction'].iloc[-1] + i for i in range(1, l_horizon+1)])
            df_tmp['Daily_h'] = 24.0
            df_tmp['WellHeadPressure'] = 0.0
            df_tmp['CasingHeadPressure'] = 0.0
            df_tmp['WellHeadTemperature'] = 0.0
            df_tmp['Daily_104m3'] = 0.0
            df_well = pd.concat([df_well, df_tmp])
            df_well_transformed = data_formatter.transform_inputs(df_well)

            input_cols = [
                tup[0]
                for tup in fixed_params['column_definition']
                if tup[2] not in {InputTypes.ID, InputTypes.TIME}
            ]
            mean, std = data_formatter.target_scaler
            df_result = pd.DataFrame()
            df_result['Date'] = pd.Series(dates_future)

            data_window = df_well_transformed.iloc[-fixed_params['total_time_steps']:][input_cols]
            data_window = torch.from_numpy(data_window.values).unsqueeze(0)
            output, all_inputs, attention_components = model(data_window)
            forecast = torch.squeeze(output).tolist()
            forecast = np.array(forecast) * std + mean
            df_result['forecast'] = forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_well['Date'][-fixed_params['total_time_steps']:-l_horizon],
                y=df_well['Daily_104m3'][-fixed_params['total_time_steps']:-l_horizon],
                name='Historical'
            ))

            fig.add_trace(go.Scatter(
                x=df_result['Date'],
                y=df_result['forecast'],
                name='Forecast'
            ))
            st.plotly_chart(fig)
            st.write(df_result)

    st.subheader("测试结果展示")
    well_test = st.selectbox(label='请输入要进行测试的气井号', options=wells_all)
    with st.form("my_form1"):
        submitted = st.form_submit_button("submit")
        if submitted:
            df_well = raw_data[raw_data['WellNo'] == well_test]
            df_well_transformed = data_formatter.transform_inputs(df_well)

            input_cols = [
                tup[0]
                for tup in fixed_params['column_definition']
                if tup[2] not in {InputTypes.ID, InputTypes.TIME}
            ]
            mean, std = data_formatter.target_scaler
            df_result = df_well[['WellNo', 'Date', 'Daily_h', 'Daily_104m3']]
            df_result['forecast'] = 0

            predictions = []
            start_ids = []
            for i in range(df_well.shape[0] - fixed_params['total_time_steps'], 0, -l_horizon):
                start_ids.append(i)
            start_ids = sorted(start_ids)
            for i in start_ids:
                data_window = df_well_transformed.iloc[i:i + fixed_params['total_time_steps']][input_cols]
                data_window = torch.from_numpy(data_window.values).unsqueeze(0)
                output, all_inputs, attention_components = model(data_window)
                predictions = predictions + torch.squeeze(output).tolist()
            predictions = np.array(predictions) * std + mean
            df_result.iloc[start_ids[0] + fixed_params['num_encoder_steps']:, -1] = predictions
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_result['Date'][:-l_horizon],
                y=df_result['Daily_104m3'][:-l_horizon],
                name='Historical'
            ))

            fig.add_trace(go.Scatter(
                x=df_result['Date'],
                y=df_result['forecast'],
                name='Forecast'
            ))
            fig.add_trace(go.Scatter(
                x=df_result['Date'][-l_horizon:],
                y=df_result['Daily_104m3'][-l_horizon:],
                name='Actual'
            ))
            st.plotly_chart(fig)


