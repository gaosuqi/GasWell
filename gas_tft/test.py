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
warnings.filterwarnings("ignore")

import expt_settings.configs
from data_formatters.base import InputTypes
import tft_model
from data_formatters import window_generator


def plot(df_well, l_horizon, mode='test'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_well['date'][:-l_horizon],
        y=df_well['actual'][:-l_horizon],
        name='Historical'
    ))

    fig.add_trace(go.Scatter(
        x=df_well['date'][-l_horizon:],
        y=df_well['forecast'][-l_horizon:],
        name='Forecast'
    ))
    if mode == 'test':
        fig.add_trace(go.Scatter(
            x=df_well['date'][-l_horizon:],
            y=df_well['actual'][-l_horizon:],
            name='Actual'
        ))

    fig.show()
    return fig


def cal_metrics_by_class(wells_class, result_path, result_stat_path, actual_and_forecast, l_horizon, is_plot=False):
    """
    按类别计算评估指标
    :param wells_class: 要测试类别的所有气井号
    :param result_path: metrics保存路径
    :param result_stat_path: metrics统计值保存路径
    :param actual_and_forecast: 测试数据的真实值和预测值，pd.DataFrame()类型
    :param l_horizon: 预测长度
    :return: metrics, metrics_stat
    """
    if not os.path.exists(result_path) or not os.path.exists(result_stat_path):
        metrics = pd.DataFrame()
        for well in wells_class:
            df_well = actual_and_forecast[actual_and_forecast['identifier'] == well]
            if df_well.empty:
                continue
            from sklearn.metrics import mean_absolute_error
            if not df_well['forecast'][-l_horizon:].isnull().any():
                mae = mean_absolute_error(df_well['forecast'][-l_horizon:], df_well['actual'][-l_horizon:])
                rcpe = abs(df_well['forecast'][-l_horizon:].sum() - df_well['actual'][-l_horizon:].sum()) / df_well['actual'][-l_horizon:].sum()
                if df_well['actual'][-l_horizon:].max() > 0.1:
                    metrics = pd.concat([metrics, pd.DataFrame([[well, mae, rcpe]])], axis=0)
                    if is_plot:
                        plot(df_well, l_horizon)
        metrics.columns = ['WellNo', 'MAE', 'RCPE']
        metrics.to_csv(result_path)
        metrics_stat = metrics.describe()
        metrics_stat.to_csv(result_stat_path)
    else:
        metrics = pd.read_csv(result_path, usecols=['WellNo', 'MAE', 'RCPE'])
        metrics_stat = pd.read_csv(result_stat_path)

    return metrics, metrics_stat


ExperimentConfig = expt_settings.configs.ExperimentConfig
config = ExperimentConfig('gas_production', 'outputs')
data_formatter = config.make_data_formatter()
data_csv_path = config.data_csv_path
test_csv_path = os.path.join(config.data_folder, 'GasProductionTFTTest.csv')
train_csv_path = os.path.join(config.data_folder, 'GasProductionTFTTrain.csv')
valid_csv_path = os.path.join(config.data_folder, 'GasProductionTFTValid.csv')


if __name__ == '__main__':
    raw_data = pd.read_csv(data_csv_path, index_col=0)
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
    model = tft_model.TFT(fixed_params).to(fixed_params['device'])
    # model.load_state_dict(torch.load(config.model_folder + '/gas_production_best_model_loss.pth', map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(config.model_folder + '/gas_production_best_model_loss_45.pth', map_location=torch.device('cpu')))
    model.eval()

    test_ds = window_generator.TSDataset(fixed_params, test_transformed, num_samples=-1)
    test_loader = DataLoader(
        test_ds,
        batch_size=fixed_params['minibatch_size'],
        num_workers=4,
        shuffle=False
    )

    dfs = []
    for idx, batch in enumerate(test_loader):
        with torch.no_grad():
            output, all_inputs, attention_components = model(batch['inputs'])
            batch_prediction = pd.DataFrame(
                output.detach().cpu().numpy()[:, :, 0],
                columns=[
                    time[0][0] for time in batch['time'][fixed_params['num_encoder_steps']:]
                ])
            cols = list(batch_prediction.columns)
            #         flat_prediction['forecast_time'] = batch['time'][:, 54 - 1, 0]
            batch_prediction['identifier'] = batch['identifier'][0][0]
            dfs.append(batch_prediction)

    all_predictions = pd.concat(dfs)
    all_predictions_unnormalized = data_formatter.format_predictions(all_predictions)

    forecast = pd.melt(all_predictions_unnormalized, id_vars='identifier', var_name='date', value_name='forecast')
    actual = test[['WellNo', 'Date', 'Cluster', 'Daily_104m3']]
    actual.columns = ['identifier', 'date', 'class', 'actual']
    actual_and_forecast = pd.merge(actual, forecast, on=['identifier', 'date'], how='left')

    well_no_paths = []
    for i in range(1, 4):
        well_no_paths.append(config.data_folder + '/WellNo/WellsClass{}.npy'.format(i))
    wells_classes = []
    for well_no_path in well_no_paths:
        wells_classes.append(np.load(well_no_path, allow_pickle=True))

    l_horizon = fixed_params['total_time_steps'] - fixed_params['num_encoder_steps']
    result_paths = []
    result_stat_paths = []
    for i in range(1, 4):
        result_paths.append(config.results_folder + '/metrics_class{}_{}.csv'.format(i, l_horizon))
        result_stat_paths.append(config.results_folder + '/metrics_class{}_{}_stat.csv'.format(i, l_horizon))

    for i in range(3):
        metrics, metrics_stat = cal_metrics_by_class(wells_classes[i], result_paths[i], result_stat_paths[i], actual_and_forecast, l_horizon)

    # l_horizon = fixed_params['total_time_steps'] - fixed_params['num_encoder_steps']
    # well_test = 'SN0083-01'
    # df_well = raw_data[raw_data['WellNo'] == well_test]
    # df_well_transformed = data_formatter.transform_inputs(df_well)
    #
    # input_cols = [
    #     tup[0]
    #     for tup in fixed_params['column_definition']
    #     if tup[2] not in {InputTypes.ID, InputTypes.TIME}
    # ]
    # mean, std = data_formatter.target_scaler
    # df_result = df_well[['WellNo', 'Date', 'Daily_h', 'Daily_104m3']]
    # df_result['forecast'] = 0
    #
    # predictions = []
    # start_ids = []
    # for i in range(df_well.shape[0] - fixed_params['total_time_steps'], 0, -l_horizon):
    #     start_ids.append(i)
    # start_ids = sorted(start_ids)
    # for i in start_ids:
    #     data_window = df_well_transformed.iloc[i:i + fixed_params['total_time_steps']][input_cols]
    #     data_window = torch.from_numpy(data_window.values).unsqueeze(0)
    #     output, all_inputs, attention_components = model(data_window)
    #     predictions = predictions + torch.squeeze(output).tolist()
    # predictions = np.array(predictions) * std + mean
    # df_result.iloc[start_ids[0] + fixed_params['num_encoder_steps']:, -1] = predictions
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #     x=df_result['Date'][:-l_horizon],
    #     y=df_result['Daily_104m3'][:-l_horizon],
    #     name='Historical'
    # ))
    #
    # fig.add_trace(go.Scatter(
    #     x=df_result['Date'],
    #     y=df_result['forecast'],
    #     name='Forecast'
    # ))
    # fig.add_trace(go.Scatter(
    #     x=df_result['Date'][-l_horizon:],
    #     y=df_result['Daily_104m3'][-l_horizon:],
    #     name='Actual'
    # ))
    # fig.show()