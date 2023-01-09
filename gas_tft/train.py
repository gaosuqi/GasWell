#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import expt_settings.configs
import tft_model
from data_formatters import window_generator
from losses.pinball_loss import PinballLoss
from losses.mae_loss import MAELoss
from losses.rce_loss import RCELoss

pd.options.display.max_columns = 1000

ExperimentConfig = expt_settings.configs.ExperimentConfig
config = ExperimentConfig('gas_production', 'outputs')
data_formatter = config.make_data_formatter()
data_csv_path = config.data_csv_path
# train_csv_path = os.path.join(config.data_folder, 'GasProductionTFTTrain.csv')
# valid_csv_path = os.path.join(config.data_folder, 'GasProductionTFTValid.csv')
# train_and_val_csv_path = os.path.join(config.data_folder, 'GasProductionTFTTrainAndVal.csv')
# test_csv_path = os.path.join(config.data_folder, 'GasProductionTFTTest.csv')

if __name__ == '__main__':
    raw_data = pd.read_csv(data_csv_path, index_col=0)
    train, valid, test, train_and_val = data_formatter.split_data(raw_data)
    # if not os.path.exists(test_csv_path) or not os.path.exists(valid_csv_path) or not os.path.exists(train_csv_path)\
    #         or not os.path.exists(train_and_val_csv_path):
    #     train, valid, test, train_and_val = data_formatter.split_data(raw_data)
    #     train.to_csv(train_csv_path, index=False)
    #     valid.to_csv(valid_csv_path, index=False)
    #     train_and_val.to_csv(train_and_val_csv_path, index=False)
    #     test.to_csv(test_csv_path, index=False)
    # else:
    #     train = pd.read_csv(train_csv_path)
    #     valid = pd.read_csv(valid_csv_path)
    #     train_and_val = pd.read_csv(train_and_val_csv_path)
    #     test = pd.read_csv(test_csv_path)
    # Sets up default params
    data_formatter.set_scalers(train)
    mean, std = data_formatter.get_mean_std()
    # Use all data for label encoding  to handle labels not present in training.
    fixed_params = data_formatter.get_experiment_params()

    params = data_formatter.get_default_model_params
    fixed_params.update(params)

    fixed_params['batch_first'] = True
    fixed_params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fixed_params['quantiles'] = [0.5]

    elect = window_generator.TSDataset(fixed_params, data_formatter.transform_inputs(train), num_samples=256)
    loader = DataLoader(
        elect,
        batch_size=fixed_params['minibatch_size'],
        num_workers=4,
        shuffle=False
    )

    valid_ds = window_generator.TSDataset(fixed_params, data_formatter.transform_inputs(valid), num_samples=1)
    valid_loader = DataLoader(
        valid_ds,
        batch_size=fixed_params['minibatch_size'],
        num_workers=4,
        shuffle=False
    )

    model = tft_model.TFT(fixed_params).to(fixed_params['device'])

    # q_loss_func = RMSSELoss(fixed_params['device'])
    # q_loss_func = SMAPELoss(fixed_params['device'])
    # q_loss_func = QuantileLoss(fixed_params['quantiles'])
    q_loss_func = PinballLoss(0.50, fixed_params['device'])
    mae_loss_func = MAELoss(mean, std, fixed_params['device'])

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    losses_mae = []
    losses_rce = []
    for i in range(fixed_params['num_epochs']):
        epoch_loss_train_mae = []
        progress_bar = tqdm(enumerate(loader))
        for batch_num, batch in progress_bar:
            optimizer.zero_grad()
            output, all_inputs, attention_components = model(batch['inputs'])
            # output.shape = [batch_size, forecast_horizon, 1],
            # all_inputs = [batch_size, total_time_steps, 5]
            loss = q_loss_func(output.squeeze(2), batch['outputs'][:, :, 0].float().to(fixed_params['device']))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), fixed_params['max_gradient_norm'])
            optimizer.step()

            mae_loss = mae_loss_func(output.squeeze(2),
                                     batch['outputs'][:, :, 0].float().to(fixed_params['device']))

            epoch_loss_train_mae.append(mae_loss.item())
        print("Epoch {}: MAE Train Loss = {}".format(i, np.mean(epoch_loss_train_mae)))

        epoch_loss_eval_mae = []
        for idx, batch in enumerate(valid_loader):
            with torch.no_grad():
                output, all_inputs, attention_components = model(batch['inputs'])
                mae_loss = mae_loss_func(output.squeeze(2),
                                         batch['outputs'][:, :, 0].float().to(fixed_params['device']))

                epoch_loss_eval_mae.append(mae_loss.item())
        print("Epoch {}: MAE Eval Loss = {}".format(i, np.mean(epoch_loss_eval_mae)))
        losses_mae.append(np.mean(epoch_loss_eval_mae))

        if np.mean(epoch_loss_eval_mae) <= min(losses_mae):
            torch.save(model.state_dict(), config.model_folder + '/gas_production_best_model_loss.pth')
