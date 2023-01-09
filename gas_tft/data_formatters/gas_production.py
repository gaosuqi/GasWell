# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：tft
@File    ：gas_production.py
@IDE     ：PyCharm
@Author  ：LXW
@Date    ：2022/12/2 19:51
'''

# Custom formatting functions for Favorita outputs.
# Defines outputs specific column definitions and data transformations.

from gas_tft.data_formatters import utils as utils
import pandas as pd
import sklearn.preprocessing
import datetime
import gas_tft.data_formatters.base

DataTypes = gas_tft.data_formatters.base.DataTypes
InputTypes = gas_tft.data_formatters.base.InputTypes


class GasProductionFormatter(gas_tft.data_formatters.base.GenericDataFormatter):
    """Defines and formats data for the GasProduction outputs.

    Attributes:
      column_definition: Defines input and data type of column used in the
        experiment.
      identifiers: Entity identifiers used in experiments.
    """

    _column_definition = [
        ('WellNo', DataTypes.CATEGORICAL, InputTypes.ID),
        ('Date', DataTypes.DATE, InputTypes.TIME),
        ('Daily_104m3', DataTypes.REAL_VALUED, InputTypes.TARGET),
        # ('Daily_h', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('WellHeadPressure', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('CasingHeadPressure', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('WellHeadTemperature', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('Daily_h', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('ElapsedProduction', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('Allocation', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT)

    ]

    def __init__(self):
        """Initialises formatter."""

        super().__init__()
        self.identifiers = None
        self._real_scalers = None
        self.target_scaler = None

    def split_data(self, df, valid_boundary=None):
        """Splits data frame into training-validation-test data frames.

        This also calibrates scaling object, and transforms data for each split.

        Args:
          df: Source data frame to split.
          valid_boundary: Starting year for validation data

        Returns:
          Tuple of transformed (train, valid, test, train_and_val) data.
        """

        fixed_params = self.get_fixed_params()
        time_steps = fixed_params['total_time_steps']
        lookback = fixed_params['num_encoder_steps']
        forecast_horizon = time_steps - lookback

        if valid_boundary is None:
            valid_boundary = pd.datetime(2022, 8, 15) - datetime.timedelta(days=2*forecast_horizon)

        # df['Date'] = pd.to_datetime(df['Date'])
        df_lists = {'train': [], 'valid': [], 'test': [], 'train_and_val': []}
        for _, sliced in df.groupby('WellNo'):
            index = pd.to_datetime(sliced['Date'])
            train = sliced.loc[index <= valid_boundary]
            train_len = len(train)
            valid_len = train_len + forecast_horizon
            if train_len >= time_steps:
                valid = sliced.iloc[train_len - lookback:valid_len, :]
                train_and_val = sliced.iloc[:valid_len, :]
                test = sliced.iloc[valid_len - lookback:valid_len + forecast_horizon, :]
                df_lists['train'].append(train)
                df_lists['valid'].append(valid)
                df_lists['train_and_val'].append(train_and_val)
                df_lists['test'].append(test)

        dfs = {k: pd.concat(df_lists[k], axis=0) for k in df_lists}

        return dfs['train'], dfs['valid'], dfs['test'], dfs['train_and_val']

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.

        Label encoding is applied to the entire outputs (i.e. including test),
        so that unseen labels can be handled at run-time.

        Args:
          df: Data to use to calibrate scalers.
        """
        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                       column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                           column_definitions)

        # Extract identifiers in case required
        self.identifiers = list(df[id_column].unique())

        # Format real scalers
        self._real_scalers = {}
        for col in ['Daily_104m3', 'Daily_h', 'Allocation']:
            self._real_scalers[col] = (df[col].mean(), df[col].std())

        self.target_scaler = (df[target_column].mean(), df[target_column].std())

    def transform_inputs(self, df):
        """Performs feature transformations.

        This includes both feature engineering, preprocessing and normalisation.

        Args:
          df: Data frame to transform.

        Returns:
          Transformed data frame.

        """
        output = df.copy()

        if self._real_scalers is None:
            raise ValueError('Scalers have not been set!')

        # Format real inputs
        for col in ['Daily_104m3', 'Daily_h', 'Allocation']:
            mean, std = self._real_scalers[col]
            output[col] = (df[col] - mean) / std

            if col == 'Daily_104m3':
                output[col] = output[col].fillna(0.)  # mean imputation

        return output

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.

        Args:
          predictions: Dataframe of model predictions.

        Returns:
          Data frame of unnormalised predictions.
        """
        output = predictions.copy()

        column_names = predictions.columns
        mean, std = self.target_scaler
        for col in column_names:
            if col not in {'forecast_time', 'identifier'}:
                output[col] = (predictions[col] * std) + mean

        return output

    # Default params
    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps': 120,
            'num_encoder_steps': 75,
            'num_epochs': 50,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 8
        }

        return fixed_params

    @property
    def get_default_model_params(self):
        """Returns default optimised model parameters."""

        model_params = {
            'dropout_rate': 0.1,
            'hidden_layer_size': 240,
            'learning_rate': 0.001,
            'minibatch_size': 128,
            'max_gradient_norm': 10,
            'num_heads': 1,
            'stack_size': 1
        }

        return model_params

    def get_mean_std(self):
        mean, std = self.target_scaler
        return mean, std

    def get_column_definition(self):
        """"Formats column definition in order expected by the TFT.

        Modified for gaswell to match column order of original experiment.

        Returns:
          gaswell-specific column definition
        """

        column_definition = self._column_definition

        # Sanity checks first.
        # Ensure only one ID and time column exist
        return column_definition
