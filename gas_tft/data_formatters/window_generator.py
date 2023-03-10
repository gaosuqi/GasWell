import time

import numpy as np
from torch.utils.data import Dataset

from gas_tft.data_formatters import utils as utils
from gas_tft.data_formatters.base import InputTypes


class TSDataset(Dataset):
    # Mostly adapted from original TFT Github, data_formatters
    def __init__(self, params, data, num_samples=-1):

        self.time_steps = int(params['total_time_steps'])
        self.input_size = int(params['input_size'])
        self.output_size = int(params['output_size'])
        self.num_encoder_steps = int(params['num_encoder_steps'])
        self.column_definition = params['column_definition']

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        input_cols = [
            tup[0]
            for tup in self.column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        data.sort_values(by=[id_col, time_col], inplace=True)
        valid_sampling_locations = []
        split_data_map = {}
        for identifier, df in data.groupby(id_col):
            num_entries = len(df)
            if num_entries >= self.time_steps:
                valid_sampling_locations += [
                    (identifier, i)
                    for i in range(num_entries - self.time_steps + 1)
                    # if list(df['Tmp_time'].items())[i + 28][1] > 0 and list(df['Tmp_time'].items())[i + 27][1] == 0
                    # if df[target_col][i + self.num_encoder_steps:i + self.time_steps].max() > 0.1
                    # 清除异常时间序列模式
                ]
            split_data_map[identifier] = df

        if 0 < num_samples < len(valid_sampling_locations):
            ranges = [
                valid_sampling_locations[i] for i in np.random.choice(
                    len(valid_sampling_locations), num_samples, replace=False)
            ]
        else:
            ranges = valid_sampling_locations
            num_samples = len(valid_sampling_locations)

        self.inputs = np.zeros((num_samples, self.time_steps, self.input_size))
        self.outputs = np.zeros((num_samples, self.time_steps, self.output_size))
        self.time = np.empty((num_samples, self.time_steps, 1), dtype=object)
        self.identifiers = np.empty((num_samples, self.time_steps, 1), dtype=object)

        for i, tup in enumerate(ranges):
            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx:start_idx + self.time_steps]
            self.inputs[i, :, :] = sliced[input_cols]
            self.outputs[i, :, :] = sliced[[target_col]]
            self.time[i, :, 0] = sliced[time_col]
            self.identifiers[i, :, 0] = sliced[id_col]

        self.sampled_data = {
            'inputs': self.inputs,
            'outputs': self.outputs[:, self.num_encoder_steps:, :],
            'active_entries': np.ones_like(self.outputs[:, self.num_encoder_steps:, :]),
            'time': self.time,
            'identifier': self.identifiers
        }

    def __getitem__(self, index):
        s = {
            'inputs': self.inputs[index],
            'outputs': self.outputs[index, self.num_encoder_steps:, :],
            'active_entries': np.ones_like(self.outputs[index, self.num_encoder_steps:, :]),
            'time': self.time[index].tolist(),
            'identifier': self.identifiers[index].tolist()
        }

        return s

    def __len__(self):
        return self.inputs.shape[0]

    def _get_single_col_by_type(self, input_type):
        """Returns name of single column for input type."""
        return utils.get_single_col_by_input_type(input_type,
                                                  self.column_definition)
